#!/usr/bin/env python3
"""Extract quantization data from ONNX or existing encodings files.

Usage:
    # QDQ ONNX -> encodings
    python run_extract.py resnet50 model.onnx output.encodings.json

    # QDQ ONNX -> pt
    python run_extract.py generic model.onnx output.pt

    # QOP ONNX -> pt
    python run_extract.py generic model_qop.onnx output.pt --source-format qop

    # encodings -> encodings
    python run_extract.py generic input.encodings output.encodings

    # SSR encodings + exported weights -> pt
    python run_extract.py ssr model.encodings output.pt --weights-path model.pth --config ssr/projects/configs/SSR_e2e.py

    # Analyze operator usage / quantization coverage
    python run_extract.py generic model.onnx --mode analyze
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import onnx

from extractors import MODEL_REGISTRY, create_extractor, detect_source_format


QUANT_OP_TYPES_EXACT = {
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearMul",
    "QLinearAveragePool",
    "QLinearGlobalAveragePool",
    "QLinearLeakyRelu",
    "QLinearSigmoid",
    "QLinearSoftmax",
    "ConvInteger",
    "MatMulInteger",
}

QUANT_OP_KEYWORDS = (
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinear",
    "Integer",
)

QDQ_BOUNDARY_OPS = {"QuantizeLinear", "DequantizeLinear"}


def is_quant_op(op_type):
    if op_type in QUANT_OP_TYPES_EXACT:
        return True
    return any(keyword in op_type for keyword in QUANT_OP_KEYWORDS)


def build_graph_maps(model):
    producer = {}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for output_name in node.output:
            producer[output_name] = node
        for input_name in node.input:
            consumers[input_name].append(node)
    return producer, consumers


def collect_op_counts(model):
    op_counts = {}
    quant_op_counts = {}
    non_quant_op_counts = {}
    node_rows = []

    total_nodes = len(model.graph.node)
    quant_node_count = 0

    for index, node in enumerate(model.graph.node):
        op_type = node.op_type
        quant_flag = is_quant_op(op_type)

        op_counts[op_type] = op_counts.get(op_type, 0) + 1
        if quant_flag:
            quant_op_counts[op_type] = quant_op_counts.get(op_type, 0) + 1
            quant_node_count += 1
        else:
            non_quant_op_counts[op_type] = non_quant_op_counts.get(op_type, 0) + 1

        node_rows.append(
            {
                "index": index,
                "name": node.name,
                "op_type": op_type,
                "is_quant_op": quant_flag,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
        )

    return {
        "node_count": total_nodes,
        "quant_node_count": quant_node_count,
        "non_quant_node_count": total_nodes - quant_node_count,
        "quant_node_ratio": (float(quant_node_count) / float(total_nodes)) if total_nodes > 0 else 0.0,
        "op_counts": dict(sorted(op_counts.items())),
        "quant_op_counts": dict(sorted(quant_op_counts.items())),
        "non_quant_op_counts": dict(sorted(non_quant_op_counts.items())),
        "all_ops": sorted(op_counts),
        "quant_ops": sorted(quant_op_counts),
        "non_quant_ops": sorted(non_quant_op_counts),
        "nodes": node_rows,
    }


def analyze_qdq_fake_quantized_ops(model):
    producer, consumers = build_graph_maps(model)
    quantized_float_tensors = set()
    fake_quantized_nodes = []
    fake_quantized_op_counts = {}

    for node in model.graph.node:
        if node.op_type == "DequantizeLinear":
            quantized_float_tensors.update(node.output)

    changed = True
    while changed:
        changed = False
        for index, node in enumerate(model.graph.node):
            if node.op_type in QDQ_BOUNDARY_OPS:
                continue

            if any(input_name in quantized_float_tensors for input_name in node.input):
                if not any(
                    existing["index"] == index for existing in fake_quantized_nodes
                ):
                    fake_quantized_nodes.append(
                        {
                            "index": index,
                            "name": node.name,
                            "op_type": node.op_type,
                            "inputs": list(node.input),
                            "outputs": list(node.output),
                            "input_from_dequantized_region": True,
                            "has_output_quantizer": any(
                                consumer.op_type == "QuantizeLinear"
                                for output_name in node.output
                                for consumer in consumers.get(output_name, [])
                            ),
                        }
                    )
                    fake_quantized_op_counts[node.op_type] = fake_quantized_op_counts.get(node.op_type, 0) + 1

                new_outputs = [output_name for output_name in node.output if output_name not in quantized_float_tensors]
                if new_outputs:
                    quantized_float_tensors.update(new_outputs)
                    changed = True

    fake_quantized_nodes.sort(key=lambda row: row["index"])
    return {
        "fake_quantized_node_count": len(fake_quantized_nodes),
        "fake_quantized_op_counts": dict(sorted(fake_quantized_op_counts.items())),
        "fake_quantized_ops": sorted(fake_quantized_op_counts),
        "fake_quantized_nodes": fake_quantized_nodes,
    }


def analyze_model(source_path, source_format):
    if Path(source_path).suffix.lower() != ".onnx":
        raise ValueError("Analyze mode currently supports ONNX inputs only.")

    model = onnx.load(str(source_path), load_external_data=False)
    result = {
        "source_path": str(source_path),
        "source_format": source_format,
        "op_inventory": collect_op_counts(model),
    }

    if source_format == "qdq":
        result["qdq_analysis"] = analyze_qdq_fake_quantized_ops(model)

    return result


def save_analysis(result, output_path=None):
    payload = json.dumps(result, indent=2)
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(payload + "\n", encoding="utf-8")
        print(f"Saved analysis to {output_file}")
    else:
        print(payload)


def main():
    supported = ", ".join(sorted(MODEL_REGISTRY.keys())) + ", generic"

    parser = argparse.ArgumentParser(
        description="Extract .encodings or .pt outputs from QDQ ONNX, QOP ONNX, or existing encodings files.",
    )
    parser.add_argument("model", help=f"Model name ({supported})")
    parser.add_argument("ckpt_path", help="Path to the source ONNX or encodings file")
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Path to save the extracted output (.encodings/.json or .pt/.pth). "
             "Optional in --mode analyze; if omitted, analysis is printed.",
    )
    parser.add_argument(
        "--mode",
        choices=("extract", "analyze"),
        default="extract",
        help="Run extraction or ONNX operator analysis.",
    )
    parser.add_argument(
        "--source-format",
        choices=("qdq", "qop", "encodings"),
        default=None,
        help="Source format. Defaults to auto-detection from the input file.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("encodings", "pt"),
        default=None,
        help="Output format. Defaults to inferring from output_path suffix.",
    )
    parser.add_argument(
        "--vit-attention-approx",
        action="store_true",
        help="For ViT .pt export, synthesize missing internal attention qparams "
             "(matmul_qk / softmax / matmul_pv input) from attention-local heuristics.",
    )
    parser.add_argument(
        "--no-vit-attention-approx",
        action="store_true",
        help="Disable synthesized internal ViT attention qparams for .pt export.",
    )
    parser.add_argument(
        "--vit-attention-head-dim",
        type=int,
        default=64,
        help="Head dimension used by --vit-attention-approx when approximating matmul_qk output scale.",
    )
    parser.add_argument(
        "--weights-path",
        default=None,
        help="Optional weights checkpoint used when exporting .pt from AIMET encodings.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Model config used when rebuilding a full torch model from encodings.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional AIMET quantsim config passed through to QuantizedSSR model import.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used when rebuilding a full torch model from encodings.",
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Apply mmcv fuse_conv_bn when rebuilding a full torch model from encodings.",
    )
    parser.add_argument(
        "--enable-bn-fold",
        action="store_true",
        help="Enable AIMET BN fold when rebuilding a full torch model from encodings.",
    )
    args = parser.parse_args()

    if args.mode == "extract" and not args.output_path:
        parser.error("output_path is required in --mode extract")

    extractor_kwargs = {}
    if args.model == "vit":
        extractor_kwargs["approximate_attention_qparams"] = (
            args.vit_attention_approx or not args.no_vit_attention_approx
        )
        extractor_kwargs["attention_head_dim"] = args.vit_attention_head_dim
    # extractor_kwargs["weights_path"] = args.weights_path
    # extractor_kwargs["config"] = args.config
    # extractor_kwargs["config_path"] = args.config_path
    # extractor_kwargs["device"] = args.device
    # extractor_kwargs["fuse_conv_bn"] = args.fuse_conv_bn
    # extractor_kwargs["enable_bn_fold"] = args.enable_bn_fold

    if args.mode == "analyze":
        source_format = args.source_format or detect_source_format(args.ckpt_path)
        result = analyze_model(args.ckpt_path, source_format)
        save_analysis(result, args.output_path)
        return

    extractor = create_extractor(
        args.ckpt_path,
        model_name=args.model,
        source_format=args.source_format,
        **extractor_kwargs,
    )
    extractor.save(args.output_path, output_format=args.output_format)


if __name__ == "__main__":
    main()
