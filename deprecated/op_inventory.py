#!/usr/bin/env python3
import argparse
import json
import os

import onnx

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate quantized model op inventory and statistics."
    )

    parser.add_argument(
        "--quant_weights",
        type=str,
        required=True,
        help="Path to the int8 quantized model artifact (ONNX).",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fp32_weights",
        type=str,
        help="Path to FP32 torch weights.",
    )
    group.add_argument(
        "--fp32_onnx",
        type=str,
        help="Path to FP32 ONNX model.",
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save JSON output.",
    )

    return parser.parse_args()


def is_quant_op(op_type):
    if op_type in QUANT_OP_TYPES_EXACT:
        return True
    return any(keyword in op_type for keyword in QUANT_OP_KEYWORDS)


def collect_onnx_op_inventory(model_path):
    model = onnx.load(model_path, load_external_data=False)

    op_counts = {}
    quant_op_counts = {}
    non_quant_op_counts = {}
    node_rows = []

    total_nodes = len(model.graph.node)
    quant_node_count = 0

    for idx, node in enumerate(model.graph.node):
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

        quant_flag = is_quant_op(op_type)
        if quant_flag:
            quant_op_counts[op_type] = quant_op_counts.get(op_type, 0) + 1
            quant_node_count += 1
        else:
            non_quant_op_counts[op_type] = non_quant_op_counts.get(op_type, 0) + 1

        node_rows.append({
            "index": idx,
            "name": node.name,
            "op_type": op_type,
            "is_quant_op": quant_flag,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })

    return {
        "node_count": total_nodes,
        "quant_node_count": quant_node_count,
        "non_quant_node_count": total_nodes - quant_node_count,
        "quant_node_ratio": (float(quant_node_count) / float(total_nodes)) if total_nodes > 0 else 0.0,
        "op_counts": dict(sorted(op_counts.items(), key=lambda x: x[0])),
        "quant_op_counts": dict(sorted(quant_op_counts.items(), key=lambda x: x[0])),
        "non_quant_op_counts": dict(sorted(non_quant_op_counts.items(), key=lambda x: x[0])),
        "all_ops": sorted(op_counts.keys()),
        "quant_ops": sorted(quant_op_counts.keys()),
        "non_quant_ops": sorted(non_quant_op_counts.keys()),
        "nodes": node_rows,
    }


def build_summary(op_inventory):
    return {
        "node_count": op_inventory["node_count"],
        "quant_node_count": op_inventory["quant_node_count"],
        "non_quant_node_count": op_inventory["non_quant_node_count"],
        "quant_node_ratio": op_inventory["quant_node_ratio"],
        "unique_ops": len(op_inventory["all_ops"]),
        "unique_quant_ops": len(op_inventory["quant_ops"]),
        "unique_non_quant_ops": len(op_inventory["non_quant_ops"]),
    }


def main():
    args = parse_args()

    if not os.path.exists(args.quant_weights):
        raise FileNotFoundError(f"Quant weights not found: {args.quant_weights}")

    if args.fp32_weights and args.fp32_onnx:
        raise ValueError("Specify only one of --fp32_weights or --fp32_onnx.")

    quant_ext = os.path.splitext(args.quant_weights)[1].lower()
    if quant_ext != ".onnx":
        raise ValueError("Quant weights must be an ONNX artifact for op inventory analysis.")

    op_inventory = collect_onnx_op_inventory(args.quant_weights)
    result = {
        "quant_weights": os.path.basename(args.quant_weights),
        "op_inventory": op_inventory,
        "op_statistics": build_summary(op_inventory),
    }

    if args.fp32_weights:
        result["fp32_weights"] = os.path.basename(args.fp32_weights)
    elif args.fp32_onnx:
        result["fp32_onnx"] = os.path.basename(args.fp32_onnx)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Results saved to {args.output_json}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()