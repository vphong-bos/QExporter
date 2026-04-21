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
"""

import argparse

from extractors import MODEL_REGISTRY, create_extractor


def main():
    supported = ", ".join(sorted(MODEL_REGISTRY.keys())) + ", generic"

    parser = argparse.ArgumentParser(
        description="Extract .encodings or .pt outputs from QDQ ONNX, QOP ONNX, or existing encodings files.",
    )
    parser.add_argument("model", help=f"Model name ({supported})")
    parser.add_argument("ckpt_path", help="Path to the source ONNX or encodings file")
    parser.add_argument("output_path", help="Path to save the extracted output (.encodings/.json or .pt/.pth)")
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
    args = parser.parse_args()

    extractor_kwargs = {}
    if args.model == "vit":
        extractor_kwargs["approximate_attention_qparams"] = (
            args.vit_attention_approx or not args.no_vit_attention_approx
        )
        extractor_kwargs["attention_head_dim"] = args.vit_attention_head_dim

    extractor = create_extractor(
        args.ckpt_path,
        model_name=args.model,
        source_format=args.source_format,
        **extractor_kwargs,
    )
    extractor.save(args.output_path, output_format=args.output_format)


if __name__ == "__main__":
    main()
