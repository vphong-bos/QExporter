from pathlib import Path

import onnx

from .base import QuantizedOnnxExtractor
from .encodings import EncodingsExtractor
from .qop import QOPExtractor
from .model.resnet import ResNetExtractor
from .model.ssr import SSRExtractor
from .model.pdl import PDLExtractor
from .model.vit import ViTExtractor

MODEL_REGISTRY = {
    "resnet50": ResNetExtractor,
    "resnet101": ResNetExtractor,
    "resnet152": ResNetExtractor,
    "ssr": SSRExtractor,
    "pdl": PDLExtractor,
    "vit": ViTExtractor,
}


def get_extractor(model_name):
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        supported = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_name}'. Supported: {supported}")
    return cls


def detect_source_format(source_path):
    source_path = Path(source_path)
    suffix = source_path.suffix.lower()
    if suffix in {".encodings", ".json"}:
        return "encodings"
    if suffix != ".onnx":
        raise ValueError(f"Unable to infer source format from: {source_path}")

    model = onnx.load(str(source_path))
    op_types = {node.op_type for node in model.graph.node}
    if (
        any(op_type.startswith("QLinear") for op_type in op_types)
        or {"ConvInteger", "MatMulInteger"} & op_types
    ):
        return "qop"
    if {"QuantizeLinear", "DequantizeLinear"} & op_types:
        return "qdq"
    raise ValueError(f"Unable to detect ONNX quantization format for: {source_path}")


def create_extractor(source_path, model_name="generic", source_format=None, **extractor_kwargs):
    source_format = source_format or detect_source_format(source_path)

    if source_format == "encodings":
        return EncodingsExtractor(source_path)
    if source_format == "qop":
        if model_name != "generic":
            print(f"Ignoring model-specific extractor '{model_name}' for qop source format.")
        return QOPExtractor(source_path)
    if source_format == "qdq":
        if model_name == "generic":
            return QuantizedOnnxExtractor(source_path)
        return get_extractor(model_name)(source_path, **extractor_kwargs)

    raise ValueError(f"Unsupported source format: {source_format}")
