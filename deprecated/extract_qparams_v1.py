#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

import onnx
import torch
from onnx import numpy_helper


class QuantizedOnnxExtractor:
    WEIGHT_Q_SUFFIX = ".weight_q"
    BIAS_Q_SUFFIX = ".bias_q"

    PASSTHROUGH_OPS = {
        "Transpose",
        "Reshape",
        "Identity",
        "Cast",
        "Flatten",
        "Squeeze",
        "Unsqueeze",
    }

    COMPUTE_OPS = {"Conv", "MatMul", "Gemm"}

    ACTIVATION_ONLY_OPS = {"Relu", "MaxPool", "AveragePool", "GlobalAveragePool"}

    def __init__(self, ckpt_path):
        self.ckpt_path = str(ckpt_path)
        self.onnx_model = onnx.load(self.ckpt_path)
        self.initializers = self._build_initializer_map()
        self.producer, self.consumers = self._build_graph_maps()

    @staticmethod
    def _to_torch_tensor(initializer):
        return torch.from_numpy(numpy_helper.to_array(initializer).copy())

    def _build_initializer_map(self):
        return {
            initializer.name: initializer
            for initializer in self.onnx_model.graph.initializer
        }

    @staticmethod
    def _normalize_prefix(name):
        while name.startswith("model."):
            name = name[len("model.") :]

        while name.startswith("module."):
            name = name[len("module.") :]

        while name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]

        replacements = [
            ("patch_embeddings.", "patch_embed."),
            ("weight_zeropoint", "weight_zero_point"),
            ("bias_zeropoint", "bias_zero_point"),
            ("input_zeropoint", "input_zero_point"),
            ("output_zeropoint", "output_zero_point"),
        ]

        for src, dst in replacements:
            name = name.replace(src, dst)

        return name

    def _build_graph_maps(self):
        producer = {}
        consumers = defaultdict(list)

        for node in self.onnx_model.graph.node:
            for output_name in node.output:
                producer[output_name] = node
            for input_name in node.input:
                consumers[input_name].append(node)

        return producer, consumers

    def _find_quantized_prefixes(self):
        return sorted(
            initializer.name[: -len(self.WEIGHT_Q_SUFFIX)]
            for initializer in self.onnx_model.graph.initializer
            if initializer.name.endswith(self.WEIGHT_Q_SUFFIX)
        )

    def _find_compute_node_from_weight_qdq(self, weight_tensor_name):
        queue = [weight_tensor_name]
        seen_tensors = set()

        while queue:
            tensor_name = queue.pop(0)
            if tensor_name in seen_tensors:
                continue
            seen_tensors.add(tensor_name)

            for consumer in self.consumers.get(tensor_name, []):
                if consumer.op_type in self.COMPUTE_OPS:
                    return consumer
                if consumer.op_type in self.PASSTHROUGH_OPS:
                    queue.extend(consumer.output)

        return None

    @staticmethod
    def _find_activation_input_name(compute_node, weight_prefix):
        weight_roots = {
            f"{weight_prefix}.weight_qdq",
            f"{weight_prefix}.weight_q",
            f"{weight_prefix}.weight",
        }

        for input_name in compute_node.input:
            if any(input_name.startswith(root) for root in weight_roots):
                continue
            if f"{weight_prefix}.weight" in input_name:
                continue
            return input_name

        if compute_node.op_type == "Conv":
            return compute_node.input[0]
        if len(compute_node.input) >= 2:
            return compute_node.input[0]
        return None

    @staticmethod
    def _to_python(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        return value

    @staticmethod
    def _qparams_to_aimet_encoding(scale_tensor, zeropoint_tensor):
        scale = QuantizedOnnxExtractor._to_python(scale_tensor)
        zeropoint = QuantizedOnnxExtractor._to_python(zeropoint_tensor)

        def one_encoding(s, zp):
            zp = int(zp)
            s = float(s)

            if zp == 0:
                dtype = "int"
                qmin, qmax = -128, 127
                is_symmetric = "True"
            elif zp == 128:
                dtype = "uint"
                qmin, qmax = 0, 255
                is_symmetric = "True"
            else:
                dtype = "uint"
                qmin, qmax = 0, 255
                is_symmetric = "False"

            offset = -zp
            enc_min = (qmin - zp) * s
            enc_max = (qmax - zp) * s

            return {
                "bitwidth": 8,
                "dtype": dtype,
                "is_symmetric": is_symmetric,
                "max": float(enc_max),
                "min": float(enc_min),
                "offset": int(offset),
                "scale": float(s),
            }

        if isinstance(scale, list):
            if not isinstance(zeropoint, list):
                zeropoint = [zeropoint] * len(scale)
            return [one_encoding(s, zp) for s, zp in zip(scale, zeropoint)]

        return one_encoding(scale, zeropoint)

    def _extract_qparams_from_dq_tensor(self, tensor_name):
        dq_node = self.producer.get(tensor_name)
        if dq_node is None or dq_node.op_type != "DequantizeLinear" or len(dq_node.input) < 3:
            return None

        scale_name = dq_node.input[1]
        zero_point_name = dq_node.input[2]

        if scale_name not in self.initializers or zero_point_name not in self.initializers:
            return None

        return {
            "scale": self._to_torch_tensor(self.initializers[scale_name]),
            "zeropoint": self._to_torch_tensor(self.initializers[zero_point_name]),
        }

    def _find_output_quant_params(self, prefix, compute_node):
        downstream_ops = {
            "BatchNormalization", "Relu", "Clip", "Add", "Transpose",
            "Reshape", "Identity", "Cast", "Flatten", "Squeeze",
            "Unsqueeze", "GlobalAveragePool", "AveragePool", "MaxPool",
        }

        output_tensor_name = compute_node.output[0]

        if compute_node.op_type in {"MatMul", "Gemm"}:
            for consumer in self.consumers.get(output_tensor_name, []):
                if consumer.op_type != "Add":
                    continue
                if any(input_name == f"{prefix}.bias_qdq" for input_name in consumer.input):
                    output_tensor_name = consumer.output[0]
                    break

        queue = [output_tensor_name]
        seen_tensors = set()

        while queue:
            tensor_name = queue.pop(0)
            if tensor_name in seen_tensors:
                continue
            seen_tensors.add(tensor_name)

            for consumer in self.consumers.get(tensor_name, []):
                if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                    scale_name = consumer.input[1]
                    zero_point_name = consumer.input[2]

                    if scale_name in self.initializers and zero_point_name in self.initializers:
                        return {
                            "scale": self._to_torch_tensor(self.initializers[scale_name]),
                            "zeropoint": self._to_torch_tensor(self.initializers[zero_point_name]),
                        }
                elif consumer.op_type in downstream_ops:
                    queue.extend(consumer.output)

        return None

    @staticmethod
    def _onnx_node_name_to_module(node_name):
        name = node_name.lstrip('/')
        parts = name.split('/')
        if len(parts) > 1:
            parts = parts[:-1]
        filtered = []
        for i, part in enumerate(parts):
            if i + 1 < len(parts) and parts[i + 1].startswith(part + "."):
                continue
            filtered.append(part)
        return '.'.join(filtered)

    def _collect_activation_only_encodings(self):
        encodings = {}

        for node in self.onnx_model.graph.node:
            if node.op_type != "Relu":
                continue

            module_name = self._onnx_node_name_to_module(node.name)

            is_initial_relu = (module_name == "relu")
            is_post_add_relu = module_name.endswith(".relu_2")
            if not is_initial_relu and not is_post_add_relu:
                continue

            if is_post_add_relu:
                module_name = module_name[:-2]

            if node.output:
                output_tensor = node.output[0]
                for consumer in self.consumers.get(output_tensor, []):
                    if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                        scale_name = consumer.input[1]
                        zp_name = consumer.input[2]
                        if scale_name in self.initializers and zp_name in self.initializers:
                            encodings[module_name] = {
                                "output": {
                                    "0": self._qparams_to_aimet_encoding(
                                        self._to_torch_tensor(self.initializers[scale_name]),
                                        self._to_torch_tensor(self.initializers[zp_name]),
                                    )
                                }
                            }
                            break

        return encodings

    def collect_aimet_encodings(self):
        activation_encodings = {}
        param_encodings = {}
        missing_input_qparams = []
        missing_output_qparams = []

        for prefix in self._find_quantized_prefixes():
            export_prefix = self._normalize_prefix(prefix)

            weight_scale_name = f"{prefix}.weight_scale"
            weight_zero_point_name = f"{prefix}.weight_zero_point"

            if (
                weight_scale_name in self.initializers
                and weight_zero_point_name in self.initializers
            ):
                weight_scale = self._to_torch_tensor(self.initializers[weight_scale_name])
                weight_zp = self._to_torch_tensor(self.initializers[weight_zero_point_name])
                param_encodings[f"{export_prefix}.weight"] = self._qparams_to_aimet_encoding(
                    weight_scale, weight_zp
                )

            bias_scale_name = f"{prefix}.bias_scale"
            bias_zero_point_name = f"{prefix}.bias_zero_point"

            if (
                bias_scale_name in self.initializers
                and bias_zero_point_name in self.initializers
            ):
                bias_scale = self._to_torch_tensor(self.initializers[bias_scale_name])
                bias_zp = self._to_torch_tensor(self.initializers[bias_zero_point_name])
                param_encodings[f"{export_prefix}.bias"] = self._qparams_to_aimet_encoding(
                    bias_scale, bias_zp
                )

            compute_node = self._find_compute_node_from_weight_qdq(f"{prefix}.weight_qdq")
            if compute_node is None:
                missing_input_qparams.append(prefix)
                missing_output_qparams.append(prefix)
                continue

            layer_act = {}

            activation_input_name = self._find_activation_input_name(compute_node, prefix)
            input_qparams = None
            if activation_input_name is not None:
                input_qparams = self._extract_qparams_from_dq_tensor(activation_input_name)

            if input_qparams is not None:
                layer_act["input"] = {
                    "0": self._qparams_to_aimet_encoding(
                        input_qparams["scale"], input_qparams["zeropoint"]
                    )
                }
            else:
                missing_input_qparams.append(prefix)

            output_qparams = self._find_output_quant_params(prefix, compute_node)
            if output_qparams is not None:
                layer_act["output"] = {
                    "0": self._qparams_to_aimet_encoding(
                        output_qparams["scale"], output_qparams["zeropoint"]
                    )
                }
            else:
                missing_output_qparams.append(prefix)

            if layer_act:
                activation_encodings[export_prefix] = layer_act

        activation_encodings.update(self._collect_activation_only_encodings())

        return {
            "activation_encodings": activation_encodings,
            "param_encodings": param_encodings,
        }, missing_input_qparams, missing_output_qparams

    def save(self, output_path):
        encodings, missing_input_qparams, missing_output_qparams = self.collect_aimet_encodings()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(encodings, f, indent=2)

        print(f"Saved encodings to {output_path}")
        print(f"Layers without explicit input activation qparams: {len(missing_input_qparams)}")
        for prefix in missing_input_qparams:
            print(f"  - {prefix}")
        print(f"Layers without explicit output activation qparams: {len(missing_output_qparams)}")
        for prefix in missing_output_qparams:
            print(f"  - {prefix}")

        return encodings

def main():
    parser = argparse.ArgumentParser(
        description="Extract quantized tensors and activation qparams from a QDQ ONNX model."
    )
    parser.add_argument("ckpt_path", help="Path to the quantized ONNX model")
    parser.add_argument("output_path", help="Path to save the extracted torch state_dict")
    args = parser.parse_args()

    extractor = QuantizedOnnxExtractor(args.ckpt_path)
    extractor.save(args.output_path)


if __name__ == "__main__":
    main()
