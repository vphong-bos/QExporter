"""QDQ ONNX extractor producing AIMET-compatible encodings or torch state."""

from collections import defaultdict

import onnx
import torch
from onnx import numpy_helper

from .base import BaseExtractor


class QuantizedOnnxExtractor(BaseExtractor):
    WEIGHT_Q_SUFFIX = ".weight_q"
    BIAS_Q_SUFFIX = ".bias_q"
    RUNTIME_INITIALIZERS = {"input_scale", "input_zero_point"}

    PASSTHROUGH_OPS = {
        "Transpose", "Reshape", "Identity", "Cast",
        "Flatten", "Squeeze", "Unsqueeze",
    }

    COMPUTE_OPS = {"Conv", "MatMul", "Gemm"}
    ACTIVATION_ONLY_OPS = {
        "Relu",
        "LeakyRelu",
        "Sigmoid",
        "Softmax",
        "Add",
        "Mul",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
    }

    def __init__(self, ckpt_path, compute_ops=None, passthrough_ops=None):
        super().__init__(ckpt_path)
        self.ckpt_path = str(ckpt_path)
        self.onnx_model = onnx.load(self.ckpt_path)
        self.initializers = self._build_initializer_map()
        self.producer, self.consumers = self._build_graph_maps()
        self.compute_ops = set(compute_ops) if compute_ops is not None else set(self.COMPUTE_OPS)
        self.passthrough_ops = set(passthrough_ops) if passthrough_ops is not None else set(self.PASSTHROUGH_OPS)

    @staticmethod
    def _to_torch_tensor(initializer):
        return torch.from_numpy(numpy_helper.to_array(initializer).copy())

    @staticmethod
    def _dequantize_tensor(q_tensor, scale_tensor, zero_point_tensor):
        q_tensor = q_tensor.to(torch.float32)
        scale_tensor = scale_tensor.to(torch.float32)
        zero_point_tensor = zero_point_tensor.to(torch.float32)

        if scale_tensor.ndim == 1 and q_tensor.ndim > 1 and scale_tensor.shape[0] == q_tensor.shape[0]:
            view_shape = [q_tensor.shape[0]] + [1] * (q_tensor.ndim - 1)
            scale_tensor = scale_tensor.reshape(view_shape)
            zero_point_tensor = zero_point_tensor.reshape(view_shape)

        return (q_tensor - zero_point_tensor) * scale_tensor

    def _build_initializer_map(self):
        return {initializer.name: initializer for initializer in self.onnx_model.graph.initializer}

    def _build_graph_maps(self):
        producer = {}
        consumers = defaultdict(list)
        for node in self.onnx_model.graph.node:
            for out in node.output:
                producer[out] = node
            for inp in node.input:
                consumers[inp].append(node)
        return producer, consumers

    def _find_quantized_prefixes(self):
        return sorted(
            initializer.name[: -len(self.WEIGHT_Q_SUFFIX)]
            for initializer in self.onnx_model.graph.initializer
            if initializer.name.endswith(self.WEIGHT_Q_SUFFIX)
        )

    @staticmethod
    def _normalize_prefix(name):
        for wrapper in ("module.", "_orig_mod.", "model."):
            while name.startswith(wrapper):
                name = name[len(wrapper):]
        return name

    def _onnx_prefix_to_state_prefix(self, prefix):
        return self._normalize_prefix(prefix)

    def _onnx_name_to_state_key(self, name):
        if name.endswith(".weight") or name.endswith(".bias"):
            prefix, suffix = name.rsplit(".", 1)
            return f"{self._onnx_prefix_to_state_prefix(prefix)}.{suffix}"
        return self._onnx_prefix_to_state_prefix(name)

    def _matmul_node_name_to_state_key(self, node_name):
        if not node_name.endswith("/MatMul"):
            return None
        prefix = node_name[: -len("/MatMul")].strip("/").replace("/", ".")
        if not prefix:
            return None
        return f"{self._onnx_prefix_to_state_prefix(prefix)}.weight"

    def _is_runtime_initializer(self, name):
        return name.startswith("/") or name in self.RUNTIME_INITIALIZERS

    def _is_plain_model_initializer(self, name):
        if self._is_runtime_initializer(name):
            return False
        return name.endswith(".weight") or name.endswith(".bias")

    def _find_compute_node_from_weight_qdq(self, weight_tensor_name):
        queue = [weight_tensor_name]
        seen = set()
        while queue:
            tensor = queue.pop(0)
            if tensor in seen:
                continue
            seen.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type in self.compute_ops:
                    return consumer
                if consumer.op_type in self.passthrough_ops:
                    queue.extend(consumer.output)
        return None

    @staticmethod
    def _find_activation_input_name(compute_node, weight_prefix):
        weight_roots = {
            f"{weight_prefix}.weight_qdq",
            f"{weight_prefix}.weight_q",
            f"{weight_prefix}.weight",
        }
        for inp in compute_node.input:
            if any(inp.startswith(root) for root in weight_roots):
                continue
            return inp
        if compute_node.op_type == "Conv":
            return compute_node.input[0]
        if len(compute_node.input) >= 2:
            return compute_node.input[0]
        return None

    def _extract_qparams_from_dq_tensor(self, tensor_name):
        upstream_ops = {
            "Transpose", "Reshape", "Identity", "Cast", "Flatten",
            "Squeeze", "Unsqueeze", "GlobalAveragePool", "AveragePool",
            "MaxPool", "Relu", "Clip", "Concat",
            "ReduceMean", "ReduceSum", "Shape", "Gather",
            "Add", "Mul", "Sub", "Div", "Tile", "Expand",
            "Pad", "Slice", "Where", "Constant", "ConstantOfShape",
            "GridSampleBilinearZerosAC0", "ScatterND", "Softmax",
        }
        visited = set()
        queue = [tensor_name]
        while queue:
            name = queue.pop(0)
            if name in visited:
                continue
            visited.add(name)
            node = self.producer.get(name)
            if node is None:
                continue
            if node.op_type == "DequantizeLinear" and len(node.input) >= 3:
                scale_name, zero_point_name = node.input[1], node.input[2]
                if scale_name in self.initializers and zero_point_name in self.initializers:
                    return {
                        "scale": self._to_torch_tensor(self.initializers[scale_name]),
                        "zeropoint": self._to_torch_tensor(self.initializers[zero_point_name]),
                    }
            if node.op_type in upstream_ops:
                queue.extend(node.input)
        return None

    def _find_output_quant_params(self, prefix, compute_node):
        output_tensor = compute_node.output[0]
        if compute_node.op_type in {"MatMul", "Gemm"}:
            for consumer in self.consumers.get(output_tensor, []):
                if consumer.op_type == "Add" and any(inp == f"{prefix}.bias_qdq" for inp in consumer.input):
                    output_tensor = consumer.output[0]
                    break
        downstream_ops = {
            "BatchNormalization", "Relu", "LeakyRelu", "Clip", "Sigmoid", "Softmax",
            "Add", "Mul", "Transpose", "Reshape", "Identity", "Cast",
            "Flatten", "Squeeze", "Unsqueeze", "GlobalAveragePool", "AveragePool", "MaxPool",
        }
        return self._find_downstream_quant_params(output_tensor, downstream_ops)

    def _find_downstream_quant_params(self, tensor_name, downstream_ops):
        visited = set()
        queue = [tensor_name]
        while queue:
            tensor = queue.pop(0)
            if tensor in visited:
                continue
            visited.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                    scale_name, zero_point_name = consumer.input[1], consumer.input[2]
                    if scale_name in self.initializers and zero_point_name in self.initializers:
                        return {
                            "scale": self._to_torch_tensor(self.initializers[scale_name]),
                            "zeropoint": self._to_torch_tensor(self.initializers[zero_point_name]),
                        }
                if consumer.op_type in downstream_ops:
                    queue.extend(consumer.output)
        return None

    def _find_output_quant_params_via_weight_path(self, prefix):
        weight_tensor_name = f"{prefix}.weight_qdq"
        visited = set()
        queue = [weight_tensor_name]
        while queue:
            tensor = queue.pop(0)
            if tensor in visited:
                continue
            visited.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                    scale_name, zero_point_name = consumer.input[1], consumer.input[2]
                    if scale_name in self.initializers and zero_point_name in self.initializers:
                        return {
                            "scale": self._to_torch_tensor(self.initializers[scale_name]),
                            "zeropoint": self._to_torch_tensor(self.initializers[zero_point_name]),
                        }
                if consumer.op_type in self.passthrough_ops:
                    queue.extend(consumer.output)
        return None

    def _add_quantized_param_triplet(self, state_dict, state_prefix, param_name, onnx_prefix):
        q_name = f"{onnx_prefix}.{param_name}_q"
        scale_name = f"{onnx_prefix}.{param_name}_scale"
        zero_point_name = f"{onnx_prefix}.{param_name}_zero_point"

        if q_name not in self.initializers:
            return False

        q_tensor = self._to_torch_tensor(self.initializers[q_name])
        scale_tensor = self._to_torch_tensor(self.initializers[scale_name])
        zero_point_tensor = self._to_torch_tensor(self.initializers[zero_point_name])

        state_dict[f"{state_prefix}.{param_name}"] = self._dequantize_tensor(q_tensor, scale_tensor, zero_point_tensor)
        state_dict[f"{state_prefix}.{param_name}_q"] = q_tensor
        state_dict[f"{state_prefix}.{param_name}_scale"] = scale_tensor
        state_dict[f"{state_prefix}.{param_name}_zero_point"] = zero_point_tensor
        return True

    def _add_anonymous_matmul_weight(self, state_dict, initializer_name, initializer):
        if not initializer_name.startswith("onnx::MatMul_"):
            return False

        matmul_consumers = [consumer for consumer in self.consumers.get(initializer_name, []) if consumer.op_type == "MatMul"]
        if len(matmul_consumers) != 1:
            return False

        state_key = self._matmul_node_name_to_state_key(matmul_consumers[0].name)
        if state_key is None:
            return False

        weight = self._to_torch_tensor(initializer).to(torch.float32)
        if weight.ndim == 2:
            weight = weight.transpose(0, 1)
        state_dict[state_key] = weight
        return True

    # @staticmethod
    # def _onnx_node_name_to_module(node_name, op_type=None):
    #     name = node_name.strip("/")
    #     if not name:
    #         return ""

    #     parts = [part for part in name.split("/") if part]

    #     if op_type is not None and len(parts) > 1:
    #         last = parts[-1]
    #         if last == op_type or last.startswith(f"{op_type}_"):
    #             parts = parts[:-1]

    #     return ".".join(parts)

    @staticmethod
    def _onnx_node_name_to_module(node_name, op_type=None):
        metadata_props = getattr(node_name, "metadata_props", None)

        if metadata_props is not None:
            node = node_name

            metadata_scope = None
            for prop in metadata_props:
                if prop.key == "pkg.torch.onnx.name_scopes":
                    metadata_scope = prop.value
                    break

            if metadata_scope:
                # Metadata may be a Python-list-looking string:
                #   "['', 'model', 'model.topdown', ..., 'model.topdown.7.relu2', 'relu_38']"
                # Prefer [-2] because [-1] is often the generated ONNX node name.
                try:
                    import ast

                    scopes = ast.literal_eval(metadata_scope)
                    if isinstance(scopes, (list, tuple)):
                        scopes = [str(scope).strip() for scope in scopes if str(scope).strip()]
                        if len(scopes) >= 2:
                            name = scopes[-2]
                        elif len(scopes) == 1:
                            name = scopes[-1]
                        else:
                            name = getattr(node, "name", "")
                    else:
                        name = str(metadata_scope)
                except Exception:
                    # Fallback for non-list metadata strings.
                    name = str(metadata_scope)
            else:
                name = getattr(node, "name", "")

            if op_type is None:
                op_type = getattr(node, "op_type", None)

        else:
            name = node_name.name if hasattr(node_name, "name") else str(node_name)

        name = str(name).strip()
        name = name.strip("/")
        if not name:
            return ""

        # If non-list metadata contains multiple scopes, prefer [-2] if possible,
        # otherwise [-1]. This mirrors the list behavior above.
        for sep in ("\n", ";", ","):
            if sep in name:
                parts = [part.strip().strip("'\"") for part in name.split(sep) if part.strip()]
                if len(parts) >= 2:
                    name = parts[-2]
                elif parts:
                    name = parts[-1]
                break

        name = name.strip("/")
        if not name:
            return ""

        parts = [part for part in name.split("/") if part]

        # Only strip op_type if it is a separate slash-path tail.
        # Do not strip from single-component dotted names like frontend.node_add.
        if op_type is not None and len(parts) > 1:
            last = parts[-1]
            if last == op_type or last.startswith(f"{op_type}_"):
                parts = parts[:-1]

        return ".".join(parts)
    
    @staticmethod
    def _to_python(value):
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        return value

    @staticmethod
    def _qparams_to_aimet_encoding(scale_tensor, zeropoint_tensor):
        scale = QuantizedOnnxExtractor._to_python(scale_tensor)
        zeropoint = QuantizedOnnxExtractor._to_python(zeropoint_tensor)

        def one_encoding(scale_value, zero_point_value):
            zero_point_value, scale_value = int(zero_point_value), float(scale_value)
            if zero_point_value == 0 or zero_point_value == 128:
                zero_point_value = 128
                dtype, qmin, qmax, is_symmetric = "int", 0, 255, "True"
            else:
                dtype, qmin, qmax, is_symmetric = "int", -128, 128, "False"
            return {
                "bitwidth": 8,
                "dtype": dtype,
                "is_symmetric": is_symmetric,
                "max": float((qmax - zero_point_value) * scale_value),
                "min": float((qmin - zero_point_value) * scale_value),
                "offset": -zero_point_value,
                "scale": scale_value,
            }

        if isinstance(scale, list):
            if not isinstance(zeropoint, list):
                zeropoint = [zeropoint] * len(scale)
            return [one_encoding(scale_value, zero_point_value) for scale_value, zero_point_value in zip(scale, zeropoint)]
        return one_encoding(scale, zeropoint)

    def _get_activation_roles(self, export_prefix):
        return {"input", "output"}

    def _get_torch_activation_roles(self, export_prefix):
        """Return activation qparam roles for .pt export.

        .pt exports are meant to preserve as much activation qparam information
        as possible for downstream comparison/debugging, so the default behavior
        is to emit both input and output qparams even if the encodings export is
        more selective. Model-specific extractors can still override this when
        the target .pt naming/layout differs.
        """
        return {"input", "output"}

    def _postprocess_torch_state(self, state_dict):
        """Allow model-specific .pt export cleanup or augmentation."""
        return state_dict

    @staticmethod
    def _dedupe_preserve_order(values):
        seen = set()
        result = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _collect_activation_only_encodings(self):
        encodings = {}
        for node in self.onnx_model.graph.node:
            if node.op_type not in self.ACTIVATION_ONLY_OPS:
                continue
            # module_name = self._onnx_node_name_to_module(node.name)
            module_name = self._onnx_node_name_to_module(node)
            layer_act = {}
            for index, input_name in enumerate(node.input):
                qparams = self._extract_qparams_from_dq_tensor(input_name)
                if qparams:
                    layer_act.setdefault("input", {})[str(index)] = self._qparams_to_aimet_encoding(
                        qparams["scale"], qparams["zeropoint"]
                    )
            if node.output:
                qparams = self._find_downstream_quant_params(
                    node.output[0],
                    {
                        "Relu", "LeakyRelu", "Clip", "Sigmoid", "Softmax",
                        "Add", "Mul", "Transpose", "Reshape", "Identity", "Cast",
                        "Flatten", "Squeeze", "Unsqueeze", "GlobalAveragePool",
                        "AveragePool", "MaxPool",
                    },
                )
                if qparams:
                    layer_act["output"] = {
                        "0": self._qparams_to_aimet_encoding(qparams["scale"], qparams["zeropoint"])
                    }
            if layer_act:
                encodings[module_name] = layer_act
        return encodings

    @staticmethod
    def _activation_input_suffix(node, index):
        if len(node.input) <= 1:
            return "input"
        return f"input_{index}"

    def _collect_activation_only_torch_state(self):
        state_dict = {}

        for node in self.onnx_model.graph.node:
            if node.op_type not in self.ACTIVATION_ONLY_OPS:
                continue

            # state_prefix = self._onnx_node_name_to_module(node.name)
            state_prefix = self._onnx_node_name_to_module(node)
            # print(
            #     "[activation-name]",
            #     f"op_type={node.op_type}",
            #     f"raw={node.name!r}",
            #     f"parsed={state_prefix!r}",
            # )
            input_found = False
            for index, input_name in enumerate(node.input):
                qparams = self._extract_qparams_from_dq_tensor(input_name)
                if qparams is None:
                    continue

                suffix = self._activation_input_suffix(node, index)
                state_dict[f"{state_prefix}.{suffix}_scale"] = qparams["scale"]
                state_dict[f"{state_prefix}.{suffix}_zero_point"] = qparams["zeropoint"]
                input_found = True

            output_qparams = None
            if node.output:
                output_qparams = self._find_downstream_quant_params(
                    node.output[0],
                    {
                        "Relu", "LeakyRelu", "Clip", "Sigmoid", "Softmax",
                        "Add", "Mul", "Transpose", "Reshape", "Identity", "Cast",
                        "Flatten", "Squeeze", "Unsqueeze", "GlobalAveragePool",
                        "AveragePool", "MaxPool",
                    },
                )
            if output_qparams is not None:
                state_dict[f"{state_prefix}.output_scale"] = output_qparams["scale"]
                state_dict[f"{state_prefix}.output_zero_point"] = output_qparams["zeropoint"]

        return state_dict, [], []

    def collect_encodings(self):
        activation_encodings = {}
        param_encodings = {}
        missing_input, missing_output = [], []

        for prefix in self._find_quantized_prefixes():
            export_prefix = self._normalize_prefix(prefix)

            weight_scale_name = f"{prefix}.weight_scale"
            weight_zero_point_name = f"{prefix}.weight_zero_point"
            if weight_scale_name in self.initializers and weight_zero_point_name in self.initializers:
                param_encodings[f"{export_prefix}.weight"] = self._qparams_to_aimet_encoding(
                    self._to_torch_tensor(self.initializers[weight_scale_name]),
                    self._to_torch_tensor(self.initializers[weight_zero_point_name]),
                )

            roles = self._get_activation_roles(export_prefix)
            if not roles:
                continue

            compute_node = self._find_compute_node_from_weight_qdq(f"{prefix}.weight_qdq")
            if compute_node is None and "input" in roles:
                missing_input.append(prefix)

            layer_act = {}
            if "input" in roles and compute_node is not None:
                activation_input_name = self._find_activation_input_name(compute_node, prefix)
                qparams = self._extract_qparams_from_dq_tensor(activation_input_name) if activation_input_name else None
                if qparams:
                    layer_act["input"] = {"0": self._qparams_to_aimet_encoding(qparams["scale"], qparams["zeropoint"])}
                else:
                    missing_input.append(prefix)

            if "output" in roles:
                qparams = self._find_output_quant_params(prefix, compute_node) if compute_node is not None else None
                if qparams is None:
                    qparams = self._find_output_quant_params_via_weight_path(prefix)
                if qparams:
                    layer_act["output"] = {"0": self._qparams_to_aimet_encoding(qparams["scale"], qparams["zeropoint"])}
                else:
                    missing_output.append(prefix)

            if layer_act:
                activation_encodings[export_prefix] = layer_act

        activation_encodings.update(self._collect_activation_only_encodings())

        return {
            "activation_encodings": activation_encodings,
            "excluded_layers": [],
            "param_encodings": param_encodings,
            "quantizer_args": {
                "activation_bitwidth": 8,
                "dtype": "int",
                "is_symmetric": True,
                "param_bitwidth": 8,
                "per_channel_quantization": True,
                "quant_scheme": "post_training_tf_enhanced",
            },
            "version": "1.0.0",
        }, missing_input, missing_output

    def collect_torch_state(self):
        state_dict = {}
        missing_input, missing_output = [], []
        handled_initializer_names = set()

        quantized_prefixes = sorted(
            {
                initializer.name[: -len(self.WEIGHT_Q_SUFFIX)]
                for initializer in self.onnx_model.graph.initializer
                if initializer.name.endswith(self.WEIGHT_Q_SUFFIX)
            }
            | {
                initializer.name[: -len(self.BIAS_Q_SUFFIX)]
                for initializer in self.onnx_model.graph.initializer
                if initializer.name.endswith(self.BIAS_Q_SUFFIX)
            }
        )

        for prefix in quantized_prefixes:
            state_prefix = self._onnx_prefix_to_state_prefix(prefix)
            roles = self._get_torch_activation_roles(state_prefix)

            if self._add_quantized_param_triplet(state_dict, state_prefix, "weight", prefix):
                handled_initializer_names.update(
                    {
                        f"{prefix}.weight_q",
                        f"{prefix}.weight_scale",
                        f"{prefix}.weight_zero_point",
                    }
                )

            if self._add_quantized_param_triplet(state_dict, state_prefix, "bias", prefix):
                handled_initializer_names.update(
                    {
                        f"{prefix}.bias_q",
                        f"{prefix}.bias_scale",
                        f"{prefix}.bias_zero_point",
                    }
                )

            if not roles:
                continue

            compute_node = self._find_compute_node_from_weight_qdq(f"{prefix}.weight_qdq")
            if compute_node is None and "input" in roles:
                missing_input.append(state_prefix)

            if "input" in roles and compute_node is not None:
                activation_input_name = self._find_activation_input_name(compute_node, prefix)
                input_qparams = self._extract_qparams_from_dq_tensor(activation_input_name) if activation_input_name else None
                if input_qparams is not None:
                    state_dict[f"{state_prefix}.input_scale"] = input_qparams["scale"]
                    state_dict[f"{state_prefix}.input_zero_point"] = input_qparams["zeropoint"]
                else:
                    missing_input.append(state_prefix)

            if "output" in roles:
                output_qparams = self._find_output_quant_params(prefix, compute_node) if compute_node is not None else None
                if output_qparams is None:
                    output_qparams = self._find_output_quant_params_via_weight_path(prefix)
                if output_qparams is not None:
                    state_dict[f"{state_prefix}.output_scale"] = output_qparams["scale"]
                    state_dict[f"{state_prefix}.output_zero_point"] = output_qparams["zeropoint"]
                else:
                    missing_output.append(state_prefix)

        for initializer in self.onnx_model.graph.initializer:
            if initializer.name in handled_initializer_names:
                continue
            if self._add_anonymous_matmul_weight(state_dict, initializer.name, initializer):
                handled_initializer_names.add(initializer.name)
                continue
            if not self._is_plain_model_initializer(initializer.name):
                continue
            state_dict[self._onnx_name_to_state_key(initializer.name)] = self._to_torch_tensor(initializer).to(torch.float32)

        activation_state, extra_missing_input, extra_missing_output = self._collect_activation_only_torch_state()
        state_dict.update(activation_state)
        missing_input.extend(extra_missing_input)
        missing_output.extend(extra_missing_output)

        state_dict = self._postprocess_torch_state(state_dict)
        return (
            state_dict,
            self._dedupe_preserve_order(missing_input),
            self._dedupe_preserve_order(missing_output),
        )
