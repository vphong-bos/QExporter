import copy
import onnx
import torch
from onnx import numpy_helper

from .base import BaseExtractor


class QOPExtractor(BaseExtractor):
    WEIGHTED_QOPS = {"QLinearConv", "QLinearMatMul"}
    ACTIVATION_QOPS = {
        "QLinearAdd",
        "QLinearMul",
        "QLinearAveragePool",
        "QLinearGlobalAveragePool",
        "QLinearLeakyRelu",
        "QLinearSigmoid",
        "QLinearSoftmax",
    }
    INTEGER_QOPS = {"ConvInteger", "MatMulInteger"}
    SUPPORTED_OPS = WEIGHTED_QOPS | ACTIVATION_QOPS | INTEGER_QOPS

    def __init__(self, source_path):
        super().__init__(source_path)
        self.model = onnx.load(source_path)
        self.init_map = self._build_init_map()

    @staticmethod
    def _to_torch(value):
        if value is None:
            return None
        return torch.from_numpy(value.copy())

    def _build_init_map(self):
        return {initializer.name: numpy_helper.to_array(initializer) for initializer in self.model.graph.initializer}

    @staticmethod
    def _normalize_prefix(name):
        name = name.lstrip("/")
        name = name.replace("/", ".")
        for wrapper in ("model.", "module.", "_orig_mod."):
            while name.startswith(wrapper):
                name = name[len(wrapper):]
        return name

    def _get_prefix(self, weight_name):
        prefix = weight_name.replace(".weight_quantized", "")
        return self._normalize_prefix(prefix)

    def _node_prefix(self, node):
        if node.name:
            return self._normalize_prefix(node.name)
        if node.output:
            return self._normalize_prefix(node.output[0])
        if node.input:
            return self._normalize_prefix(node.input[0])
        return self._normalize_prefix(node.op_type)

    def _get_initializer_tensor(self, name):
        return self._to_torch(self.init_map.get(name))

    @staticmethod
    def _to_python(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
        return value

    @classmethod
    def _qparams_to_aimet_encoding(cls, scale_tensor, zeropoint_tensor):
        scale = cls._to_python(scale_tensor)
        zeropoint = cls._to_python(zeropoint_tensor)

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

    def _set_activation_qparams(self, state_dict, prefix, role, scale_name, zero_point_name):
        scale_tensor = self._get_initializer_tensor(scale_name)
        zero_point_tensor = self._get_initializer_tensor(zero_point_name)
        if scale_tensor is None or zero_point_tensor is None:
            return False
        state_dict[f"{prefix}.{role}_scale"] = scale_tensor
        state_dict[f"{prefix}.{role}_zero_point"] = zero_point_tensor
        return True

    def _set_activation_encodings(self, layer_act, index, scale_name, zero_point_name):
        scale_tensor = self._get_initializer_tensor(scale_name)
        zero_point_tensor = self._get_initializer_tensor(zero_point_name)
        if scale_tensor is None or zero_point_tensor is None:
            return False
        layer_act.setdefault("input", {})[str(index)] = self._qparams_to_aimet_encoding(scale_tensor, zero_point_tensor)
        return True



    def _attention_projection_prefixes(self, values):
        prefixes = set()
        projection_markers = (".q_proj", ".k_proj", ".v_proj")
        for value in values:
            for marker in projection_markers:
                if marker in value and ".attn." in value:
                    prefixes.add(value.split(marker, 1)[0])
                    break
        return sorted(prefixes)

    def _fill_missing_attention_proj_inputs_state(self, state_dict):
        for attention_prefix in self._attention_projection_prefixes(state_dict.keys()):
            fallback_scale = None
            fallback_zero_point = None
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                scale = state_dict.get(f"{attention_prefix}.{proj_name}.input_scale")
                zero_point = state_dict.get(f"{attention_prefix}.{proj_name}.input_zero_point")
                if scale is not None and zero_point is not None:
                    fallback_scale = scale
                    fallback_zero_point = zero_point
                    break

            if fallback_scale is None or fallback_zero_point is None:
                continue

            for proj_name in ("q_proj", "k_proj", "v_proj"):
                state_dict.setdefault(
                    f"{attention_prefix}.{proj_name}.input_scale",
                    fallback_scale.clone() if hasattr(fallback_scale, "clone") else fallback_scale,
                )
                state_dict.setdefault(
                    f"{attention_prefix}.{proj_name}.input_zero_point",
                    fallback_zero_point.clone() if hasattr(fallback_zero_point, "clone") else fallback_zero_point,
                )

    def _fill_missing_attention_proj_inputs_encodings(self, activation_encodings):
        for attention_prefix in self._attention_projection_prefixes(activation_encodings.keys()):
            fallback_encoding = None
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj_entry = activation_encodings.get(f"{attention_prefix}.{proj_name}")
                if not proj_entry:
                    continue
                input_entry = proj_entry.get("input", {})
                if "0" in input_entry:
                    fallback_encoding = copy.deepcopy(input_entry["0"])
                    break

            if fallback_encoding is None:
                continue

            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj_entry = activation_encodings.setdefault(f"{attention_prefix}.{proj_name}", {})
                proj_entry.setdefault("input", {})
                proj_entry["input"].setdefault("0", copy.deepcopy(fallback_encoding))
    def collect_encodings(self):
        activation_encodings = {}
        param_encodings = {}
        missing_input = []
        missing_output = []

        for node in self.model.graph.node:
            if node.op_type not in self.SUPPORTED_OPS:
                continue

            prefix = self._get_prefix(node.input[3]) if node.op_type in self.WEIGHTED_QOPS else self._node_prefix(node)
            layer_act = {}

            if node.op_type in self.WEIGHTED_QOPS:
                if len(node.input) >= 6:
                    weight_scale = self._get_initializer_tensor(node.input[4])
                    weight_zero_point = self._get_initializer_tensor(node.input[5])
                    if weight_scale is not None and weight_zero_point is not None:
                        param_encodings[f"{prefix}.weight"] = self._qparams_to_aimet_encoding(weight_scale, weight_zero_point)

                if len(node.input) >= 3 and not self._set_activation_encodings(layer_act, 0, node.input[1], node.input[2]):
                    missing_input.append(prefix)
                if len(node.input) >= 8:
                    output_scale = self._get_initializer_tensor(node.input[6])
                    output_zero_point = self._get_initializer_tensor(node.input[7])
                    if output_scale is not None and output_zero_point is not None:
                        layer_act["output"] = {"0": self._qparams_to_aimet_encoding(output_scale, output_zero_point)}
                    else:
                        missing_output.append(prefix)
                else:
                    missing_output.append(prefix)
            elif node.op_type in self.ACTIVATION_QOPS:
                input_pairs = [
                    (0, 1, 2),
                    (1, 4, 5),
                ] if node.op_type in {"QLinearAdd", "QLinearMul"} else [(0, 1, 2)]

                input_found = False
                for input_index, scale_idx, zp_idx in input_pairs:
                    if len(node.input) > zp_idx and self._set_activation_encodings(layer_act, input_index, node.input[scale_idx], node.input[zp_idx]):
                        input_found = True
                if not input_found:
                    missing_input.append(prefix)

                if len(node.input) >= 8:
                    output_scale = self._get_initializer_tensor(node.input[6])
                    output_zero_point = self._get_initializer_tensor(node.input[7])
                    if output_scale is not None and output_zero_point is not None:
                        layer_act["output"] = {"0": self._qparams_to_aimet_encoding(output_scale, output_zero_point)}
                    else:
                        missing_output.append(prefix)
                else:
                    missing_output.append(prefix)
            else:
                missing_output.append(prefix)

            if layer_act:
                activation_encodings[prefix] = layer_act

        self._fill_missing_attention_proj_inputs_encodings(activation_encodings)
        missing_input = [
            prefix
            for prefix in missing_input
            if "0" not in activation_encodings.get(prefix, {}).get("input", {})
        ]

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
        missing_input = []
        missing_output = []

        for node in self.model.graph.node:
            if node.op_type not in self.SUPPORTED_OPS:
                continue

            inputs = node.input
            if node.op_type in self.WEIGHTED_QOPS:
                weight_name = inputs[3]
                if weight_name not in self.init_map:
                    missing_input.append(self._node_prefix(node))
                    continue

                prefix = self._get_prefix(weight_name)
                state_dict[f"{prefix}.weight"] = self._to_torch(self.init_map[weight_name])
                state_dict[f"{prefix}.weight_scale"] = self._get_initializer_tensor(inputs[4])
                state_dict[f"{prefix}.weight_zero_point"] = self._get_initializer_tensor(inputs[5])
                if not self._set_activation_qparams(state_dict, prefix, "input", inputs[1], inputs[2]):
                    missing_input.append(prefix)
                if len(inputs) >= 8:
                    if not self._set_activation_qparams(state_dict, prefix, "output", inputs[6], inputs[7]):
                        missing_output.append(prefix)
                else:
                    missing_output.append(prefix)

                if len(inputs) > 8:
                    bias_name = inputs[8]
                    if bias_name in self.init_map:
                        state_dict[f"{prefix}.bias"] = self._to_torch(self.init_map[bias_name])
                continue

            prefix = self._node_prefix(node)

            if node.op_type in self.ACTIVATION_QOPS:
                input_pairs = [
                    ("input", 1, 2),
                    ("input_1", 4, 5),
                ] if node.op_type in {"QLinearAdd", "QLinearMul"} else [("input", 1, 2)]

                input_found = False
                for role, scale_idx, zp_idx in input_pairs:
                    if len(inputs) > zp_idx and self._set_activation_qparams(state_dict, prefix, role, inputs[scale_idx], inputs[zp_idx]):
                        input_found = True
                if not input_found:
                    missing_input.append(prefix)

                if len(inputs) >= 8:
                    if not self._set_activation_qparams(state_dict, prefix, "output", inputs[6], inputs[7]):
                        missing_output.append(prefix)
                else:
                    missing_output.append(prefix)
                continue

            if len(inputs) > 2:
                input_zero_point = self._get_initializer_tensor(inputs[2])
                if input_zero_point is not None:
                    state_dict[f"{prefix}.input_zero_point"] = input_zero_point
            if len(inputs) > 3:
                weight_zero_point = self._get_initializer_tensor(inputs[3])
                if weight_zero_point is not None:
                    state_dict[f"{prefix}.weight_zero_point"] = weight_zero_point
            if len(inputs) > 1 and inputs[1] in self.init_map:
                state_dict[f"{prefix}.weight"] = self._to_torch(self.init_map[inputs[1]])
            missing_output.append(prefix)

        self._fill_missing_attention_proj_inputs_state(state_dict)
        missing_input = [
            prefix
            for prefix in missing_input
            if f"{prefix}.input_scale" not in state_dict
            or f"{prefix}.input_zero_point" not in state_dict
        ]
        return state_dict, missing_input, missing_output
