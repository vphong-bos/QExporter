# Missing input qparams (25):
#   - encoder_layers.0.attention.output.dense
#   - encoder_layers.0.layernorm_after
#   - encoder_layers.0.layernorm_before
#   - encoder_layers.1.attention.output.dense
#   - encoder_layers.1.layernorm_after
#   - encoder_layers.1.layernorm_before
#   - encoder_layers.10.attention.output.dense
#   - encoder_layers.11.attention.output.dense
#   - encoder_layers.11.layernorm_after
#   - encoder_layers.2.attention.output.dense
#   - encoder_layers.2.layernorm_after
#   - encoder_layers.2.layernorm_before
#   - encoder_layers.3.attention.output.dense
#   - encoder_layers.3.layernorm_after
#   - encoder_layers.3.layernorm_before
#   - encoder_layers.4.attention.output.dense
#   - encoder_layers.4.layernorm_after
#   - encoder_layers.4.layernorm_before
#   - encoder_layers.5.attention.output.dense
#   - encoder_layers.5.layernorm_after
#   - encoder_layers.5.layernorm_before
#   - encoder_layers.6.attention.output.dense
#   - encoder_layers.7.attention.output.dense
#   - encoder_layers.8.attention.output.dense
#   - encoder_layers.9.attention.output.dense
# Missing output qparams (13):
#   - encoder_layers.0.layernorm_after
#   - encoder_layers.0.layernorm_before
#   - encoder_layers.1.layernorm_after
#   - encoder_layers.1.layernorm_before
#   - encoder_layers.11.layernorm_after
#   - encoder_layers.2.layernorm_after
#   - encoder_layers.2.layernorm_before
#   - encoder_layers.3.layernorm_after
#   - encoder_layers.3.layernorm_before
#   - encoder_layers.4.layernorm_after
#   - encoder_layers.4.layernorm_before
#   - encoder_layers.5.layernorm_after
#   - encoder_layers.5.layernorm_before

import copy
import math

import torch

from ..qdq import QuantizedOnnxExtractor
from ..encodings import EncodingsExtractor


class ViTExtractor(QuantizedOnnxExtractor):
    SPECIAL_FLOAT_INITIALIZERS = {"cls_token", "position_embeddings"}

    def __init__(
        self,
        ckpt_path,
        approximate_attention_qparams=True,
        attention_head_dim=64,
        weights_path=None,
    ):
        # Extend passthrough ops for ViT's transformer architecture
        super().__init__(ckpt_path)
        self.approximate_attention_qparams = approximate_attention_qparams
        self.attention_head_dim = attention_head_dim
        
        # Add more passthrough ops for ViT
        self.passthrough_ops = self.passthrough_ops | {
            "Mul", "Add", "Sub", "Div",  # Arithmetic
            "Tile", "Expand", "Concat",   # Reshaping
            "Constant", "ConstantOfShape",  # Constants
            "Where", "RoiAlign", "Pad", "Slice", "Resize",  # Other
            "ReduceMean", "ReduceSum", "Shape", "Gather",  # Reductions
        }

    def _onnx_prefix_to_state_prefix(self, prefix):
        if prefix == "patch_embed":
            return "vit.embeddings.patch_embeddings.projection"
        if prefix.startswith("encoder_layers."):
            return prefix.replace("encoder_layers.", "vit.encoder.layer.", 1)
        if prefix in {"cls_token", "position_embeddings"}:
            return f"vit.embeddings.{prefix}"
        if prefix.startswith("layernorm."):
            return f"vit.{prefix}"
        return prefix

    def _find_compute_node_from_weight_qdq(self, weight_tensor_name):
        """Override to include LayerNormalization as compute op for ViT."""
        queue = [weight_tensor_name]
        seen = set()
        vit_compute_ops = self.COMPUTE_OPS | {"LayerNormalization"}
        
        while queue:
            tensor = queue.pop(0)
            if tensor in seen:
                continue
            seen.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type in vit_compute_ops:
                    return consumer
                if consumer.op_type in self.passthrough_ops:
                    queue.extend(consumer.output)
        return None

    def _is_plain_model_initializer(self, name):
        if name in self.SPECIAL_FLOAT_INITIALIZERS:
            return True
        return super()._is_plain_model_initializer(name)

    def _get_activation_roles(self, export_prefix):
        """ViT-specific role assignment.
        
        - Layernorm layers: output only
        - Attention output dense: output only (similar to SSR)
        - Everything else: input + output
        """
        prefix_lower = export_prefix.lower()
        
        if "layernorm" in prefix_lower:
            return {"output"}
        
        if "attention.output.dense" in prefix_lower:
            return {"output"}
        
        return {"input", "output"}

    def _get_torch_activation_roles(self, export_prefix):
        """ViT .pt export should look like the target candidate checkpoint layout.

        - Layernorm params keep weight/bias qparams only; do not emit activation
          input/output qparams for them.
        - attention.output.dense keeps both input/output activation qparams in the
          candidate format.
        - Everything else keeps both input/output activation qparams.
        """
        prefix_lower = export_prefix.lower()

        if "layernorm" in prefix_lower:
            return set()

        if "attention.output.dense" in prefix_lower:
            return {"input", "output"}

        return {"input", "output"}

    def _postprocess_torch_state(self, state_dict):
        # Legacy candidate file did not carry this one output qparam pair.
        state_dict.pop("vit.encoder.layer.4.output.dense.output_scale", None)
        state_dict.pop("vit.encoder.layer.4.output.dense.output_zero_point", None)

        if self.approximate_attention_qparams:
            self._add_approx_attention_qparams(state_dict)

        return state_dict

    @staticmethod
    def _scalar_tensor(value, dtype=torch.float32):
        return torch.tensor(float(value), dtype=dtype)

    def _layer_from_state_key(self, key):
        parts = key.split(".")
        try:
            layer_idx = parts.index("layer") + 1
        except ValueError:
            return None
        if layer_idx >= len(parts):
            return None
        layer_token = parts[layer_idx]
        return int(layer_token) if layer_token.isdigit() else None

    def _matmul_qk_scale_for_layer(self, layer):
        mul_q_name = f"/encoder_layers.{layer}/attention/attention/Mul_output_0_scale"
        mul_k_name = f"/encoder_layers.{layer}/attention/attention/Mul_1_output_0_scale"
        if mul_q_name not in self.initializers or mul_k_name not in self.initializers:
            return None

        mul_q = float(self._to_torch_tensor(self.initializers[mul_q_name]).reshape(-1)[0])
        mul_k = float(self._to_torch_tensor(self.initializers[mul_k_name]).reshape(-1)[0])

        # Approximate the quantized score scale from the two attention branches.
        # The score tensor is a reduction over one head dimension, and these
        # Mul scales are already after the 1/sqrt(head_dim) attention scaling.
        return mul_q * mul_k * 127.0 * float(self.attention_head_dim)

    def _add_approx_attention_qparams(self, state_dict):
        attention_layers = sorted(
            {
                self._layer_from_state_key(key)
                for key in state_dict
                if ".attention.attention." in key
            }
            - {None}
        )

        if not attention_layers:
            return

        default_zero_point = self._scalar_tensor(128.0)

        for layer in attention_layers:
            attention_prefix = f"vit.encoder.layer.{layer}.attention.attention"

            qk_scale_key = f"{attention_prefix}.matmul_qk.output_scale"
            qk_zp_key = f"{attention_prefix}.matmul_qk.output_zero_point"
            if qk_scale_key not in state_dict:
                approx_scale = self._matmul_qk_scale_for_layer(layer)
                if approx_scale is not None:
                    state_dict[qk_scale_key] = self._scalar_tensor(approx_scale)
                    state_dict[qk_zp_key] = default_zero_point.clone()


class ViTAimetExtractor(EncodingsExtractor):
    def __init__(
        self,
        source_path,
        weights_path=None,
        attention_head_dim=64,
        approximate_attention_qparams=True,
    ):
        super().__init__(source_path)
        self.weights_path = weights_path
        self.attention_head_dim = attention_head_dim
        self._augmented_encodings = None
        self._state_dict_cache = None

    @staticmethod
    def _normalize_prefix(name):
        for wrapper in ("model.", "module.", "_orig_mod."):
            while name.startswith(wrapper):
                name = name[len(wrapper):]
        return name

    def _encoding_prefix_to_state_prefix(self, prefix):
        if prefix == "patch_embed":
            return "vit.embeddings.patch_embeddings.projection"
        if prefix.startswith("encoder_layers."):
            return prefix.replace("encoder_layers.", "vit.encoder.layer.", 1)
        if prefix in {"cls_token", "position_embeddings"}:
            return f"vit.embeddings.{prefix}"
        if prefix.startswith("layernorm."):
            return f"vit.{prefix}"
        return prefix

    @staticmethod
    def _encoding_to_zero_point(encoding, *, for_params):
        zero_point = int(-encoding["offset"])
        if for_params and str(encoding.get("is_symmetric", "False")).lower() == "true":
            zero_point = 0
        return zero_point

    @staticmethod
    def _encoding_to_tensor(encoding, *, for_params):
        scale = torch.tensor(float(encoding["scale"]), dtype=torch.float32)
        if for_params:
            zero_point = torch.tensor(
                ViTAimetExtractor._encoding_to_zero_point(encoding, for_params=True),
                dtype=torch.int8,
            )
        else:
            zero_point = torch.tensor(
                ViTAimetExtractor._encoding_to_zero_point(encoding, for_params=False),
                dtype=torch.uint8,
            )
        return scale, zero_point

    @staticmethod
    def _sequence_to_tensors(encodings, *, for_params):
        if isinstance(encodings, dict):
            return ViTAimetExtractor._encoding_to_tensor(
                encodings,
                for_params=for_params,
            )
        scales = []
        zero_points = []
        for encoding in encodings:
            scale, zero_point = ViTAimetExtractor._encoding_to_tensor(
                encoding,
                for_params=for_params,
            )
            scales.append(scale)
            zero_points.append(zero_point)
        return torch.stack(scales), torch.stack(zero_points)

    @staticmethod
    def _make_activation_encoding(scale, zero_point=128):
        scale = float(scale)
        zero_point = int(zero_point)
        return {
            "bitwidth": 8,
            "dtype": "int",
            "is_symmetric": "True",
            "max": float((255 - zero_point) * scale),
            "min": float((0 - zero_point) * scale),
            "offset": -zero_point,
            "scale": scale,
        }

    @staticmethod
    def _first_encoding(entry, role, index="0"):
        return entry.get(role, {}).get(index)

    def _iter_attention_layers(self, activation_encodings):
        layers = set()
        for name in activation_encodings:
            if not name.startswith("encoder_layers.") or ".attention.attention." not in name:
                continue
            parts = name.split(".")
            if len(parts) > 2 and parts[1].isdigit():
                layers.add(int(parts[1]))
        return sorted(layers)

    def _approx_qk_output_scale(self, query_encoding, key_encoding):
        query_scale = float(query_encoding["scale"])
        key_scale = float(key_encoding["scale"])
        return query_scale * key_scale * 127.0 * math.sqrt(float(self.attention_head_dim))

    def _augment_attention_encodings(self, encodings):
        activation_encodings = encodings.setdefault("activation_encodings", {})
        for layer in self._iter_attention_layers(activation_encodings):
            base_prefix = f"encoder_layers.{layer}.attention.attention"
            query = activation_encodings.get(f"{base_prefix}.query")
            key = activation_encodings.get(f"{base_prefix}.key")
            value = activation_encodings.get(f"{base_prefix}.value")
            if not query or not key:
                continue

            query_output = self._first_encoding(query, "output")
            key_output = self._first_encoding(key, "output")
            if query_output is None or key_output is None:
                continue

            matmul_qk_key = f"{base_prefix}.matmul_qk"
            matmul_qk_entry = activation_encodings.setdefault(matmul_qk_key, {})
            matmul_qk_entry.setdefault("input", {})
            matmul_qk_entry["input"].setdefault("0", copy.deepcopy(query_output))
            matmul_qk_entry["input"].setdefault("1", copy.deepcopy(key_output))
            matmul_qk_entry.setdefault("output", {})
            matmul_qk_entry["output"].setdefault(
                "0",
                self._make_activation_encoding(
                    self._approx_qk_output_scale(query_output, key_output)
                ),
            )

            softmax_key = f"{base_prefix}.softmax"
            softmax_entry = activation_encodings.setdefault(softmax_key, {})
            softmax_entry.setdefault("input", {})
            softmax_entry["input"].setdefault(
                "0",
                copy.deepcopy(matmul_qk_entry["output"]["0"]),
            )

            if value is not None:
                value_output = self._first_encoding(value, "output")
                if value_output is not None:
                    matmul_pv_key = f"{base_prefix}.matmul_pv"
                    matmul_pv_entry = activation_encodings.setdefault(matmul_pv_key, {})
                    matmul_pv_entry.setdefault("input", {})
                    matmul_pv_entry["input"].setdefault("1", copy.deepcopy(value_output))

        return encodings

    def _get_augmented_encodings(self):
        if self._augmented_encodings is None:
            self._augmented_encodings = self._augment_attention_encodings(
                copy.deepcopy(self.encodings)
            )
        return self._augmented_encodings

    def collect_encodings(self):
        return self._get_augmented_encodings(), [], []

    def _load_weights_state_dict(self):
        if self.weights_path is None:
            raise ValueError(
                "weights_path is required to export a .pt file from AIMET encodings."
            )
        if self._state_dict_cache is not None:
            return self._state_dict_cache

        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
                state_dict = checkpoint["model"].state_dict()
            elif all(isinstance(key, str) for key in checkpoint.keys()):
                state_dict = checkpoint
            else:
                raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")
        elif hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

        normalized = {}
        for key, value in state_dict.items():
            normalized[self._normalize_prefix(key)] = value.detach().cpu()
        self._state_dict_cache = normalized
        return self._state_dict_cache

    @staticmethod
    def _reshape_param_scales(param_tensor, scale_tensor, zero_point_tensor):
        if scale_tensor.ndim == 1 and param_tensor.ndim > 1 and scale_tensor.shape[0] == param_tensor.shape[0]:
            view_shape = [param_tensor.shape[0]] + [1] * (param_tensor.ndim - 1)
            scale_tensor = scale_tensor.reshape(view_shape)
            zero_point_tensor = zero_point_tensor.reshape(view_shape)
        return scale_tensor, zero_point_tensor

    def _quantize_weight(self, weight_tensor, encodings):
        scale_tensor, zero_point_tensor = self._sequence_to_tensors(encodings, for_params=True)
        scale_view, zero_point_view = self._reshape_param_scales(
            weight_tensor,
            scale_tensor,
            zero_point_tensor,
        )
        quantized = torch.round(weight_tensor.to(torch.float32) / scale_view) + zero_point_view.to(torch.float32)
        quantized = quantized.clamp(-128, 127).to(torch.int8)
        return quantized, scale_tensor, zero_point_tensor

    @staticmethod
    def _quantize_bias(bias_tensor, bias_scale):
        quantized = torch.round(bias_tensor.to(torch.float32) / bias_scale.to(torch.float32))
        return quantized.to(torch.int32)

    def _copy_plain_weights(self, state_dict, weights_state):
        for key, value in weights_state.items():
            if key in state_dict:
                continue
            state_dict[key] = value.to(torch.float32) if torch.is_floating_point(value) else value.clone()

    def _add_activation_state(self, state_dict, activation_encodings):
        for prefix, roles in activation_encodings.items():
            state_prefix = self._encoding_prefix_to_state_prefix(prefix)
            for role_name, role_values in roles.items():
                for index, encoding in role_values.items():
                    scale, zero_point = self._encoding_to_tensor(encoding, for_params=False)
                    suffix = role_name
                    if index != "0":
                        suffix = f"{role_name}_{index}"
                    state_dict[f"{state_prefix}.{suffix}_scale"] = scale
                    state_dict[f"{state_prefix}.{suffix}_zero_point"] = zero_point

    def collect_torch_state(self):
        encodings = self._get_augmented_encodings()
        weights_state = self._load_weights_state_dict()
        state_dict = {}

        for name, value in encodings.get("param_encodings", {}).items():
            if not name.endswith(".weight"):
                continue

            state_key = self._encoding_prefix_to_state_prefix(name)
            weight_tensor = weights_state.get(state_key)
            if weight_tensor is None:
                continue

            weight_tensor = weight_tensor.to(torch.float32)
            quantized_weight, weight_scale, weight_zero_point = self._quantize_weight(
                weight_tensor,
                value,
            )
            state_dict[state_key] = weight_tensor
            state_dict[f"{state_key}_q"] = quantized_weight
            state_dict[f"{state_key}_scale"] = weight_scale
            state_dict[f"{state_key}_zero_point"] = weight_zero_point

            bias_key = state_key[:-len(".weight")] + ".bias"
            bias_tensor = weights_state.get(bias_key)
            activation_key = name[:-len(".weight")]
            activation_entry = encodings.get("activation_encodings", {}).get(activation_key, {})
            input_encoding = self._first_encoding(activation_entry, "input")
            if bias_tensor is None or input_encoding is None:
                continue

            input_scale = torch.tensor(float(input_encoding["scale"]), dtype=torch.float32)
            bias_scale = weight_scale.to(torch.float32) * input_scale
            state_dict[bias_key] = bias_tensor.to(torch.float32)
            state_dict[f"{bias_key}_scale"] = bias_scale
            state_dict[f"{bias_key}_zero_point"] = torch.zeros_like(weight_scale, dtype=torch.int32)
            state_dict[f"{bias_key}_q"] = self._quantize_bias(bias_tensor, bias_scale)

        self._add_activation_state(state_dict, encodings.get("activation_encodings", {}))
        self._copy_plain_weights(state_dict, weights_state)
        return state_dict, [], []
