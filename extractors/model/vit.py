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

import torch

from ..qdq import QuantizedOnnxExtractor


class ViTExtractor(QuantizedOnnxExtractor):
    SPECIAL_FLOAT_INITIALIZERS = {"cls_token", "position_embeddings"}

    def __init__(self, ckpt_path, approximate_attention_qparams=True, attention_head_dim=64):
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
