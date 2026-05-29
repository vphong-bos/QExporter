"""
    SSR model extractor (handles norm layers, attention components, embeddings).
"""

# Missing output qparams (1):
#   - model.pts_bbox_head.tokenlearner.layer_norm

import copy

from ..qdq import QuantizedOnnxExtractor

import re

class SSRExtractor(QuantizedOnnxExtractor):
    _SSR_DUP_LAYER_RE = re.compile(r"^layer\d+$", re.IGNORECASE)

    @classmethod
    def _collapse_ssr_duplicate_layer_segments(cls, prefix):
        """Collapse SSR duplicate adjacent layer names.

        Example:
            model.backbone.layer1.layer1.conv1
            -> model.backbone.layer1.conv1

        Only collapses adjacent layerN/layerN-style duplicates, so names like
        layer1.block.layer1 are preserved.
        """
        normalized = str(prefix).replace("/", ".")
        parts = [part for part in normalized.split(".") if part]

        collapsed = []
        for part in parts:
            if (
                collapsed
                and collapsed[-1] == part
                and cls._SSR_DUP_LAYER_RE.match(part)
            ):
                continue
            collapsed.append(part)

        return ".".join(collapsed)

    def _ssr_prefix_aliases(self, prefix):
        aliases = []

        for candidate in (prefix, self._normalize_prefix(prefix)):
            candidate = str(candidate).replace("/", ".")
            collapsed = self._collapse_ssr_duplicate_layer_segments(candidate)

            for alias in (candidate, collapsed):
                if alias and alias not in aliases:
                    aliases.append(alias)

        return aliases

    def _has_torch_qparams(self, state_dict, prefix, role):
        return any(
            f"{alias}.{role}_scale" in state_dict
            and f"{alias}.{role}_zero_point" in state_dict
            for alias in self._ssr_prefix_aliases(prefix)
        )

    def _has_encoding_qparams(self, activation_encodings, prefix, role, index="0"):
        return any(
            index in activation_encodings.get(alias, {}).get(role, {})
            for alias in self._ssr_prefix_aliases(prefix)
        )
    
    def __init__(
        self,
        ckpt_path,
        approximate_attention_qparams=True,
        attention_head_dim=64,
    ):
        super().__init__(ckpt_path)
        self.approximate_attention_qparams = approximate_attention_qparams
        self.attention_head_dim = attention_head_dim
        
        self.passthrough_ops = self.passthrough_ops | {
            "Mul", "Add", "Sub", "Div",
            "Tile", "Expand", "Concat",
            "Constant", "ConstantOfShape",
            "Where", "RoiAlign", "Pad", "Slice", "Resize",
        }

    @classmethod
    def _collapse_ssr_duplicate_layer_segments(cls, prefix):
        """Reduce names like:
            model.backbone.layer1.layer1.conv1
            model/backbone/layer1/layer1/conv1

        into:
            model.backbone.layer1.conv1
        """
        prefix = str(prefix).replace("/", ".")
        parts = [p for p in prefix.split(".") if p]

        collapsed = []
        for part in parts:
            if (
                collapsed
                and collapsed[-1] == part
                and cls._SSR_DUP_LAYER_RE.match(part)
            ):
                continue
            collapsed.append(part)

        return ".".join(collapsed)

    def _normalize_prefix(self, prefix):
        prefix = super()._normalize_prefix(prefix)
        return self._collapse_ssr_duplicate_layer_segments(prefix)

    def _normalize_ssr_state_keys(self, state_dict):
        normalized = {}

        for key, value in state_dict.items():
            new_key = self._collapse_ssr_duplicate_layer_segments(key)

            # Prefer existing normalized key if both forms exist.
            if new_key not in normalized:
                normalized[new_key] = value

        state_dict.clear()
        state_dict.update(normalized)
        return state_dict

    def _normalize_ssr_activation_encodings(self, encodings):
        activation_encodings = encodings.get("activation_encodings", {})
        normalized = {}

        for key, value in activation_encodings.items():
            new_key = self._collapse_ssr_duplicate_layer_segments(key)

            # Prefer existing normalized key if both forms exist.
            if new_key not in normalized:
                normalized[new_key] = value

        encodings["activation_encodings"] = normalized
        return encodings

    def _find_compute_node_from_weight_qdq(self, weight_tensor_name):
        """Override to include LayerNormalization, InstanceNormalization, and Gather as compute ops for SSR."""
        queue = [weight_tensor_name]
        seen = set()
        ssr_compute_ops = self.COMPUTE_OPS | {"LayerNormalization", "InstanceNormalization", "Gather"}
        
        while queue:
            tensor = queue.pop(0)
            if tensor in seen:
                continue
            seen.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type in ssr_compute_ops:
                    return consumer
                if consumer.op_type in self.passthrough_ops:
                    queue.extend(consumer.output)
        return None

    def _find_output_quant_params(self, prefix, compute_node):
        """Override to handle SSR's extended passthrough ops (Tile, Expand, Shape, etc.)."""
        
        ssr_downstream_ops = {
            "BatchNormalization", "Relu", "Clip", "Add", "Transpose",
            "Reshape", "Identity", "Cast", "Flatten", "Squeeze",
            "Unsqueeze", "GlobalAveragePool", "AveragePool", "MaxPool",
            "Tile", "Expand", "Concat", "Mul", "Sub", "Div",
            "Shape", "Gather", "Constant", "ConstantOfShape",
            "Where", "RoiAlign", "Pad", "Slice", "Resize",
        }
        
        output_tensor = compute_node.output[0]

        if compute_node.op_type in {"MatMul", "Gemm"}:
            for consumer in self.consumers.get(output_tensor, []):
                if consumer.op_type == "Add" and any(
                    inp == f"{prefix}.bias_qdq" for inp in consumer.input
                ):
                    output_tensor = consumer.output[0]
                    break
        
        visited = set()
        queue = [output_tensor]
        while queue:
            tensor = queue.pop(0)
            if tensor in visited or len(visited) > 100:  # Increased limit
                continue
            visited.add(tensor)
            for consumer in self.consumers.get(tensor, []):
                if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                    s, zp = consumer.input[1], consumer.input[2]
                    if s in self.initializers and zp in self.initializers:
                        return {
                            "scale": self._to_torch_tensor(self.initializers[s]),
                            "zeropoint": self._to_torch_tensor(self.initializers[zp]),
                        }
                if consumer.op_type in ssr_downstream_ops:
                    queue.extend(consumer.output)
        return None

    def _get_activation_roles(self, export_prefix):
        """SSR-specific role assignment based on layer type.
        
        - Norm layers: output only 
        - Embeddings (col_embed, row_embed): output only (Gather has weight + constant index)
        - Layer norms in specific modules: output only
        - Attention projections: input + output
        - Everything else: input + output
        """
        prefix_lower = export_prefix.lower()
        
        if "navi_se.mlp_reduce" in prefix_lower:
            return {"output"}

        if "tokenlearner.layer_norm" in prefix_lower:
            return {"output"}

        # Output-only: norm layers, embeddings, and attention projections that are not quantized on input.
        if (
            "norm" in prefix_lower
            or "embed" in prefix_lower
            or prefix_lower.endswith(".output_proj")
            or prefix_lower.endswith(".value_proj")
            or prefix_lower.endswith(".attention_weights")
            or prefix_lower.endswith(".sampling_offsets")
        ):
            return {"output"}
        
        return {"input", "output"}

    def _get_torch_activation_roles(self, export_prefix):
        """For SSR .pt export, request both sides where SSR actually quantizes both.

        This keeps useful input/output qparams for debug exports while avoiding
        false missing-input reports on output-only structures like norms,
        embeddings, and certain attention helper projections.
        """
        return self._get_activation_roles(export_prefix)

    @staticmethod
    def _scalar_tensor(value, dtype=None):
        import torch

        tensor_dtype = torch.float32 if dtype is None else dtype
        return torch.tensor(float(value), dtype=tensor_dtype)

    def _attention_prefixes(self, state_dict):
        prefixes = set()
        projection_markers = (".q_proj.", ".k_proj.", ".v_proj.")
        for key in state_dict:
            for marker in projection_markers:
                if marker in key:
                    prefixes.add(key.split(marker, 1)[0])
                    break
        return sorted(prefixes)

    def _add_attention_pair(self, state_dict, prefix, suffix, scale, zero_point):
        scale_key = f"{prefix}.{suffix}_scale"
        zp_key = f"{prefix}.{suffix}_zero_point"
        if scale_key not in state_dict:
            state_dict[scale_key] = scale.clone() if hasattr(scale, "clone") else scale
        if zp_key not in state_dict:
            state_dict[zp_key] = zero_point.clone() if hasattr(zero_point, "clone") else zero_point

    def _add_attention_alias_pair(self, state_dict, prefixes, suffix, scale, zero_point):
        for prefix in prefixes:
            self._add_attention_pair(state_dict, prefix, suffix, scale, zero_point)

    def _fill_missing_attention_proj_inputs(self, state_dict):
        for attention_prefix in self._attention_prefixes(state_dict):
            source_pairs = [
                (
                    state_dict.get(f"{attention_prefix}.q_proj.input_scale"),
                    state_dict.get(f"{attention_prefix}.q_proj.input_zero_point"),
                ),
                (
                    state_dict.get(f"{attention_prefix}.k_proj.input_scale"),
                    state_dict.get(f"{attention_prefix}.k_proj.input_zero_point"),
                ),
                (
                    state_dict.get(f"{attention_prefix}.v_proj.input_scale"),
                    state_dict.get(f"{attention_prefix}.v_proj.input_zero_point"),
                ),
                (
                    state_dict.get(f"{attention_prefix.rsplit('.attn', 1)[0]}.input_1_scale"),
                    state_dict.get(f"{attention_prefix.rsplit('.attn', 1)[0]}.input_1_zero_point"),
                ),
                (
                    state_dict.get(f"{attention_prefix.rsplit('.attn', 1)[0]}.input_scale"),
                    state_dict.get(f"{attention_prefix.rsplit('.attn', 1)[0]}.input_zero_point"),
                ),
            ]

            fallback_scale = None
            fallback_zero_point = None
            for scale, zero_point in source_pairs:
                if scale is not None and zero_point is not None:
                    fallback_scale = scale
                    fallback_zero_point = zero_point
                    break

            if fallback_scale is None or fallback_zero_point is None:
                continue

            for proj_name in ("q_proj", "k_proj", "v_proj"):
                self._add_attention_pair(
                    state_dict,
                    f"{attention_prefix}.{proj_name}",
                    "input",
                    fallback_scale,
                    fallback_zero_point,
                )

    def _fill_missing_attention_proj_inputs_in_encodings(self, encodings):
        activation_encodings = encodings.setdefault("activation_encodings", {})
        attention_prefixes = set()
        for name in activation_encodings:
            for marker in (".q_proj", ".k_proj", ".v_proj"):
                if marker in name and ".attn." in name:
                    attention_prefixes.add(name.rsplit(marker, 1)[0])
                    break

        for attention_prefix in sorted(attention_prefixes):
            parent_prefix = attention_prefix.rsplit('.attn', 1)[0]
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
                parent_entry = activation_encodings.get(parent_prefix, {})
                parent_inputs = parent_entry.get("input", {}) if isinstance(parent_entry, dict) else {}
                for index in ("1", "0"):
                    if index in parent_inputs:
                        fallback_encoding = copy.deepcopy(parent_inputs[index])
                        break

            if fallback_encoding is None:
                continue

            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj_key = f"{attention_prefix}.{proj_name}"
                proj_entry = activation_encodings.setdefault(proj_key, {})
                proj_entry.setdefault("input", {})
                proj_entry["input"].setdefault("0", copy.deepcopy(fallback_encoding))

        return encodings

    def _add_approx_attention_qparams(self, state_dict):
        import math
        import torch

        for attention_prefix in self._attention_prefixes(state_dict):
            q_out_scale = state_dict.get(f"{attention_prefix}.q_proj.output_scale")
            k_out_scale = state_dict.get(f"{attention_prefix}.k_proj.output_scale")
            q_out_zp = state_dict.get(f"{attention_prefix}.q_proj.output_zero_point")
            v_out_scale = state_dict.get(f"{attention_prefix}.v_proj.output_scale")
            v_out_zp = state_dict.get(f"{attention_prefix}.v_proj.output_zero_point")
            softmax_out_scale = state_dict.get(f"{attention_prefix}.softmax.output_scale")
            softmax_out_zp = state_dict.get(f"{attention_prefix}.softmax.output_zero_point")

            if q_out_scale is not None and k_out_scale is not None:
                query_qk_prefix = f"{attention_prefix}.query_qk"
                matmul_qk_prefix = f"{attention_prefix}.matmul_qk"
                qk_scale_key = f"{query_qk_prefix}.output_scale"
                qk_zp_key = f"{query_qk_prefix}.output_zero_point"
                if qk_scale_key not in state_dict:
                    approx_scale = (
                        q_out_scale.to(torch.float32) * k_out_scale.to(torch.float32)
                        * (127.0 * math.sqrt(float(self.attention_head_dim)))
                    )
                    state_dict[qk_scale_key] = approx_scale
                    if qk_zp_key not in state_dict:
                        if q_out_zp is not None and q_out_zp.numel() == 1:
                            state_dict[qk_zp_key] = q_out_zp.clone()
                        else:
                            state_dict[qk_zp_key] = torch.tensor(128.0, dtype=torch.float32)
                self._add_attention_alias_pair(
                    state_dict,
                    [matmul_qk_prefix],
                    "output",
                    state_dict[qk_scale_key],
                    state_dict[qk_zp_key],
                )

                query_qk_scale = state_dict.get(qk_scale_key)
                query_qk_zp = state_dict.get(qk_zp_key)
                if query_qk_scale is not None and query_qk_zp is not None:
                    self._add_attention_alias_pair(
                        state_dict,
                        [f"{attention_prefix}.softmax"],
                        "input",
                        query_qk_scale,
                        query_qk_zp,
                    )

            if softmax_out_scale is not None and softmax_out_zp is not None:
                self._add_attention_alias_pair(
                    state_dict,
                    [
                        f"{attention_prefix}.query_pv",
                        f"{attention_prefix}.matmul_pv",
                    ],
                    "input",
                    softmax_out_scale,
                    softmax_out_zp,
                )

            if v_out_scale is not None and v_out_zp is not None:
                self._add_attention_alias_pair(
                    state_dict,
                    [
                        f"{attention_prefix}.query_pv",
                        f"{attention_prefix}.matmul_pv",
                    ],
                    "input_1",
                    v_out_scale,
                    v_out_zp,
                )

    def _postprocess_torch_state(self, state_dict):
        self._fill_missing_attention_proj_inputs(state_dict)
        if self.approximate_attention_qparams:
            self._add_approx_attention_qparams(state_dict)
        return state_dict

    def collect_torch_state(self):
        state_dict, missing_input, missing_output = super().collect_torch_state()

        missing_input = [
            prefix
            for prefix in missing_input
            if not self._has_torch_qparams(state_dict, prefix, "input")
        ]

        missing_output = [
            prefix
            for prefix in missing_output
            if not self._has_torch_qparams(state_dict, prefix, "output")
        ]

        return state_dict, missing_input, missing_output

        state_dict = self._normalize_ssr_state_keys(state_dict)

        missing_input = [
            self._collapse_ssr_duplicate_layer_segments(prefix)
            for prefix in missing_input
        ]
        missing_output = [
            self._collapse_ssr_duplicate_layer_segments(prefix)
            for prefix in missing_output
        ]

        return state_dict, missing_input, missing_output
    
    def collect_encodings(self):
        encodings, missing_input, missing_output = super().collect_encodings()
        encodings = self._fill_missing_attention_proj_inputs_in_encodings(encodings)

        activation_encodings = encodings.get("activation_encodings", {})

        missing_input = [
            self._collapse_ssr_duplicate_layer_segments(prefix)
            for prefix in missing_input
            if not self._has_encoding_qparams(
                activation_encodings,
                prefix,
                "input",
                "0",
            )
        ]

        missing_output = [
            prefix
            for prefix in missing_output
            if not self._has_encoding_qparams(
                activation_encodings,
                prefix,
                "output",
                "0",
            )
        ]

        return encodings, missing_input, missing_output
