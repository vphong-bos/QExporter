"""
    SSR model extractor (handles norm layers, attention components, embeddings).
"""

# Missing output qparams (1):
#   - model.pts_bbox_head.tokenlearner.layer_norm

from ..qdq import QuantizedOnnxExtractor


class SSRExtractor(QuantizedOnnxExtractor):
    def __init__(self, ckpt_path):
        super().__init__(ckpt_path)
        
        self.passthrough_ops = self.passthrough_ops | {
            "Mul", "Add", "Sub", "Div",
            "Tile", "Expand", "Concat",
            "Constant", "ConstantOfShape",
            "Where", "RoiAlign", "Pad", "Slice", "Resize",
        }

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
