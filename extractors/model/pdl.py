"""PDL extractor.

AIMET convention for PDL:
  - backbone.stem.conv1: output encoding only
  - relu (post-Add): output encoding, named as layerX.Y.relu
  - relu after conv1/conv2 in blocks: skipped
"""

from ..qdq import QuantizedOnnxExtractor


class PDLExtractor(QuantizedOnnxExtractor):

    def _get_activation_roles(self, export_prefix):
        if export_prefix == "backbone.stem.conv1":
            return {"output"}
        return set()

    def _collect_activation_only_encodings(self):
        encodings = {}
        for node in self.onnx_model.graph.node:
            if node.op_type != "Relu":
                continue
            module_name = self._onnx_node_name_to_module(node.name)

            # Keep only: initial relu and post-Add relu_2 per block
            if module_name == "relu":
                pass  # initial relu before maxpool
            elif module_name.endswith(".relu_2"):
                module_name = module_name[:-2]  # relu_2 -> relu
            else:
                continue

            if not node.output:
                continue
            for consumer in self.consumers.get(node.output[0], []):
                if consumer.op_type == "QuantizeLinear" and len(consumer.input) >= 3:
                    s, zp = consumer.input[1], consumer.input[2]
                    if s in self.initializers and zp in self.initializers:
                        encodings[module_name] = {
                            "output": {
                                "0": self._qparams_to_aimet_encoding(
                                    self._to_torch_tensor(self.initializers[s]),
                                    self._to_torch_tensor(self.initializers[zp]),
                                )
                            }
                        }
                        break
        return encodings
