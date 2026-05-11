"""ResNet Bottleneck extractor (ResNet50 / 101 / 152).

AIMET convention for ResNet Bottleneck:
  - conv1: input encoding only
  - fc: output encoding only
  - conv3 / downsample.0: output encoding only
  - conv1, conv2 in blocks: no activation encoding (weight param only)
  - relu (post-Add): output encoding, named as layerX.Y.relu
  - relu after conv1/conv2 in blocks: skipped
"""

from .base import QuantizedOnnxExtractor


class ResNetExtractor(QuantizedOnnxExtractor):

    def _get_activation_roles(self, export_prefix):
        if export_prefix == "conv1":
            return {"input"}
        if export_prefix == "fc":
            return {"output"}
        if export_prefix.endswith(".conv3") or ".downsample." in export_prefix:
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
