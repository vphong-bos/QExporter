import onnx
import torch
from onnx import numpy_helper

from .base import BaseExtractor


class QOPExtractor(BaseExtractor):
    SUPPORTED_OPS = {"QLinearConv", "QLinearMatMul"}

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

    def _get_prefix(self, weight_name):
        prefix = weight_name.replace(".weight_quantized", "")
        return f"model.{prefix}"

    def collect_torch_state(self):
        state_dict = {}

        for node in self.model.graph.node:
            if node.op_type not in self.SUPPORTED_OPS:
                continue

            inputs = node.input
            weight_name = inputs[3]
            if weight_name not in self.init_map:
                continue

            prefix = self._get_prefix(weight_name)
            state_dict[f"{prefix}.weight"] = self._to_torch(self.init_map[weight_name])
            state_dict[f"{prefix}.weight_scale"] = self._to_torch(self.init_map.get(inputs[4]))
            state_dict[f"{prefix}.weight_zeropoint"] = self._to_torch(self.init_map.get(inputs[5]))
            state_dict[f"{prefix}.input_scale"] = self._to_torch(self.init_map.get(inputs[1]))
            state_dict[f"{prefix}.input_zeropoint"] = self._to_torch(self.init_map.get(inputs[2]))
            state_dict[f"{prefix}.output_scale"] = self._to_torch(self.init_map.get(inputs[6]))
            state_dict[f"{prefix}.output_zeropoint"] = self._to_torch(self.init_map.get(inputs[7]))

            if len(inputs) > 8:
                bias_name = inputs[8]
                if bias_name in self.init_map:
                    state_dict[f"{prefix}.bias"] = self._to_torch(self.init_map[bias_name])

        return state_dict, [], []
