import torch
import torch.nn as nn
import json
from typing import Dict, Any
from collections import OrderedDict

class QuantModelWrapper(nn.Module):
    #TODO: Add support for loading and converting from various checkpoint formats (e.g., ONNX, etc.)
    def __init__(self, model, encodings_path):
        assert model, "Model cannot be empty."
        assert encodings_path, "Encodings path cannot be empty."

        super(QuantModelWrapper, self).__init__()
        self.model = model
        self.encodings_path = str(encodings_path)

        self.encodings = None
        self.param_encodings = None
        self.activation_encodings = None

        self._load_encodings(encodings_path)
        print(f"Loaded encodings from: {self.encodings_path}")

    # -------------------------
    # ENCODING ACCESS
    # -------------------------

    def get_param_key(self, limit: int = 25) -> str:
        param_keys = list(self.param_encodings.keys())
        return param_keys[:limit] if limit < len(param_keys) else param_keys

    def get_activation_keys(self, limit: int = 25) -> str:
        act_keys = list(self.activation_encodings.keys())
        return act_keys[:limit] if limit < len(act_keys) else act_keys

    def get_param_encoding(self, param_name: str) -> Dict[str, Any]:
        enc_list = self.param_encodings.get(param_name)
        if enc_list is None:
            raise KeyError(f"No encoding found for param: {param_name}")
        if not isinstance(enc_list, list) or len(enc_list) == 0:
            raise ValueError(f"Invalid encoding for param: {param_name}")
        return enc_list

    def get_activation_encoding(self, act_name: str) -> Dict[str, Any]:
        enc = self.activation_encodings.get(act_name)
        if enc is None:
            raise KeyError(f"No encoding found for activation: {act_name}")
        if not isinstance(enc, dict):
            raise ValueError(f"Invalid encoding for activation: {act_name}")
        return enc

    # -------------------------
    # ENCODING LOADING
    # -------------------------

    def _load_encodings(self):
        suffix = self.encodings_path.suffix.lower()
        if suffix in [".json", ".encodings", ".txt"]:
            with open(self.encodings_path, "r") as f:
                self.encodings = json.load(f)
                self.param_encodings = self.encodings.get("param_encodings", {})
                self.activation_encodings = self.encodings.get("activation_encodings", {})
        elif suffix in [".pt", ".pth"]:
            print("See you later, alligator! (Unsupported pytorch state dict quant format yet)")
            pass #TODO: Adding support for pt state dict
        else:
            raise ValueError(f"Unsupported encodings format: {suffix}")

    # -------------------------
    # QUANTIZATION LOGIC
    # -------------------------

    @staticmethod
    def _get_qrange(bitwidth: int = 8, signed: bool = True):
        if signed:
            return -(2 ** (bitwidth - 1)), (2 ** (bitwidth - 1)) - 1, torch.int8
        return 0, (2 ** bitwidth) - 1, torch.uint8

    @staticmethod
    def quantize_tensor(
        x: torch.Tensor,
        scale: float,
        zero_point: int,
        bitwidth: int = 8,
        signed: bool = True,
    ) -> torch.Tensor:
        qmin, qmax, dtype = QuantModelWrapper._get_qrange(bitwidth, signed)
        q = torch.round(x / scale) + zero_point
        q = torch.clamp(q, qmin, qmax)
        return q.to(dtype)

    # -------------------------
    # PARAMETER QUANTIZATION
    # -------------------------

    def _map_per_tensor_qparams(self, x: torch.Tensor, enc: Dict[str, Any]):
        return self.quantize_tensor(
            x=x,
            scale=float(enc["scale"]),
            zero_point=int(enc["offset"]),
            bitwidth=int(enc.get("bitwidth", 8)),
            signed=True,
        )
    def _map_per_channel_qparams(self, x: torch.Tensor, enc_list: list):
        q_slices = []
        for ch, enc in enumerate(enc_list):
            q_slices.append(
                self.quantize_tensor(
                    x=x[ch],
                    scale=float(enc["scale"]),
                    zero_point=int(enc["offset"]),
                    bitwidth=int(enc.get("bitwidth", 8)),
                    signed=True,
                )
            )
        return torch.stack(q_slices, dim=0)

    def quantize_params(self) -> OrderedDict:
        q_state = OrderedDict()

        state_dict = self.model.state_dict() if hasattr(self.model, "state_dict") else self.model

        for name, tensor in state_dict.items():
            enc_list = self.param_encodings.get(name)

            if enc_list is None:
                q_state[name] = tensor
                continue

            if not isinstance(enc_list, list) or len(enc_list) == 0:
                raise ValueError(f"Invalid encoding for param: {name}")

            if len(enc_list) == 1:
                q_state[name] = self._quantize_per_tensor_param(tensor, enc_list[0])
            else:
                q_state[name] = self._quantize_per_channel_param(tensor, enc_list)

        return q_state

    # -------------------------
    # ACTIVATION QUANTIZATION
    # -------------------------

    def quantize_activation(self, x, name):
        enc_list = self.get_activation_encoding(name)
        if enc_list is None:
            raise KeyError(f"Missing activation encoding: {name}")

        enc = enc_list[0]

        return self.quantize_tensor(
            x,
            enc["scale"],
            enc["offset"],
            enc.get("bitwidth", 8),
            signed=True,
        )

        
    # -------------------------
    # FORWARD METHOD (TO BE IMPLEMENTED BY SUBCLASS)
    # -------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Subclass must implement forward() using quantized ops"
        )