import json
import os
import sys
from pathlib import Path

import torch

from .base import BaseExtractor


REPO_ROOT = Path(__file__).resolve().parents[2]
QUANTIZED_SSR_ROOT = REPO_ROOT / "QuantizedSSR"
QDEBUGGER_ROOT = REPO_ROOT / "QDebugger"


class EncodingsExtractor(BaseExtractor):
    def __init__(
        self,
        source_path,
        *,
        model_name="generic",
        weights_path=None,
        config=None,
        config_path=None,
        device="cpu",
        fuse_conv_bn=False,
        enable_bn_fold=False,
    ):
        super().__init__(source_path)
        with open(source_path, "r", encoding="utf-8") as handle:
            self.encodings = json.load(handle)
        self.model_name = model_name
        self.weights_path = weights_path
        self.config = config
        self.config_path = config_path
        self.device = device
        self.fuse_conv_bn = fuse_conv_bn
        self.enable_bn_fold = enable_bn_fold

    def collect_encodings(self):
        return self.encodings, [], []

    @staticmethod
    def _normalize_prefix(name):
        for wrapper in ("model.", "module.", "_orig_mod."):
            while name.startswith(wrapper):
                name = name[len(wrapper):]
        return name

    def _normalize_state_dict(self, state_dict):
        normalized = {}
        for key, value in state_dict.items():
            clean_key = self._normalize_prefix(key)
            if torch.is_tensor(value):
                normalized[clean_key] = value.detach().cpu()
            else:
                normalized[clean_key] = value
        return normalized

    def _extract_state_dict_from_checkpoint(self, checkpoint):
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                return checkpoint["state_dict"]
            if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                return checkpoint["model_state_dict"]
            if "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
                return checkpoint["model"].state_dict()
            if all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint
        elif hasattr(checkpoint, "state_dict"):
            return checkpoint.state_dict()
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    def _load_generic_weights_state(self):
        if self.weights_path is None:
            raise ValueError(
                "weights_path is required to export a .pt file from encodings."
            )
        checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        return self._normalize_state_dict(self._extract_state_dict_from_checkpoint(checkpoint))

    def _load_ssr_model_state(self):
        if self.weights_path is None:
            raise ValueError(
                "weights_path is required to rebuild an SSR torch model from encodings."
            )
        if self.config is None:
            raise ValueError(
                "config is required to rebuild an SSR torch model from encodings."
            )

        self._ensure_quantized_ssr_import_path()
        self._register_quantized_ssr_runtime()
        from quantization.quantize_function import load_quantized_model

        quant_obj = load_quantized_model(
            quant_weights=self.weights_path,
            device=torch.device(self.device),
            encoding_path=self.source_path,
            config=self.config,
            fuse_conv_bn=self.fuse_conv_bn,
            config_path=self.config_path,
            enable_bn_fold=self.enable_bn_fold,
        )
        model = quant_obj.get("model")
        if model is None:
            raise ValueError("QuantizedSSR loader did not return a torch model.")
        return self._normalize_state_dict(model.state_dict())

    @staticmethod
    def _ensure_quantized_ssr_import_path():
        ordered_paths = [
            str(QDEBUGGER_ROOT),
            str(QUANTIZED_SSR_ROOT),
            str(REPO_ROOT),
        ]

        py_deps = os.environ.get("PY_DEPS_DIR")
        if py_deps:
            if py_deps in sys.path:
                sys.path.remove(py_deps)
            sys.path.insert(0, py_deps)

        for path in reversed(ordered_paths):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)

        if "" not in sys.path:
            sys.path.append("")

    @staticmethod
    def _register_quantized_ssr_runtime():
        from aimet_torch.v2.nn import QuantizationMixin
        from mmcv.cnn.bricks.drop import Dropout
        from mmdet.models.losses.focal_loss import FocalLoss
        from mmdet.models.losses.iou_loss import GIoULoss
        from mmdet.models.losses.smooth_l1_loss import L1Loss
        from ssr.projects.mmdet3d_plugin.SSR.modules.temporal_self_attention import DummyQuant

        QuantizationMixin.ignore(FocalLoss)
        QuantizationMixin.ignore(L1Loss)
        QuantizationMixin.ignore(GIoULoss)
        QuantizationMixin.ignore(Dropout)

        @QuantizationMixin.implements(DummyQuant)
        class QuantizedDummyQuant(QuantizationMixin, DummyQuant):
            def __quant_init__(self):
                super().__quant_init__()
                self.input_quantizers = torch.nn.ModuleList([None])
                self.output_quantizers = torch.nn.ModuleList([None])

            def forward(self, x):
                if self.input_quantizers[0]:
                    x = self.input_quantizers[0](x)

                with self._patch_quantized_parameters():
                    ret = super().forward(x)

                if self.output_quantizers[0]:
                    ret = self.output_quantizers[0](ret)

                return ret

        # Import registers QuantizedLinear and related quantized op hooks.
        from quantization.registered_ops import QuantizedLinear  # noqa: F401

    def collect_torch_state(self):
        if self.model_name == "ssr":
            return self._load_ssr_model_state(), [], []
        return self._load_generic_weights_state(), [], []
