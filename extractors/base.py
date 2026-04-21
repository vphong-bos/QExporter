from pathlib import Path

import torch

class BaseExtractor:
    def __init__(self, source_path):
        self.source_path = str(source_path)

    def collect_encodings(self):
        raise ValueError(f"{self.__class__.__name__} does not support .encodings output")

    def collect_torch_state(self):
        raise ValueError(f"{self.__class__.__name__} does not support .pt output")

    def _infer_output_format(self, output_path, output_format=None):
        if output_format is not None:
            return output_format
        suffix = Path(output_path).suffix.lower()
        return "pt" if suffix in {".pt", ".pth"} else "encodings"

    @staticmethod
    def _report_missing(missing_in, missing_out):
        if missing_in:
            print(f"Missing input qparams ({len(missing_in)}):")
            for prefix in missing_in:
                print(f"  - {prefix}")
        if missing_out:
            print(f"Missing output qparams ({len(missing_out)}):")
            for prefix in missing_out:
                print(f"  - {prefix}")

    def save(self, output_path, output_format=None):
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_format = self._infer_output_format(output_path, output_format)

        if output_format == "encodings":
            result, missing_in, missing_out = self.collect_encodings()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            n_act = len(result.get("activation_encodings", {}))
            n_par = len(result.get("param_encodings", {}))
            print(f"Saved encodings to {output_path}  ({n_act} activation, {n_par} param)")
        elif output_format == "pt":
            result, missing_in, missing_out = self.collect_torch_state()
            torch.save(result, output_path)
            print(f"Saved {len(result)} tensors to {output_path}")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        self._report_missing(missing_in, missing_out)
        return result


from .qdq import QuantizedOnnxExtractor  # noqa: E402
