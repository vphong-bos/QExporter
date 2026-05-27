# QExporter

Basic usage:

```bash
python run_extract.py yolov8s oft_int8.onnx
```

This now behaves like:
- `output_path` is optional.
- Default output is saved under `qparams/`.
- Relative custom output names are also saved under `qparams/`.
- `--source-format` is optional and auto-detected.
- `--format` is optional and defaults to `encodings` when output is omitted.

Examples:

```bash
# Auto detect source format (for example qdq) and save encodings to qparams/oft_int8.encodings
python run_extract.py yolov8s oft_int8.onnx

# Custom output name, still saved inside qparams/
python run_extract.py yolov8s oft_int8.onnx yolov8s_custom.encodings

# Export pt instead of encodings
python run_extract.py yolov8s oft_int8.onnx --format pt

# Force source format only when needed
python run_extract.py generic model_qop.onnx output.pt --source-format qop
```
