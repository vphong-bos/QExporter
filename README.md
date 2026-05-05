# QExporter

QExporter extracts quantization parameters from QDQ-format ONNX checkpoints.
It can export either:

- a Torch `state_dict`-like `.pth` file containing quantized tensors and scale /
  zero-point tensors; or
- an AIMET-compatible `.encodings` JSON file containing activation and parameter
  encodings.

The extractor is intended for ONNX graphs where quantized layers are represented
with `QuantizeLinear` / `DequantizeLinear` nodes and initializer names such as
`<layer>.weight_q`, `<layer>.weight_scale`, and
`<layer>.weight_zero_point`.

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:

- `torch`
- `onnx`
- `onnxruntime`
- `aimet-torch`

## Extract a State Dict-Like Checkpoint

Use `extract_qparams.py` when you want a Torch checkpoint that can be loaded with
`torch.load`.

```bash
python extract_qparams.py path/to/model.onnx path/to/qparams.pth
```

The output is a Python dictionary saved by `torch.save`. For each quantized layer,
the script writes keys like:

```text
<layer>.weight
<layer>.weight_scale
<layer>.weight_zeropoint
<layer>.bias
<layer>.bias_scale
<layer>.bias_zeropoint
<layer>.input_scale
<layer>.input_zeropoint
<layer>.output_scale
<layer>.output_zeropoint
```

Bias and activation keys are emitted only when the corresponding tensors or graph
quantizers can be found.

Example:

```python
import torch

state = torch.load("path/to/qparams.pth", map_location="cpu")
print(state.keys())
print(state["layer1.0.conv1.weight_scale"])
```

During export, the script reports layers where explicit input or output
activation quantization parameters were not found.

## Extract AIMET Encodings

Use `run_extract.py` when you want an AIMET-style encodings JSON file.

```bash
python run_extract.py generic path/to/model.onnx path/to/output.encodings
```

Supported model names:

- `generic`
- `resnet50`
- `resnet101`
- `resnet152`

The generic extractor exports weight parameter encodings and input/output
activation encodings for every quantized `Conv`, `MatMul`, or `Gemm` layer it can
trace.

The ResNet extractor follows AIMET QuantSim naming conventions for ResNet
Bottleneck models:

- `conv1`: input activation encoding only
- `fc`: output activation encoding only
- block `conv3` and `downsample.*`: output activation encoding only
- post-add ReLU nodes: output activation encoding, named as
  `layerX.Y.relu`
- block `conv1` / `conv2` intermediate ReLU activations are skipped

Example:

```bash
python run_extract.py resnet50 resnet50_int8.onnx resnet50.encodings
```

The JSON has this top-level structure:

```json
{
  "activation_encodings": {},
  "excluded_layers": [],
  "param_encodings": {},
  "quantizer_args": {
    "activation_bitwidth": 8,
    "dtype": "int",
    "is_symmetric": true,
    "param_bitwidth": 8,
    "per_channel_quantization": true,
    "quant_scheme": "post_training_tf_enhanced"
  },
  "version": "1.0.0"
}
```

Each encoding stores AIMET fields such as `bitwidth`, `dtype`,
`is_symmetric`, `min`, `max`, `offset`, and `scale`.

## How Extraction Works

QExporter scans ONNX initializers for names ending in `.weight_q`. The part
before that suffix is treated as the layer prefix.

For each layer prefix, it:

1. reads quantized weights from `<prefix>.weight_q`;
2. reads weight qparams from `<prefix>.weight_scale` and
   `<prefix>.weight_zero_point`;
3. optionally reads bias tensors and bias qparams;
4. traces from `<prefix>.weight_qdq` through passthrough ops to find the compute
   node (`Conv`, `MatMul`, or `Gemm`);
5. traces graph inputs and outputs to find activation `scale` and `zero_point`
   initializers attached to QDQ nodes.

Common wrapper prefixes are stripped from exported names:

- `model.`
- `module.`
- `_orig_mod.`

Some scripts also normalize names such as `patch_embeddings.` to `patch_embed.`
and `*_zeropoint` to `*_zero_point`.

## Compare Against an FP32 Checkpoint

`get_mapping_insight.py` compares a floating-point checkpoint with an encodings
file and prints matched, missing, ambiguous, and shape-mismatched parameter
names.

```bash
python get_mapping_insight.py path/to/fp32_checkpoint.pth path/to/output.encodings --limit 100
```

The FP32 input may be a PyTorch checkpoint, pickle file, or ONNX file. The qparam
input is expected to be an AIMET-style encodings JSON file.

## Repository Layout

```text
run_extract.py                 # Main AIMET encodings CLI
extract_qparams.py             # Torch state_dict-like qparam checkpoint CLI
extractors/base.py             # Generic QDQ ONNX extractor
extractors/resnet.py           # ResNet-specific AIMET naming/filtering
extractors/__init__.py         # Model registry
get_mapping_insight.py         # FP32-to-qparam mapping/debug report
qparams/*.encodings            # Example encoding files
```

`extract_qparams_v1.py` and `extract_qparams_resnet50.py` are older standalone
variants kept for reference. Prefer `run_extract.py` for AIMET encodings and
`extract_qparams.py` for state_dict-like extraction.

## Limitations

- The ONNX model must be a quantized QDQ graph with quantization parameters
  stored as graph initializers.
- The default layer discovery depends on `.weight_q` initializer names.
- Activation qparams are discovered by graph traversal, so unusual graph rewrites
  or unsupported passthrough ops may require extending the extractor.
- Current model-specific AIMET behavior is implemented for ResNet Bottleneck
  models. Other architectures can use `generic` or add a subclass in
  `extractors/`.
