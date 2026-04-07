---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
  - turboquant
  - tq3
  - compressed
  - quantized
  - moe
library_name: transformers
pipeline_tag: text-generation
---

# Gemma 4 26B-A4B-it -- TQ3 Compressed (FP16 weights with TQ3 noise)

TurboQuant TQ3-compressed version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

**Compression**: 3-bit TurboQuant with group size 128. Zero calibration data. Weights are decompressed back to FP16 for maximum compatibility -- any framework that loads HuggingFace models can use this checkpoint directly.

**For smaller GPUs**, use the [native TQ3 checkpoint](https://huggingface.co/varjosoft/gemma-4-26B-A4B-it-TQ3-native) instead (12 GB on disk, loads on L40S 48GB).

## Results

| Metric | Original (BF16) | This checkpoint (FP16 with TQ3 noise) |
|--------|----------|-------------------|
| Quality (20 scenarios) | baseline | **4.79/5** |
| Serving speed (A100) | 9-16 tok/s | **14-17 tok/s** (with runtime re-compression) |
| Checkpoint on disk | ~52 GB | **~52 GB** (FP16, same parameter count) |
| Runtime GPU memory | ~52 GB | **~12 GB** (with `enable_weight_quantization(bits=3)`) |

Quality validated on 20 multi-turn conversation scenarios scored by Llama-3.3-70B in an LLM-as-a-judge setup.

## Usage

This checkpoint stores standard FP16 weights. It loads like any HuggingFace model but requires `transformers >= 5.5.0` for Gemma 4 support.

### With runtime TQ3 re-compression (~12 GB GPU memory)

The weights already carry TQ3 quantization noise, so re-compressing them at runtime introduces near-zero additional error while reducing GPU memory from ~52 GB to ~12 GB.

```python
from turboquant_vllm import enable_weight_quantization

enable_weight_quantization(bits=3)
# Then: vllm serve varjosoft/gemma-4-26B-A4B-it-TQ3
```

Requires [turboquant-plus-vllm](https://github.com/varjoranta/turboquant-vllm) and an A100 80GB or larger GPU (the full checkpoint must fit in GPU memory during loading before compression).

### Standard loading (~52 GB GPU memory)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("varjosoft/gemma-4-26B-A4B-it-TQ3")
tokenizer = AutoTokenizer.from_pretrained("varjosoft/gemma-4-26B-A4B-it-TQ3")
```

### For smaller GPUs

Use the [native TQ3 checkpoint](https://huggingface.co/varjosoft/gemma-4-26B-A4B-it-TQ3-native) instead. It stores packed 3-bit indices directly (12 GB on disk). Tested on L40S 48GB and H100 80GB using a custom loader that keeps weights compressed in GPU memory.

## How It Was Made

Compressed using [turboquant-plus-vllm](https://github.com/varjoranta/turboquant-vllm):

1. Loaded original BF16 checkpoint to CPU
2. Applied TurboQuant TQ3 compression per weight group (Walsh-Hadamard rotation + Gaussian Lloyd-Max codebook + norm correction)
3. Decompressed back to FP16 and saved as standard safetensors

The weights carry TQ3 quantization noise but are stored as standard FP16. Any framework that loads HuggingFace models can use this checkpoint directly.

## Algorithm

Inspired by [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026). Our implementation uses a Gaussian Lloyd-Max codebook as an approximation of the paper's distortion-rate framework. Norm correction stores `original_norm / reconstruction_norm` per group to fix magnitude shrinkage at 3-bit.

## Citation

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

*Compressed by [Varjosoft Oy](https://varjosoft.com) using [turboquant-plus-vllm](https://github.com/varjoranta/turboquant-vllm).*
