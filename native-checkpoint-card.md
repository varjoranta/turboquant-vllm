---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
model_type: gemma4
tags:
  - turboquant
  - tq3
  - compressed
  - quantized
  - moe
  - native-checkpoint
library_name: transformers
pipeline_tag: text-generation
model-index:
  - name: gemma-4-26B-A4B-it-TQ3-native
    results: []
---

# Gemma 4 26B-A4B-it -- Native TQ3 Checkpoint (12 GB)

Native 3-bit TurboQuant checkpoint of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). Weights are stored as packed 3-bit indices with per-group norms. **12 GB on disk** instead of 52 GB.

This checkpoint is designed for GPUs that cannot fit the original 52 GB BF16 weights. Tested on L40S 48GB ($0.91/hr) and H100 80GB.

## Usage

Requires [turboquant-plus-vllm](https://github.com/varjoranta/turboquant-vllm) and `transformers >= 5.5.0` (Gemma 4 support).

```bash
pip install turboquant-plus-vllm@git+https://github.com/varjoranta/turboquant-vllm.git
pip install 'transformers>=5.5'
```

```python
from turboquant_vllm import load_tq3_model

model, tokenizer = load_tq3_model("varjosoft/gemma-4-26B-A4B-it-TQ3-native")

chat = [{"role": "user", "content": "What is the capital of Finland?"}]
text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")

import torch
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## How It Works

The custom loader (`load_tq3_model`) creates the model skeleton on a meta device (zero memory), loads packed weights directly to GPU, and decompresses on-the-fly during each forward pass. Linear layers become compressed wrapper modules. MoE expert weights use chunked decompression (8 experts at a time) to limit GPU memory peak.

**This is NOT a standard HuggingFace checkpoint.** It cannot be loaded with `AutoModelForCausalLM.from_pretrained()`. You must use `load_tq3_model()` from the turboquant-plus-vllm library.

## Results

| Metric | Value |
|--------|-------|
| **Checkpoint size** | 12 GB (vs 52 GB BF16) |
| **GPU memory for weights** | 13.5 GB |
| **Compression ratio** | 4.3x |
| **Quality** | Same packed weights as runtime TQ3, which scores 4.79/5 on our 20-scenario benchmark |
| **Load time** | ~6 seconds (H100), ~12 seconds (L40S) |
| **Minimum GPU** | L40S 48GB tested. 24 GB-class feasible subject to KV cache and runtime overhead |

Quality validated by spot-check on H100 80GB: correct on factual recall, arithmetic, science, and Finnish language prompts.

## What This Checkpoint Contains

- `model-00001-of-00003.safetensors` through `model-00003-of-00003.safetensors`: packed 3-bit weight indices (`.tq_packed`) and per-group norms (`.tq_norms`) for compressed layers; FP16 tensors for embeddings, layer norms, and biases
- `tq_config.json`: compression parameters (bits=3, group_size=128, seed=42)
- `config.json`, `tokenizer.json`, `tokenizer_config.json`: standard HuggingFace model config and tokenizer

## How It Was Made

```python
from turboquant_vllm.checkpoint import save_tq3_checkpoint

save_tq3_checkpoint("google/gemma-4-26B-A4B-it", "./gemma4-tq3-native")
# CPU only, ~60 GB RAM, ~2 minutes. No GPU needed.
```

Each weight tensor is read from the original safetensors one at a time (lazy loading), compressed with TQ3 (Walsh-Hadamard rotation + Lloyd-Max codebook + norm correction), and saved as packed uint8 indices + float32 norms. Non-weight tensors (embeddings, norms, biases) are kept as FP16.

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
