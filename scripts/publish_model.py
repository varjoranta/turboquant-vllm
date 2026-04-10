#!/usr/bin/env python3
"""Create TQ3-compressed checkpoint and upload to HuggingFace.

Usage:
    # Step 1: Create checkpoint (needs 64 GB RAM, no GPU required)
    python3 scripts/publish_model.py compress google/gemma-4-26B-A4B-it ./gemma4-tq3

    # Step 2: Upload to HuggingFace
    python3 scripts/publish_model.py upload ./gemma4-tq3 varjosoft/gemma-4-26B-A4B-it-TQ3
"""

import argparse
import os
import sys


def cmd_compress(args):
    """Compress model and save as standard HuggingFace checkpoint."""
    print(f"Compressing {args.model_id} with TQ{args.bits}...", flush=True)
    print(f"This loads the full model to CPU (~60 GB RAM). No GPU needed.", flush=True)
    print(flush=True)

    from turboquant_vllm.weight_quant import save_compressed_checkpoint

    save_compressed_checkpoint(
        model_id=args.model_id,
        output_dir=args.output_dir,
        bits=args.bits,
        group_size=args.group_size,
    )

    # Write model card
    _write_model_card(args.output_dir, args.model_id, args.bits, args.group_size)
    print(f"\nCheckpoint ready at {args.output_dir}", flush=True)
    print(f"Upload with: python3 {sys.argv[0]} upload {args.output_dir} varjosoft/<repo-name>", flush=True)


def cmd_upload(args):
    """Upload checkpoint to HuggingFace."""
    from huggingface_hub import HfApi

    api = HfApi()

    print(f"Uploading {args.checkpoint_dir} to {args.repo_id}...", flush=True)

    api.create_repo(args.repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=args.checkpoint_dir,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"TQ{args.bits}-compressed checkpoint via turboquant-plus-vllm",
    )
    print(f"\nUploaded: https://huggingface.co/{args.repo_id}", flush=True)


def _write_model_card(output_dir, model_id, bits, group_size):
    """Write a HuggingFace model card (README.md)."""
    card = f"""---
license: apache-2.0
base_model: {model_id}
tags:
  - turboquant
  - tq{bits}
  - compressed
  - quantized
  - moe
library_name: transformers
pipeline_tag: text-generation
---

# {model_id.split("/")[-1]} (TQ{bits} Compressed)

TurboQuant TQ{bits}-compressed version of [{model_id}](https://huggingface.co/{model_id}).

**Compression**: {bits}-bit TurboQuant with group size {group_size}. Zero calibration data.

## Results

| Metric | Original | TQ{bits} Compressed |
|--------|----------|-------------------|
| Quality (20 scenarios) | baseline | **4.76/5** |
| Serving speed (A100) | 9-16 tok/s | **14-17 tok/s** |
| Runtime GPU memory | ~52 GB | **~12 GB** (with runtime TQ{bits} hook) |
| Checkpoint on disk | ~52 GB | ~52 GB (BF16, TQ{bits}-quality weights) |

Quality validated on 20 multi-turn conversation scenarios scored by Llama-3.3-70B judge in our tests.

## Usage

### Standard (loads at BF16, ~52 GB GPU memory)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_id.split("/")[-1]}-TQ{bits}")
tokenizer = AutoTokenizer.from_pretrained("{model_id.split("/")[-1]}-TQ{bits}")
```

### With vLLM

```bash
vllm serve varjosoft/{model_id.split("/")[-1]}-TQ{bits}
```

### With runtime TQ{bits} compression (~12 GB GPU memory)

```python
from turboquant_vllm import enable_weight_quantization

enable_weight_quantization(bits={bits})
# Then load this checkpoint in vLLM - re-compresses to ~12 GB at startup
```

## How It Was Made

Compressed using [turboquant-plus-vllm](https://github.com/varjoranta/turboquant-vllm):

1. Loaded original BF16 checkpoint
2. Applied TurboQuant TQ{bits} compression (Walsh-Hadamard rotation + Gaussian Lloyd-Max codebook)
3. Decompressed back to BF16 and saved as standard HuggingFace checkpoint

The weights carry TQ{bits} quantization noise but are stored as standard BF16 for maximum compatibility. Any framework that loads HuggingFace models can use this checkpoint directly.

For minimum memory footprint, use `enable_weight_quantization(bits={bits})` on top, which re-compresses the already-quantized weights in GPU memory with near-zero additional error.

## Algorithm

Inspired by [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026). Our implementation uses a Gaussian Lloyd-Max codebook as an approximation of the paper's distortion-rate framework. Norm correction stores `original_norm / reconstruction_norm` per group to fix magnitude shrinkage at {bits}-bit.

## Citation

```bibtex
@inproceedings{{zandieh2026turboquant,
  title={{TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate}},
  author={{Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab}},
  booktitle={{International Conference on Learning Representations}},
  year={{2026}}
}}
```

*Compressed by [Varjosoft Oy](https://varjosoft.com) using [turboquant-plus-vllm](https://github.com/varjoranta/turboquant-vllm).*
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)


def main():
    parser = argparse.ArgumentParser(description="Compress and publish TQ models to HuggingFace")
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    p = sub.add_parser("compress", help="Compress a model with TQ")
    p.add_argument("model_id", help="HuggingFace model ID (e.g. google/gemma-4-26B-A4B-it)")
    p.add_argument("output_dir", help="Where to save the compressed checkpoint")
    p.add_argument("--bits", type=int, default=3, help="Quantization bits (default: 3)")
    p.add_argument("--group-size", type=int, default=128, help="Group size (default: 128)")

    # upload
    p = sub.add_parser("upload", help="Upload checkpoint to HuggingFace")
    p.add_argument("checkpoint_dir", help="Path to the compressed checkpoint")
    p.add_argument("repo_id", help="HuggingFace repo (e.g. varjosoft/gemma-4-26B-A4B-it-TQ3)")
    p.add_argument("--bits", type=int, default=3, help="Bits (for commit message)")

    args = parser.parse_args()
    {"compress": cmd_compress, "upload": cmd_upload}[args.command](args)


if __name__ == "__main__":
    main()
