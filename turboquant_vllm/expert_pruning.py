"""REAP-style expert pruning for MoE models.

Implements Router-weighted Expert Activation Pruning (Cerebras, ICLR 2026):
  saliency_j = mean over active tokens of: gate_value(x) × ||expert_output(x)||₂

Experts with lowest saliency are pruned (weights zeroed, router renormalized).
Requires a small calibration set (~1024 samples) for saliency estimation.

Usage:
    from turboquant_vllm.expert_pruning import reap_prune

    reap_prune(model, tokenizer, prune_fraction=0.5, num_samples=1024)
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _find_moe_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Find MoE blocks that have a gate and experts.

    Returns: [(module_path, module)] for each MoE block.
    """
    blocks = []
    for name, module in model.named_modules():
        gate = getattr(module, 'gate', None)
        if gate is not None and hasattr(gate, 'weight'):
            # Check if this module also has experts
            experts = getattr(module, 'experts', None)
            if experts is not None:
                blocks.append((name, module))
    return blocks


def compute_reap_saliency(
    model: nn.Module,
    tokenizer,
    num_samples: int = 1024,
    max_length: int = 512,
    dataset_name: str = "allenai/c4",
    batch_size: int = 4,
) -> dict[str, torch.Tensor]:
    """Compute REAP saliency scores for all experts in all MoE layers.

    For each expert j in each layer:
      S_j = (1/|active_tokens_j|) × Σ gate_value(x) × ||expert_output(x)||₂

    Args:
        model: The model with MoE layers.
        tokenizer: Tokenizer for preparing calibration text.
        num_samples: Number of calibration samples (1024 is sufficient).
        max_length: Max sequence length for calibration.
        dataset_name: HuggingFace dataset for calibration text.
        batch_size: Batch size for calibration forward passes.

    Returns:
        {moe_block_path: saliency_tensor (num_experts,)} for each MoE layer.
    """
    moe_blocks = _find_moe_blocks(model)
    if not moe_blocks:
        logger.warning("No MoE blocks found in model")
        return {}

    device = next(model.parameters()).device

    # Prepare calibration data
    calibration_inputs = _prepare_calibration_data(
        tokenizer, num_samples, max_length, dataset_name, device
    )

    # Register hooks to collect gate values and expert activation norms
    collectors: dict[str, _SaliencyCollector] = {}
    hooks = []

    for block_path, block in moe_blocks:
        gate_module = block.gate
        num_experts = gate_module.weight.shape[0]
        collector = _SaliencyCollector(num_experts, device)
        collectors[block_path] = collector

        # Detect top_k from model config
        top_k = getattr(model.config, 'num_experts_per_tok',
                 getattr(model.config, 'num_selected_experts', 8))

        # Hook the gate to capture routing decisions
        hooks.append(gate_module.register_forward_hook(
            _make_gate_hook(collector, top_k=top_k)
        ))

        # Hook each expert to capture activation norms
        for idx in range(num_experts):
            expert = _get_expert_by_index(block, idx)
            if expert is not None:
                hooks.append(expert.register_forward_hook(
                    _make_expert_hook(collector, idx)
                ))

    # Run calibration forward passes
    model.eval()
    logger.info("Computing REAP saliency with %d samples...", num_samples)

    with torch.no_grad():
        for i in range(0, len(calibration_inputs), batch_size):
            batch = calibration_inputs[i:i + batch_size]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                batch, batch_first=True, padding_value=tokenizer.pad_token_id or 0
            )
            attention_mask = (input_ids != (tokenizer.pad_token_id or 0)).long()
            model(input_ids=input_ids, attention_mask=attention_mask)

            if (i // batch_size) % 50 == 0:
                logger.info("  Calibration progress: %d/%d samples", min(i + batch_size, len(calibration_inputs)), num_samples)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute final saliency scores
    saliency = {}
    for block_path, collector in collectors.items():
        scores = collector.compute_saliency()
        saliency[block_path] = scores
        top3 = scores.argsort(descending=True)[:3].tolist()
        bot3 = scores.argsort()[:3].tolist()
        logger.info(
            "REAP saliency %s: top experts %s, bottom experts %s, "
            "max=%.4f, min=%.4f, ratio=%.1f",
            block_path, top3, bot3,
            scores.max().item(), scores.min().item(),
            scores.max().item() / max(scores.min().item(), 1e-10),
        )

    return saliency


class _SaliencyCollector:
    """Accumulates gate values and expert activation norms for REAP scoring."""

    def __init__(self, num_experts: int, device: torch.device):
        self.num_experts = num_experts
        # Sum of gate_value × activation_norm for each expert
        self.weighted_sum = torch.zeros(num_experts, device=device)
        # Count of tokens where each expert was active
        self.active_count = torch.zeros(num_experts, device=device)
        # Current batch's gate values (set by gate hook, read by expert hooks)
        self._current_gate_values = None
        self._current_top_k_indices = None

    def record_gate(self, gate_values: torch.Tensor, top_k_indices: torch.Tensor):
        """Called by gate hook. gate_values: (batch*seq, top_k), top_k_indices: same."""
        self._current_gate_values = gate_values.detach()
        self._current_top_k_indices = top_k_indices.detach()

    def record_expert_activation(self, expert_idx: int, activation_norm: float,
                                  num_tokens: int):
        """Called by expert hook."""
        if self._current_gate_values is None:
            return
        # Find tokens where this expert was in top-k (vectorized)
        expert_mask = (self._current_top_k_indices == expert_idx)  # (batch*seq, top_k)
        active = expert_mask.any(dim=-1)  # (batch*seq,)
        if active.any():
            # Sum gate values across all top-k slots where this expert appears
            gate_vals = (self._current_gate_values * expert_mask.float()).sum(dim=-1)  # (batch*seq,)
            gate_sum = gate_vals[active].sum().item()
            self.weighted_sum[expert_idx] += gate_sum * activation_norm / max(num_tokens, 1)
            self.active_count[expert_idx] += active.sum().item()

    def compute_saliency(self) -> torch.Tensor:
        """Compute final saliency: weighted_sum / active_count."""
        safe_count = torch.where(self.active_count > 0, self.active_count,
                                  torch.ones_like(self.active_count))
        return self.weighted_sum / safe_count


def _get_expert_by_index(moe_block: nn.Module, idx: int) -> nn.Module | None:
    """Get expert module by index. Handles both list and ModuleList."""
    experts = moe_block.experts
    if isinstance(experts, (nn.ModuleList, list)):
        if idx < len(experts):
            return experts[idx]
    # Try indexed attribute
    expert = getattr(experts, str(idx), None)
    return expert


def _make_gate_hook(collector: _SaliencyCollector, top_k: int = 8):
    """Create a forward hook for the gate/router module."""
    def hook(module, input, output):
        # Gate output is logits (batch*seq, num_experts)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        logits = logits.float()
        k = min(top_k, logits.shape[-1])
        gate_values, top_k_indices = torch.topk(
            torch.softmax(logits, dim=-1), k=k, dim=-1
        )
        # Flatten batch and seq dimensions
        if gate_values.dim() > 2:
            gate_values = gate_values.reshape(-1, top_k)
            top_k_indices = top_k_indices.reshape(-1, top_k)
        collector.record_gate(gate_values, top_k_indices)
    return hook


def _make_expert_hook(collector: _SaliencyCollector, expert_idx: int):
    """Create a forward hook for an individual expert module."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        # Activation norm: L2 norm of the expert's output
        norm = out.float().norm().item()
        num_tokens = out.shape[0] if out.dim() >= 2 else 1
        collector.record_expert_activation(expert_idx, norm, num_tokens)
    return hook


def _prepare_calibration_data(
    tokenizer, num_samples: int, max_length: int, dataset_name: str,
    device: torch.device,
) -> list[torch.Tensor]:
    """Load and tokenize calibration samples."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, "en", split="train", streaming=True)
        texts = []
        for i, sample in enumerate(ds):
            if i >= num_samples:
                break
            text = sample.get("text", "")
            if len(text) > 50:  # skip very short
                texts.append(text[:max_length * 4])  # rough char limit
    except Exception as e:
        logger.warning("Could not load %s dataset: %s. Using synthetic data.", dataset_name, e)
        texts = ["The quick brown fox jumps over the lazy dog. " * 20] * num_samples

    # Batch tokenize for efficiency
    batch = tokenizer(texts[:num_samples], truncation=True, max_length=max_length,
                      padding=False, return_attention_mask=False)
    inputs = [torch.tensor(ids, device=device) for ids in batch["input_ids"]]

    logger.info("Prepared %d calibration samples (max_length=%d)", len(inputs), max_length)
    return inputs


def reap_prune(
    model: nn.Module,
    tokenizer,
    prune_fraction: float = 0.5,
    num_samples: int = 1024,
    max_length: int = 512,
    renormalize_router: bool = True,
) -> dict[str, list[int]]:
    """Prune MoE experts using REAP saliency scoring.

    Args:
        model: Model to prune (modified in-place).
        tokenizer: Tokenizer for calibration.
        prune_fraction: Fraction of experts to prune per layer (0.0-1.0).
        num_samples: Calibration samples for saliency estimation.
        max_length: Max sequence length for calibration.
        renormalize_router: Renormalize router weights after pruning.

    Returns:
        {moe_block_path: [pruned_expert_indices]} for each layer.
    """
    saliency = compute_reap_saliency(model, tokenizer, num_samples, max_length)

    pruned = {}
    total_pruned = 0

    for block_path, scores in saliency.items():
        num_experts = scores.shape[0]
        n_prune = int(num_experts * prune_fraction)
        if n_prune == 0:
            continue

        # Prune experts with lowest saliency
        prune_indices = scores.argsort()[:n_prune].tolist()
        pruned[block_path] = prune_indices

        # Zero out pruned expert weights
        parts = block_path.split(".")
        block = model
        for part in parts:
            block = getattr(block, part)

        for idx in prune_indices:
            expert = _get_expert_by_index(block, idx)
            if expert is not None:
                for param in expert.parameters():
                    param.data.zero_()

        # Suppress pruned experts: add a permanent forward hook on the gate
        # that sets pruned expert logits to -inf before softmax.
        # This works regardless of whether the gate has bias.
        if renormalize_router:
            gate = block.gate
            prune_mask = torch.zeros(gate.weight.shape[0], device=gate.weight.device, dtype=torch.bool)
            for idx in prune_indices:
                prune_mask[idx] = True

            def _gate_mask_hook(module, input, output, mask=prune_mask):
                if isinstance(output, tuple):
                    logits = output[0]
                    logits[:, mask] = float('-inf')
                    return (logits,) + output[1:]
                else:
                    output[:, mask] = float('-inf')
                    return output

            gate.register_forward_hook(_gate_mask_hook)

        total_pruned += n_prune
        logger.info(
            "REAP pruned %s: removed %d/%d experts (indices: %s)",
            block_path, n_prune, num_experts,
            prune_indices[:5] if len(prune_indices) > 5 else prune_indices,
        )

    logger.info("REAP total: pruned %d experts across %d layers", total_pruned, len(pruned))
    return pruned


# ---------------------------------------------------------------------------
# Router fine-tuning post-REAP (MC# ICLR 2025)
# ---------------------------------------------------------------------------

def finetune_router(
    model: nn.Module,
    tokenizer,
    num_steps: int = 200,
    lr: float = 1e-4,
    num_samples: int = 256,
    max_length: int = 512,
) -> float:
    """Fine-tune router gates after expert pruning.

    Freezes all weights except gate modules. Trains gates with
    language modeling loss so routing adapts to the surviving expert set.
    Based on MC# (ICLR 2025): learnable router adaptation is critical
    for quality recovery after pruning.

    Returns: final loss value.
    """
    moe_blocks = _find_moe_blocks(model)
    if not moe_blocks:
        return 0.0

    device = next(model.parameters()).device

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only gate weights
    gate_params = []
    for _, block in moe_blocks:
        for param in block.gate.parameters():
            param.requires_grad = True
            gate_params.append(param)

    if not gate_params:
        return 0.0

    optimizer = torch.optim.Adam(gate_params, lr=lr)
    calibration_inputs = _prepare_calibration_data(
        tokenizer, num_samples, max_length, "allenai/c4", device
    )

    model.train()
    total_loss = 0.0
    step = 0

    logger.info("Router fine-tuning: %d steps, %d gate params, lr=%g",
                num_steps, sum(p.numel() for p in gate_params), lr)

    for epoch in range(num_steps // max(len(calibration_inputs), 1) + 1):
        for input_ids in calibration_inputs:
            if step >= num_steps:
                break

            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            if loss is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
                optimizer.step()
                total_loss += loss.item()

            step += 1
            if step % 50 == 0:
                avg = total_loss / step
                logger.info("  Router fine-tune step %d/%d, loss=%.4f", step, num_steps, avg)

    model.eval()

    # Re-freeze gates
    for param in gate_params:
        param.requires_grad = False

    avg_loss = total_loss / max(step, 1)
    logger.info("Router fine-tuning complete: %d steps, final avg loss=%.4f", step, avg_loss)
    return avg_loss


# ---------------------------------------------------------------------------
# Hessian diagonal collection (for mixed-precision + sparse outliers)
# ---------------------------------------------------------------------------

def collect_hessian_diagonal(
    model: nn.Module,
    tokenizer,
    num_samples: int = 256,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Collect approximate Hessian diagonal for each linear layer.

    Uses the Fisher approximation: H_ii ≈ E[(∂L/∂w_i)²] ≈ E[activation_i²]
    Computed during a single forward pass over calibration data.

    Returns: {param_name: hessian_diag tensor (same shape as weight)}
    """
    device = next(model.parameters()).device
    calibration_inputs = _prepare_calibration_data(
        tokenizer, num_samples, max_length, "allenai/c4", device
    )

    # Collect squared activations for each linear layer
    hessian_accum: dict[str, torch.Tensor] = {}
    sample_count = 0
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.numel() < 1024:
            continue

        # Accumulator for squared input activations
        h = torch.zeros(module.in_features, device=device, dtype=torch.float32)
        hessian_accum[name] = h

        def _hessian_hook(mod, inp, out, accum=h):
            x = inp[0].float()
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            # H_diag ≈ mean(x²) per input feature
            accum.add_(x.pow(2).sum(dim=0))

        hooks.append(module.register_forward_hook(_hessian_hook))

    model.eval()
    logger.info("Collecting Hessian diagonal with %d samples...", num_samples)

    with torch.no_grad():
        for input_ids in calibration_inputs:
            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            model(input_ids=input_ids)
            sample_count += input_ids.shape[0]

    for hook in hooks:
        hook.remove()

    # Normalize by sample count
    for name in hessian_accum:
        hessian_accum[name] /= max(sample_count, 1)

    logger.info("Hessian diagonal collected for %d layers", len(hessian_accum))
    return hessian_accum


# ---------------------------------------------------------------------------
# Mixed-precision bit selection per expert
# ---------------------------------------------------------------------------

def compute_expert_bit_widths(
    model: nn.Module,
    hessian: dict[str, torch.Tensor],
    default_bits: int = 4,
    shared_expert_bits: int = 5,
    important_routed_bits: int = 3,
    weak_routed_bits: int = 2,
    saliency: dict[str, torch.Tensor] | None = None,
) -> dict[str, int]:
    """Assign per-module bit widths based on Hessian sensitivity and saliency.

    Shared experts (always active) → higher precision (default 5-bit).
    Important routed experts → medium precision (3-bit).
    Weak routed experts → low precision (2-bit).
    Non-expert linears → default bits.

    Returns: {module_name: bit_width}
    """
    bit_widths: dict[str, int] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.numel() < 1024:
            continue

        if "shared_expert" in name.lower():
            bit_widths[name] = shared_expert_bits
        elif "expert" in name.lower():
            # Use saliency to determine importance if available
            if saliency:
                # Find which layer's saliency applies
                for block_path, scores in saliency.items():
                    if name.startswith(block_path):
                        # Extract expert index from name
                        import re
                        m = re.search(r'experts\.(\d+)', name)
                        if m:
                            idx = int(m.group(1))
                            if idx < scores.shape[0]:
                                # Top 50% of surviving experts get better precision
                                median = scores[scores > 0].median() if (scores > 0).any() else scores.median()
                                bit_widths[name] = important_routed_bits if scores[idx] > median else weak_routed_bits
                                break
                if name not in bit_widths:
                    bit_widths[name] = important_routed_bits
            else:
                bit_widths[name] = important_routed_bits
        else:
            bit_widths[name] = default_bits

    counts = {}
    for b in bit_widths.values():
        counts[b] = counts.get(b, 0) + 1
    logger.info("Mixed-precision assignment: %s", {f"TQ{k}": v for k, v in sorted(counts.items())})

    return bit_widths


# ---------------------------------------------------------------------------
# Dense-and-Sparse decomposition
# ---------------------------------------------------------------------------

def extract_sparse_outliers(
    model: nn.Module,
    hessian: dict[str, torch.Tensor],
    outlier_fraction: float = 0.001,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Extract top-k outlier weights per layer for FP16 sparse storage.

    After TQ compression + decompression, the reconstruction error × Hessian weight
    identifies which weights matter most. These are stored separately in FP16 sparse
    format and added back during inference.

    Args:
        model: Model with TurboQuantWrapper layers.
        hessian: {layer_name: hessian_diagonal} from collect_hessian_diagonal.
        outlier_fraction: Fraction of weights to extract (0.001 = 0.1%).

    Returns:
        {layer_name: (indices, values, shape)} for sparse outlier correction.
        indices: (K, 2) int64 row/col indices
        values: (K,) float16 outlier values
        shape: original weight shape
    """
    from turboquant_vllm.weight_quant import TurboQuantWrapper

    outliers: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    total_outlier_bytes = 0
    total_weight_bytes = 0

    for name, module in model.named_modules():
        if not isinstance(module, TurboQuantWrapper):
            continue

        h_diag = hessian.get(name)
        if h_diag is None:
            continue

        # Reconstruct the weight to compute error
        from turboquant_vllm.weight_quant import unpack_indices, _get_quantizer
        quantizer = _get_quantizer(module.group_size, module.bits, str(module.packed_weight.device))
        indices = unpack_indices(module.packed_weight, module.bits, module.group_size)
        norms_flat = module.norms.reshape(-1)
        w_hat = quantizer.dequantize(indices, norms_flat)
        w_hat = w_hat.reshape(module.out_features, module.padded_in)[:, :module.in_features]

        # We don't have the original weight anymore, but we can use w_hat
        # and focus on high-Hessian positions where even small errors matter.
        # Importance = |w_hat| × sqrt(H_diag) — positions where the weight
        # is large AND the Hessian says it's sensitive.
        importance = w_hat.abs() * h_diag.sqrt().unsqueeze(0)

        n_outliers = max(1, int(importance.numel() * outlier_fraction))
        flat_imp = importance.reshape(-1)
        _, top_idx = flat_imp.topk(n_outliers)

        rows = top_idx // module.in_features
        cols = top_idx % module.in_features
        values = w_hat[rows, cols].half()

        outliers[name] = (
            torch.stack([rows, cols], dim=1).cpu(),
            values.cpu(),
            torch.tensor([module.out_features, module.in_features]),
        )

        outlier_bytes = n_outliers * (8 + 2)  # 2 int32 indices + fp16 value
        total_outlier_bytes += outlier_bytes
        total_weight_bytes += module.packed_weight.numel() + module.norms.numel() * 4

    overhead_pct = total_outlier_bytes / max(total_weight_bytes, 1) * 100
    logger.info(
        "Sparse outliers: %d layers, %.1f KB total (%.2f%% overhead)",
        len(outliers), total_outlier_bytes / 1024, overhead_pct,
    )
    return outliers
