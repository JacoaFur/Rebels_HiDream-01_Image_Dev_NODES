"""
lora_stack.py — LoRA merge/unmerge for HiDream-01 (Qwen3VL backbone).

HiDream-01 isn't a diffusers pipeline, so we can't lean on diffusers'
`load_lora_weights`. This module handles raw safetensors LoRA files
(kohya / PEFT naming variants) by merging directly into Linear weights,
with a state tracker so the stack can be swapped without reloading
the 16GB base model.

Memory note: deltas are stored on CPU for unmerge, never duplicated on GPU.
"""

import torch
from safetensors.torch import load_file


# State key attached to the loaded model dict in nodes_extra.py
STATE_ATTR = "_rebel_lora_stack_state"


# ---------------------------------------------------------------------------
# Key parsing — handle kohya, PEFT, and diffusers-style names
# ---------------------------------------------------------------------------

_PREFIX_STRIPS = (
    "base_model.model.",
    "lora_unet_",       # kohya
    "lora_te_",
    "model.diffusion_model.",
    "transformer.",
)

def _strip_prefix(key):
    for p in _PREFIX_STRIPS:
        if key.startswith(p):
            return key[len(p):]
    return key


def _normalize_module_path(key):
    """Map kohya's underscore-joined path back to dotted submodule path.
    kohya: lora_unet_blocks_0_attn_to_q -> blocks.0.attn.to_q
    PEFT : blocks.0.attn.to_q.lora_A.weight -> blocks.0.attn.to_q  (already dotted)
    """
    key = _strip_prefix(key)
    # PEFT/diffusers style is already dotted
    if "." in key and "_" not in key.split(".")[0]:
        return key
    return key.replace("_", ".")


# ---------------------------------------------------------------------------
# Pair extraction from LoRA state dict
# ---------------------------------------------------------------------------

def _extract_lora_pairs(lora_sd):
    """Group LoRA tensors by target module. Returns {module_path: {A, B, alpha}}."""
    pairs = {}

    def _put(base_key, slot, value):
        clean = _normalize_module_path(base_key)
        pairs.setdefault(clean, {})[slot] = value

    for key, tensor in lora_sd.items():
        if key.endswith(".lora_A.weight") or key.endswith(".lora_down.weight"):
            base = key.rsplit(".lora_", 1)[0]
            _put(base, "A", tensor)
        elif key.endswith(".lora_B.weight") or key.endswith(".lora_up.weight"):
            base = key.rsplit(".lora_", 1)[0]
            _put(base, "B", tensor)
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            alpha_val = tensor.item() if tensor.numel() == 1 else float(tensor.flatten()[0])
            _put(base, "alpha", alpha_val)

    return pairs


# ---------------------------------------------------------------------------
# Find a Linear submodule by dotted path, tolerant of common renames
# ---------------------------------------------------------------------------

def _find_linear(model, dotted_path):
    """Return the nn.Linear at dotted_path, or None if not found / not Linear."""
    try:
        mod = model.get_submodule(dotted_path)
    except (AttributeError, ValueError):
        return None
    if isinstance(mod, torch.nn.Linear):
        return mod
    # Some weights live under .base_layer (PEFT-wrapped, edge case)
    if hasattr(mod, "base_layer") and isinstance(mod.base_layer, torch.nn.Linear):
        return mod.base_layer
    return None


# ---------------------------------------------------------------------------
# Core merge / unmerge
# ---------------------------------------------------------------------------

def merge_lora(model, lora_sd, strength):
    """Merge a single LoRA into the model in-place. Returns list of
    (dotted_path, delta_cpu) tuples for later unmerge."""
    if abs(strength) < 1e-8:
        return []

    pairs = _extract_lora_pairs(lora_sd)
    applied = []
    matched = 0
    unmatched_examples = []

    for path, parts in pairs.items():
        if "A" not in parts or "B" not in parts:
            continue
        linear = _find_linear(model, path)
        if linear is None:
            if len(unmatched_examples) < 3:
                unmatched_examples.append(path)
            continue

        W = linear.weight  # [out, in]
        A = parts["A"].to(W.device, W.dtype)   # [r, in]
        B = parts["B"].to(W.device, W.dtype)   # [out, r]
        rank = A.shape[0]
        alpha = parts.get("alpha", float(rank))
        scale = (alpha / rank) * strength

        delta = (B @ A) * scale  # [out, in]
        if delta.shape != W.shape:
            # Shape mismatch — skip this entry rather than crash
            continue

        with torch.no_grad():
            W.add_(delta)
        applied.append((path, delta.detach().to("cpu")))
        matched += 1

    print(f"[Rebels_HiDream_01] LoRA merge: {matched}/{len(pairs)} pairs applied "
          f"(strength={strength:.3f})")
    if matched == 0 and unmatched_examples:
        print(f"[Rebels_HiDream_01] No matches found. Example unmatched paths: "
              f"{unmatched_examples}")

    return applied


def unmerge_lora(model, applied):
    """Reverse a previously applied merge. `applied` is the list returned by merge_lora."""
    for path, delta_cpu in applied:
        linear = _find_linear(model, path)
        if linear is None:
            continue
        with torch.no_grad():
            linear.weight.sub_(delta_cpu.to(linear.weight.device, linear.weight.dtype))


# ---------------------------------------------------------------------------
# Stack-level helpers (used by the LoRA Stack Injector node)
# ---------------------------------------------------------------------------

def stack_fingerprint(stack):
    """Hashable identity of a LoRA stack — used to detect re-application.
    stack: list of (lora_path, strength, bypass) tuples."""
    return tuple(
        (path, round(float(s), 6), bool(b))
        for path, s, b in stack
    )


def apply_stack(model, stack):
    """Apply a full stack. Returns list-of-list of applied deltas, one per slot.
    Bypassed slots contribute an empty list."""
    per_slot = []
    for path, strength, bypass in stack:
        if bypass or not path or abs(strength) < 1e-8:
            per_slot.append([])
            continue
        try:
            sd = load_file(path)
        except Exception as e:
            print(f"[Rebels_HiDream_01] LoRA load failed: {path}: {e}")
            per_slot.append([])
            continue
        per_slot.append(merge_lora(model, sd, strength))
    return per_slot


def unmerge_stack(model, per_slot_applied):
    """Reverse a previously-applied stack (in reverse order for numerical stability)."""
    for applied in reversed(per_slot_applied):
        unmerge_lora(model, applied)
