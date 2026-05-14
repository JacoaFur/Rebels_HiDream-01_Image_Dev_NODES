"""
nodes_extra.py — Additional ComfyUI nodes for the Rebel HiDream-01 pack:

  - RebelHiDreamO1LoraStackInjector  : 4-slot LoRA stack with per-slot
                                        strength + bypass. Tracks applied
                                        state on the model handle so the
                                        16GB base never needs reloading.

  - RebelHiDreamO1SeamVisualizer     : renders a patch-grid seam heatmap
                                        from a finished IMAGE plus a scalar
                                        seam_score. Useful for A/B comparing
                                        smoothing settings or as marketing
                                        screenshots.

Wire into your existing __init__.py NODE_CLASS_MAPPINGS — see snippet at end
of file.
"""

import folder_paths
import torch

from .lora_stack import apply_stack, unmerge_stack, stack_fingerprint, STATE_ATTR
from .seam_smoothing import seam_heatmap_from_image, apply_colormap


# ===========================================================================
# 4-slot LoRA Stack Injector
# ===========================================================================

class RebelHiDreamO1LoraStackInjector:
    """Apply up to 4 LoRAs to a HIDREAM_O1_MODEL handle with per-slot strength
    and bypass toggle. Bypassed slots are skipped — no forward-pass cost, no
    weight modification.

    Re-running the node with the same stack is a no-op (fingerprint match).
    Changing any slot triggers a clean unmerge of the previous stack followed
    by a fresh merge of the new one — base model weights stay in-place, no
    16GB reload."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            lora_files = folder_paths.get_filename_list("loras")
        except Exception:
            lora_files = []
        none_or_lora = ["None"] + list(lora_files)

        strength = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        bypass = ("BOOLEAN", {"default": False, "label_on": "bypass", "label_off": "active"})

        return {
            "required": {
                "model": ("HIDREAM_O1_MODEL",),
                "lora_1_name":     (none_or_lora,),
                "lora_1_strength": strength,
                "lora_1_bypass":   bypass,
                "lora_2_name":     (none_or_lora,),
                "lora_2_strength": strength,
                "lora_2_bypass":   bypass,
                "lora_3_name":     (none_or_lora,),
                "lora_3_strength": strength,
                "lora_3_bypass":   bypass,
                "lora_4_name":     (none_or_lora,),
                "lora_4_strength": strength,
                "lora_4_bypass":   bypass,
            }
        }

    RETURN_TYPES = ("HIDREAM_O1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "Rebels/HiDream-01"

    def inject(self, model,
               lora_1_name, lora_1_strength, lora_1_bypass,
               lora_2_name, lora_2_strength, lora_2_bypass,
               lora_3_name, lora_3_strength, lora_3_bypass,
               lora_4_name, lora_4_strength, lora_4_bypass):

        slots = [
            (lora_1_name, lora_1_strength, lora_1_bypass),
            (lora_2_name, lora_2_strength, lora_2_bypass),
            (lora_3_name, lora_3_strength, lora_3_bypass),
            (lora_4_name, lora_4_strength, lora_4_bypass),
        ]

        # Resolve names to paths, normalize bypass + missing-file cases
        stack = []
        for name, s, bp in slots:
            if bp or name in (None, "", "None"):
                stack.append(("", 0.0, True))
                continue
            path = folder_paths.get_full_path("loras", name)
            if path is None:
                print(f"[Rebels_HiDream_01] WARN: LoRA '{name}' not found, "
                      f"slot bypassed")
                stack.append(("", 0.0, True))
                continue
            stack.append((path, float(s), False))

        new_fp = stack_fingerprint(stack)
        existing = model.get(STATE_ATTR)

        # Fast path: identical to last run
        if existing is not None and existing.get("fingerprint") == new_fp:
            return (model,)

        # Unmerge previous stack if any
        if existing is not None:
            try:
                unmerge_stack(model["model"], existing.get("applied", []))
            except Exception as e:
                print(f"[Rebels_HiDream_01] WARN: unmerge failed: {e}")

        # Apply new stack
        applied = apply_stack(model["model"], stack)
        model[STATE_ATTR] = {"fingerprint": new_fp, "applied": applied}

        active = sum(1 for path, _, bp in stack if path and not bp)
        print(f"[Rebels_HiDream_01] LoRA stack: {active}/4 slots active")

        return (model,)


# ===========================================================================
# Seam Visualizer
# ===========================================================================

class RebelHiDreamO1SeamVisualizer:
    """Render a heatmap showing where patch-grid seams concentrate in an image,
    plus an aggregate seam_score (0 = clean, higher = more visible tiling).

    Run two passes (smoothing off vs on), feed each through this node, and
    compare the resulting heatmaps side by side to verify your seam-smoothing
    settings are actually doing something. The score is a single-number
    summary good for headlines / tweets / A-B chart axes.

    Modes:
      patch_aligned   - gradient magnitude weighted by proximity to patch grid
                        (best general view; default)
      gradient        - raw gradient magnitude with no grid weighting
                        (sanity check — should NOT show grid structure if
                        smoothing is working)
      patch_grid_only - gradient sampled strictly on patch boundaries
                        (most aggressive; isolates pure seam contribution)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "patch_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8,
                                       "tooltip": "Patch period to highlight. "
                                                  "HiDream-01 = 32."}),
                "mode": (["patch_aligned", "gradient", "patch_grid_only"],
                         {"default": "patch_aligned"}),
                "colormap": (["inferno", "magma", "viridis", "grayscale"],
                             {"default": "inferno"}),
                "overlay_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0.0 = pure heatmap; 1.0 = blend with original."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("heatmap", "seam_score")
    FUNCTION = "visualize"
    CATEGORY = "Rebels/HiDream-01"

    def visualize(self, image, patch_size, mode, colormap, overlay_strength):
        heatmap, score = seam_heatmap_from_image(image, patch_size=patch_size, mode=mode)
        rgb = apply_colormap(heatmap, colormap=colormap)

        if overlay_strength > 0.0:
            rgb = (1.0 - overlay_strength) * rgb + overlay_strength * image.float()
            rgb = rgb.clamp(0.0, 1.0)

        print(f"[Rebels_HiDream_01] Seam score: {score:.4f} "
              f"(mode={mode}, patch={patch_size})")

        return (rgb.float().contiguous(), float(score))


# ===========================================================================
# Mapping registration — copy/merge into your existing __init__.py
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "RebelHiDreamO1LoraStackInjector": RebelHiDreamO1LoraStackInjector,
    "RebelHiDreamO1SeamVisualizer":    RebelHiDreamO1SeamVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelHiDreamO1LoraStackInjector": "Rebel HiDream-01 LoRA Stack Injector",
    "RebelHiDreamO1SeamVisualizer":    "Rebel HiDream-01 Seam Visualizer",
}
