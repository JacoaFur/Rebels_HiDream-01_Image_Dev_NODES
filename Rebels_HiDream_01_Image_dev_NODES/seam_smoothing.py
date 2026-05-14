"""
seam_smoothing.py — Rebel HiDream-01 seam smoothing module.

Implements the patch-coherence smoothing pass used by RebelHiDreamO1Sampler:

  1. Variable shift offsets per smoothing step (static / rotate / all)
  2. Strength schedule curves (constant / linear / cosine / front_loaded / late)
  3. Coherence delta surfaced via optional callback
  4. Adaptive trigger — skip cheap forward when latent is already coherent
  5. Multi-scale shift passes (coarse + fine)
  6. CFG-aware shifted forward (apply guidance on shifted grid too)

Plus a heatmap helper used by RebelHiDreamO1SeamVisualizer.

Designed for RTX 3070 / 8GB: every extra cost is gated to the seam window,
no extra VRAM peak beyond what the main forward already requires.
"""

import math
import torch
import einops


SHIFT_MODES = ["static", "rotate", "all"]
SCHEDULES = ["constant", "linear", "cosine", "front_loaded", "late"]


# ---------------------------------------------------------------------------
# 1. Variable shift offsets
# ---------------------------------------------------------------------------

def get_shift_offsets(h_patches, w_patches, smoothing_step_idx, mode="rotate"):
    """Return list of (h_shift, w_shift) tuples for this smoothing step.

    static : always centered half-shift.
    rotate : cycles through 4 distinct patterns, one per smoothing step.
    all    : returns all 4 patterns (used as multi-pass within one step).
    """
    base_patterns = [
        (max(1, h_patches // 2),  max(1, w_patches // 2)),   # center
        (max(1, h_patches // 3),  max(1, w_patches // 3)),   # third
        (max(1, h_patches // 4),  0),                         # vertical only
        (0,                        max(1, w_patches // 4)),  # horizontal only
    ]
    if mode == "static":
        return [base_patterns[0]]
    if mode == "all":
        return base_patterns
    # rotate (default)
    return [base_patterns[smoothing_step_idx % len(base_patterns)]]


# ---------------------------------------------------------------------------
# 2. Strength schedule curves
# ---------------------------------------------------------------------------

def get_smoothing_strength(base_strength, smoothing_step_idx, total_smoothing_steps,
                           schedule="constant"):
    """Per-step strength multiplier.

    smoothing_step_idx is 0-indexed within the smoothing window.
    Returns a float in [0, base_strength] (or slightly outside for `late`/`linear`).
    """
    if total_smoothing_steps <= 1:
        return base_strength

    t = smoothing_step_idx / (total_smoothing_steps - 1)  # 0..1

    if schedule == "constant":
        return base_strength
    if schedule == "linear":
        return base_strength * (1.0 - t)
    if schedule == "cosine":
        return base_strength * (0.5 * (1.0 + math.cos(math.pi * t)))
    if schedule == "front_loaded":
        return base_strength * (1.0 - t) ** 2
    if schedule == "late":
        return base_strength * t
    return base_strength


# ---------------------------------------------------------------------------
# 3+4. Coherence metric + adaptive trigger
# ---------------------------------------------------------------------------

def estimate_seam_intensity(z, h_patches, w_patches):
    """Cheap estimate of seam intensity using local first-order differences.
    Used by adaptive trigger BEFORE any forward pass."""
    z_img = einops.rearrange(z.float(), 'B (H W) C -> B C H W', H=h_patches, W=w_patches)
    dx = (z_img[..., 1:] - z_img[..., :-1]).abs().mean()
    dy = (z_img[..., 1:, :] - z_img[..., :-1, :]).abs().mean()
    return ((dx + dy) / 2.0).item()


def coherence_delta(z, x_pred_unshifted):
    """Magnitude of disagreement between current latent and shifted prediction.
    Computed during smoothing — surfaced via seam_callback."""
    return (x_pred_unshifted.to(torch.float32) - z.to(torch.float32)).abs().mean().item()


# ---------------------------------------------------------------------------
# Main entry point — called from pipeline.py
# ---------------------------------------------------------------------------

def apply_seam_smoothing(
    z, samples, ref_patches, t_pixeldit, sigma, dtype,
    h_patches, w_patches,
    smoothing_step_idx, total_smoothing_steps,
    base_strength, schedule, shift_mode,
    forward_once, guidance_scale, tgt_image_len,
    multiscale=False, cfg_aware=False, adaptive_threshold=0.0,
):
    """Apply seam smoothing to z. Returns (new_z, info).

    info keys:
      strength_used    : float, the strength actually applied
      coherence_delta  : float, mean |x_pred_s - z| across offsets
      seam_intensity   : float, the cheap pre-check value
      n_forwards       : int, how many model forwards this step cost
      skipped          : bool, True if smoothing was skipped
      offsets          : list of (h, w) tuples used
    """
    info = {
        "strength_used": 0.0,
        "coherence_delta": 0.0,
        "seam_intensity": 0.0,
        "n_forwards": 0,
        "skipped": False,
        "offsets": [],
    }

    # ---- Adaptive trigger: cheap pre-check ----
    if adaptive_threshold and adaptive_threshold > 0.0:
        intensity = estimate_seam_intensity(z, h_patches, w_patches)
        info["seam_intensity"] = intensity
        if intensity < adaptive_threshold:
            info["skipped"] = True
            return z, info

    # ---- Strength from schedule ----
    strength = get_smoothing_strength(
        base_strength, smoothing_step_idx, total_smoothing_steps, schedule
    )
    info["strength_used"] = strength
    if strength <= 0.0:
        info["skipped"] = True
        return z, info

    # ---- Build offset list ----
    offsets = list(get_shift_offsets(h_patches, w_patches, smoothing_step_idx, shift_mode))
    if multiscale:
        fine = (max(1, h_patches // 8), max(1, w_patches // 8))
        if fine not in offsets:
            offsets.append(fine)
    info["offsets"] = offsets

    z_img = einops.rearrange(z, 'B (H W) C -> B C H W', H=h_patches, W=w_patches)
    accumulated = None
    deltas = []
    n_fwd = 0

    use_cfg = cfg_aware and len(samples) > 1 and guidance_scale > 1.0

    for shift_h, shift_w in offsets:
        z_shifted = torch.roll(z_img, shifts=(shift_h, shift_w), dims=(2, 3))
        z_s = einops.rearrange(z_shifted, 'B C H W -> B (H W) C')

        # Forward pass on shifted grid — conditional
        if ref_patches is None:
            x_pred_cond_s = forward_once(samples[0], z_s.clone(), t_pixeldit)
        else:
            vinputs_s = torch.cat([z_s, ref_patches], dim=1)
            x_pred_cond_s = forward_once(samples[0], vinputs_s, t_pixeldit)
        n_fwd += 1

        # CFG-aware: also run uncond, apply guidance on shifted grid
        if use_cfg:
            if ref_patches is None:
                x_pred_uncond_s = forward_once(samples[1], z_s.clone(), t_pixeldit)
            else:
                vinputs_s_u = torch.cat([z_s, ref_patches], dim=1)
                x_pred_uncond_s = forward_once(samples[1], vinputs_s_u, t_pixeldit)
            n_fwd += 1

            v_cond_s = (x_pred_cond_s.float() - z_s.float()) / sigma
            v_uncond_s = (x_pred_uncond_s.float() - z_s.float()) / sigma
            v_guided_s = v_uncond_s + guidance_scale * (v_cond_s - v_uncond_s)
            # Reconstruct x_pred from guided velocity (x_pred = z + v * sigma)
            x_pred_s = (z_s.float() + v_guided_s * sigma).to(dtype)
        else:
            x_pred_s = x_pred_cond_s.to(dtype)

        # Unshift back to original frame
        x_pred_s_img = einops.rearrange(x_pred_s, 'B (H W) C -> B C H W',
                                        H=h_patches, W=w_patches)
        x_unshifted = torch.roll(x_pred_s_img, shifts=(-shift_h, -shift_w), dims=(2, 3))
        x_unshifted = einops.rearrange(x_unshifted, 'B C H W -> B (H W) C')

        deltas.append(coherence_delta(z, x_unshifted))

        if accumulated is None:
            accumulated = x_unshifted
        else:
            accumulated = accumulated + x_unshifted

    # Average across offsets
    blended_pred = accumulated / float(len(offsets))

    # Blend into z (this REPLACES the buggy second sched.step call)
    z_new = (1.0 - strength) * z + strength * blended_pred

    info["coherence_delta"] = sum(deltas) / len(deltas) if deltas else 0.0
    info["n_forwards"] = n_fwd

    return z_new, info


# ---------------------------------------------------------------------------
# Heatmap helper — used by RebelHiDreamO1SeamVisualizer
# ---------------------------------------------------------------------------

def seam_heatmap_from_image(image_tensor, patch_size=32, mode="patch_aligned"):
    """Generate a per-pixel seam-intensity heatmap from a finished image.

    image_tensor : (B, H, W, C) float in [0, 1] (ComfyUI IMAGE format)
    patch_size   : grid period to highlight (HiDream-01 = 32)
    mode         : "gradient" | "patch_aligned" | "patch_grid_only"

    Returns      : (B, H, W) float32 in [0, 1], higher = more seam-like.
    Plus a scalar seam_score (mean of the heatmap).
    """
    if image_tensor.dim() != 4:
        raise ValueError(f"expected (B,H,W,C) image, got shape {tuple(image_tensor.shape)}")

    # Luminance — Rec. 709 weights
    img = image_tensor.float()
    lum = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])  # B,H,W

    # First-order spatial gradients
    pad = torch.nn.functional.pad(lum.unsqueeze(1), (1, 1, 1, 1), mode='replicate')
    dx = (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]).abs().squeeze(1)
    dy = (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]).abs().squeeze(1)
    grad = (dx + dy) / 2.0  # B, H, W

    if mode == "gradient":
        heatmap = grad

    elif mode == "patch_grid_only":
        # Mask: 1 on grid boundaries, 0 elsewhere
        B, H, W = lum.shape
        mask_h = torch.zeros(H, device=lum.device)
        mask_w = torch.zeros(W, device=lum.device)
        mask_h[::patch_size] = 1.0
        mask_w[::patch_size] = 1.0
        grid = mask_h.unsqueeze(1) + mask_w.unsqueeze(0)  # H, W
        grid = (grid > 0).float().unsqueeze(0)  # 1, H, W
        heatmap = grad * grid

    else:  # patch_aligned (default) — gradient weighted by proximity to patch grid
        B, H, W = lum.shape
        # Distance from each pixel to nearest patch boundary, normalized
        ys = torch.arange(H, device=lum.device).float()
        xs = torch.arange(W, device=lum.device).float()
        dy_grid = torch.minimum(ys % patch_size, (patch_size - ys % patch_size) % patch_size)
        dx_grid = torch.minimum(xs % patch_size, (patch_size - xs % patch_size) % patch_size)
        # 1.0 at boundary, decays away (half-patch radius)
        weight_y = (1.0 - dy_grid / (patch_size / 2)).clamp_min(0.0)
        weight_x = (1.0 - dx_grid / (patch_size / 2)).clamp_min(0.0)
        weight = torch.maximum(weight_y.unsqueeze(1), weight_x.unsqueeze(0)).unsqueeze(0)
        heatmap = grad * weight

    # Normalize per-image to [0, 1]
    flat = heatmap.flatten(1)
    h_min = flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
    h_max = flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
    span = (h_max - h_min).clamp_min(1e-6)
    heatmap = (heatmap - h_min) / span

    seam_score = float(heatmap.mean().item())
    return heatmap, seam_score


def apply_colormap(heatmap, colormap="inferno"):
    """Convert (B, H, W) heatmap in [0, 1] to (B, H, W, 3) RGB image.
    Uses piecewise-linear LUTs — no matplotlib dependency, deterministic output."""
    h = heatmap.clamp(0.0, 1.0)
    if colormap == "grayscale":
        return h.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()

    # 6-stop linear LUTs — approximations of matplotlib colormaps, sized for
    # screenshot-quality output rather than scientific accuracy.
    STOPS = {
        "inferno": [
            (0.00, 0.00, 0.00, 0.00),
            (0.20, 0.20, 0.04, 0.30),
            (0.40, 0.55, 0.10, 0.40),
            (0.60, 0.85, 0.25, 0.20),
            (0.80, 0.98, 0.55, 0.10),
            (1.00, 1.00, 1.00, 0.60),
        ],
        "magma": [
            (0.00, 0.00, 0.00, 0.00),
            (0.25, 0.20, 0.08, 0.40),
            (0.50, 0.55, 0.18, 0.50),
            (0.75, 0.95, 0.45, 0.40),
            (1.00, 1.00, 0.98, 0.75),
        ],
        "viridis": [
            (0.00, 0.27, 0.00, 0.33),
            (0.25, 0.23, 0.32, 0.55),
            (0.50, 0.13, 0.57, 0.55),
            (0.75, 0.48, 0.80, 0.32),
            (1.00, 0.99, 0.91, 0.14),
        ],
    }
    stops = STOPS.get(colormap, STOPS["inferno"])

    flat = h.flatten()
    r = torch.zeros_like(flat)
    g = torch.zeros_like(flat)
    b = torch.zeros_like(flat)
    for i in range(len(stops) - 1):
        p0, r0, g0, b0 = stops[i]
        p1, r1, g1, b1 = stops[i + 1]
        mask = (flat >= p0) & (flat <= p1)
        if mask.any():
            t = (flat[mask] - p0) / max(p1 - p0, 1e-9)
            r[mask] = r0 + t * (r1 - r0)
            g[mask] = g0 + t * (g1 - g0)
            b[mask] = b0 + t * (b1 - b0)

    rgb = torch.stack([r, g, b], dim=-1).reshape(*h.shape, 3).contiguous()
    return rgb
