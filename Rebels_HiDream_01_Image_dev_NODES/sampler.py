"""
Rebel HiDream-O1 Sampler.
Wraps upstream models.pipeline.generate_image() and converts PIL → IMAGE tensor.

Resolution presets match HiDream-O1's PREDEFINED_RESOLUTIONS — anything outside
these snaps to the nearest aspect ratio inside the pipeline. The "Custom (force)"
option monkey-patches find_closest_resolution to bypass snapping for that one
call. Use at your own risk: outside trained resolutions the model often emits
black or garbled images.
"""
import numpy as np
import torch


# Mirror of upstream PREDEFINED_RESOLUTIONS, with friendly labels.
RESOLUTION_PRESETS = {
    "2048x2048 (1:1 square)":      (2048, 2048),
    "2304x1728 (4:3 landscape)":   (2304, 1728),
    "1728x2304 (3:4 portrait)":    (1728, 2304),
    "2560x1440 (16:9 landscape)":  (2560, 1440),
    "1440x2560 (9:16 portrait)":   (1440, 2560),
    "2496x1664 (3:2 landscape)":   (2496, 1664),
    "1664x2496 (2:3 portrait)":    (1664, 2496),
    "3104x1312 (21:9 ultrawide)":  (3104, 1312),
    "1312x3104 (9:21 tall)":       (1312, 3104),
    "2304x1792 (~9:7 landscape)":  (2304, 1792),
    "1792x2304 (~7:9 portrait)":   (1792, 2304),
    "Custom (force, may break)":   None,
}


class RebelHiDreamO1Sampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":  ("HIDREAM_O1_MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic photograph of a fox in a snowy forest at golden hour",
                }),
                "resolution_preset": (list(RESOLUTION_PRESETS.keys()),
                                      {"default": "2048x2048 (1:1 square)"}),
                "custom_width":  ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                          "tooltip": "Only used when resolution_preset is 'Custom (force…)'."}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64,
                                          "tooltip": "Only used when resolution_preset is 'Custom (force…)'."}),
                "steps": ("INT",   {"default": 28,  "min": 1,   "max": 100}),
                "cfg":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed":  ("INT",   {"default": 32,  "min": 0,   "max": 0xffffffffffffffff}),
            },
            "optional": {
                "shift":             ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "scheduler_name":    (["flash", "default"], {"default": "flash"}),
                "noise_scale_start": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "noise_scale_end":   ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "noise_clip_std":    ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Rebels/HiDream-O1"

    def sample(self, model, prompt, resolution_preset,
               custom_width, custom_height, steps, cfg, seed,
               shift=1.0, scheduler_name="flash",
               noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5):
        generate_image    = model["generate_image"]
        DEFAULT_TIMESTEPS = model["DEFAULT_TIMESTEPS"]

        preset = RESOLUTION_PRESETS[resolution_preset]
        force_custom = preset is None

        if force_custom:
            width, height = custom_width, custom_height
            print(f"[Rebels_HiDream_O1] Custom resolution forced: {width}x{height} "
                  f"(bypassing find_closest_resolution; may produce black/garbled output)")
        else:
            width, height = preset

        timesteps_list = DEFAULT_TIMESTEPS if scheduler_name == "flash" else None

        extra_kwargs = {}
        if scheduler_name == "flash":
            extra_kwargs["noise_scale_start"] = noise_scale_start
            extra_kwargs["noise_scale_end"]   = noise_scale_end
            extra_kwargs["noise_clip_std"]    = noise_clip_std

        # Monkey-patch the resolution snapper if the user asked for custom dims.
        # We patch it on the pipeline module (which imported it) so the in-flight
        # generate_image call sees the no-op version.
        patched_originals = {}
        if force_custom:
            try:
                import models.pipeline as _pipeline
                if hasattr(_pipeline, "find_closest_resolution"):
                    patched_originals["pipeline"] = _pipeline.find_closest_resolution
                    _pipeline.find_closest_resolution = lambda w, h: (w, h)
                import models.utils as _utils
                if hasattr(_utils, "find_closest_resolution"):
                    patched_originals["utils"] = _utils.find_closest_resolution
                    _utils.find_closest_resolution = lambda w, h: (w, h)
            except Exception as e:
                print(f"[Rebels_HiDream_O1] Could not patch find_closest_resolution: {e}")

        try:
            pil_image = generate_image(
                model=model["model"],
                processor=model["processor"],
                prompt=prompt,
                ref_image_paths=[],
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                shift=shift,
                timesteps_list=timesteps_list,
                scheduler_name=scheduler_name,
                seed=seed,
                keep_original_aspect=False,
                **extra_kwargs,
            )
        finally:
            # Always restore upstream functions, even if generate_image raised.
            if "pipeline" in patched_originals:
                import models.pipeline as _pipeline
                _pipeline.find_closest_resolution = patched_originals["pipeline"]
            if "utils" in patched_originals:
                import models.utils as _utils
                _utils.find_closest_resolution = patched_originals["utils"]

        arr = np.asarray(pil_image.convert("RGB")).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).unsqueeze(0)
        return (image_tensor,)
