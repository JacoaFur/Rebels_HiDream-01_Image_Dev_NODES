"""
Rebel HiDream-O1 Sampler.
Wraps upstream models.pipeline.generate_image() and converts PIL → IMAGE tensor.
"""
import numpy as np
import torch


class RebelHiDreamO1Sampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":   ("HIDREAM_O1_MODEL",),
                "prompt":  ("STRING", {
                    "multiline": True,
                    "default": "A cinematic photograph of a fox in a snowy forest at golden hour",
                }),
                "width":   ("INT",   {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height":  ("INT",   {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps":   ("INT",   {"default": 28,   "min": 1,   "max": 100}),
                "cfg":     ("FLOAT", {"default": 0.0,  "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed":    ("INT",   {"default": 32,   "min": 0,   "max": 0xffffffffffffffff}),
            },
            "optional": {
                "shift":              ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "scheduler_name":     (["flash", "default"], {"default": "flash"}),
                "noise_scale_start":  ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "noise_scale_end":    ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "noise_clip_std":     ("FLOAT", {"default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Rebels/HiDream-O1"

    def sample(self, model, prompt, width, height, steps, cfg, seed,
               shift=1.0, scheduler_name="flash",
               noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5):
        generate_image    = model["generate_image"]
        DEFAULT_TIMESTEPS = model["DEFAULT_TIMESTEPS"]

        timesteps_list = DEFAULT_TIMESTEPS if scheduler_name == "flash" else None

        extra_kwargs = {}
        if scheduler_name == "flash":
            extra_kwargs["noise_scale_start"] = noise_scale_start
            extra_kwargs["noise_scale_end"]   = noise_scale_end
            extra_kwargs["noise_clip_std"]    = noise_clip_std

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

        arr = np.asarray(pil_image.convert("RGB")).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).unsqueeze(0)
        return (image_tensor,)