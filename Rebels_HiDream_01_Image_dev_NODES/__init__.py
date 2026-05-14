"""
Rebels_HiDream_01_Image_Dev_NODES
ComfyUI custom nodes for HiDream-01-Image-Dev (GGUF + bf16 safetensors).
"""
from .loader import RebelHiDreamO1Loader, RebelHiDreamO1LoaderHF
from .sampler import RebelHiDreamO1Sampler
from .nodes_extra import RebelHiDreamO1LoraStackInjector, RebelHiDreamO1SeamVisualizer

NODE_CLASS_MAPPINGS = {
    "RebelHiDreamO1Loader":            RebelHiDreamO1Loader,
    "RebelHiDreamO1LoaderHF":          RebelHiDreamO1LoaderHF,
    "RebelHiDreamO1Sampler":           RebelHiDreamO1Sampler,
    "RebelHiDreamO1LoraStackInjector": RebelHiDreamO1LoraStackInjector,
    "RebelHiDreamO1SeamVisualizer":    RebelHiDreamO1SeamVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelHiDreamO1Loader":            "Rebel HiDream-01 Loader (GGUF)",
    "RebelHiDreamO1LoaderHF":          "Rebel HiDream-01 Loader (BF16 / Safetensors)",
    "RebelHiDreamO1Sampler":           "Rebel HiDream-01 Sampler",
    "RebelHiDreamO1LoraStackInjector": "Rebel HiDream-01 LoRA Stack Injector",
    "RebelHiDreamO1SeamVisualizer":    "Rebel HiDream-01 Seam Visualizer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
