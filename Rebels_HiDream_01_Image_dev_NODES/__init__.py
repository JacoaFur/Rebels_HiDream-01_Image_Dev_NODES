"""
Rebels_HiDream_01_Image_Dev_NODES
ComfyUI custom nodes for HiDream-O1-Image-Dev (GGUF + bf16 safetensors).
"""
from .loader import RebelHiDreamO1Loader, RebelHiDreamO1LoaderHF
from .sampler import RebelHiDreamO1Sampler

NODE_CLASS_MAPPINGS = {
    "RebelHiDreamO1Loader":   RebelHiDreamO1Loader,
    "RebelHiDreamO1LoaderHF": RebelHiDreamO1LoaderHF,
    "RebelHiDreamO1Sampler":  RebelHiDreamO1Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelHiDreamO1Loader":   "Rebel HiDream-O1 Loader (GGUF)",
    "RebelHiDreamO1LoaderHF": "Rebel HiDream-O1 Loader (BF16 / Safetensors)",
    "RebelHiDreamO1Sampler":  "Rebel HiDream-O1 Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
