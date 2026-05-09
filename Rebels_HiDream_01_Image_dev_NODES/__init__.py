"""
Rebels_HiDream_01_Image_Dev_NODES
ComfyUI custom nodes for HiDream-O1-Image-Dev (smthem Q6_K GGUF).
"""
from .loader import RebelHiDreamO1Loader
from .sampler import RebelHiDreamO1Sampler

NODE_CLASS_MAPPINGS = {
    "RebelHiDreamO1Loader":  RebelHiDreamO1Loader,
    "RebelHiDreamO1Sampler": RebelHiDreamO1Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelHiDreamO1Loader":  "Rebel HiDream-O1 Loader (GGUF)",
    "RebelHiDreamO1Sampler": "Rebel HiDream-O1 Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]