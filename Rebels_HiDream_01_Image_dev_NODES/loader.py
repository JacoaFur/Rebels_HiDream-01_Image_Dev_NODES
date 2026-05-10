"""
Rebel HiDream-O1 Loaders.
  - RebelHiDreamO1Loader      → GGUF path (uses gguf_ops.load_gguf)
  - RebelHiDreamO1LoaderHF    → bf16/safetensors path (standard from_pretrained)
Both return a HIDREAM_O1_MODEL handle the sampler can drive.
"""
import os
import sys
import torch
import folder_paths
from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map

from .gguf_ops import load_gguf


_DIFFUSION_MODELS_DIR = os.path.join(folder_paths.models_dir, "diffusion_models")
if "hidream_o1" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["hidream_o1"] = (
        [_DIFFUSION_MODELS_DIR],
        {".gguf", ".safetensors"},
    )


_SPECIAL_TOKENS = {
    "boi_token": "<|boi_token|>",
    "bor_token": "<|bor_token|>",
    "eor_token": "<|eor_token|>",
    "bot_token": "<|bot_token|>",
    "tms_token": "<|tms_token|>",
}


OFFLOAD_PRESETS = {
    "aggressive": {"cuda_gb": 5.5,  "cpu_gb": 13.0},
    "balanced":   {"cuda_gb": 9.5,  "cpu_gb": 20.0},
    "minimal":    {"cuda_gb": 22.0, "cpu_gb": 8.0},
}


def _ensure_upstream_path(upstream_path: str):
    if not os.path.isdir(upstream_path):
        raise FileNotFoundError(
            f"Upstream HiDream-O1-Image repo not found at:\n  {upstream_path}\n\n"
            f"Clone it with:\n  git clone https://github.com/HiDream-ai/HiDream-O1-Image.git"
        )
    needed = os.path.join(upstream_path, "models", "pipeline.py")
    if not os.path.isfile(needed):
        raise FileNotFoundError(
            f"upstream_repo_path must contain models/pipeline.py — missing at {needed}"
        )
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)


def _add_special_tokens(tokenizer):
    for attr, tok in _SPECIAL_TOKENS.items():
        setattr(tokenizer, attr, tok)


# ===========================================================================
# GGUF loader (unchanged — keep using if you want)
# ===========================================================================
class RebelHiDreamO1Loader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_name": (folder_paths.get_filename_list("hidream_o1"),),
                "tokenizer_path": ("STRING", {
                    "default": "HiDream-ai/HiDream-O1-Image-Dev",
                    "multiline": False,
                }),
                "upstream_repo_path": ("STRING", {
                    "default": r"C:\Users\noahp\HiDream-O1-Image",
                    "multiline": False,
                }),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "offload": (list(OFFLOAD_PRESETS.keys()), {"default": "aggressive"}),
                "compute_dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("HIDREAM_O1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Rebels/HiDream-O1"

    def load(self, gguf_name, tokenizer_path, upstream_repo_path, device, offload, compute_dtype):
        gguf_path = folder_paths.get_full_path("hidream_o1", gguf_name)
        if gguf_path is None or not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF '{gguf_name}' not found in {_DIFFUSION_MODELS_DIR}")

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }[compute_dtype]
        preset = OFFLOAD_PRESETS[offload]

        _ensure_upstream_path(upstream_repo_path)
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        from models.pipeline import generate_image, DEFAULT_TIMESTEPS

        config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
        with init_empty_weights():
            model = Qwen3VLForConditionalGeneration(config)

        print(f"[Rebels_HiDream_O1] Loading GGUF: {gguf_path}")
        missing, unexpected = load_gguf(gguf_path, model, target_dtype=torch_dtype)
        if missing:
            print(f"[Rebels_HiDream_O1] WARN: {len(missing)} missing keys: {list(missing)[:5]}")
        if unexpected:
            print(f"[Rebels_HiDream_O1] WARN: {len(unexpected)} unexpected keys: {list(unexpected)[:5]}")

        if device == "cuda" and torch.cuda.is_available() and offload != "minimal":
            max_memory = {
                0:     f"{preset['cuda_gb']:.1f}GiB",
                "cpu": f"{preset['cpu_gb']:.1f}GiB",
            }
            device_map = infer_auto_device_map(
                model, max_memory=max_memory, dtype=torch_dtype,
                no_split_module_classes=["Qwen3VLDecoderLayer"],
            )
            model = dispatch_model(model, device_map=device_map)
        elif device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        model.eval()

        processor = AutoProcessor.from_pretrained(tokenizer_path)
        tokenizer = (
            processor
            if isinstance(processor, PreTrainedTokenizerBase)
            else processor.tokenizer
        )
        _add_special_tokens(tokenizer)

        return ({
            "model":             model,
            "processor":         processor,
            "tokenizer":         tokenizer,
            "generate_image":    generate_image,
            "DEFAULT_TIMESTEPS": DEFAULT_TIMESTEPS,
            "device":            device,
            "dtype":             torch_dtype,
            "offload":           offload,
            "gguf_path":         gguf_path,
        },)


# ===========================================================================
# BF16 / safetensors loader (NEW — use this for the full-precision model)
# ===========================================================================
class RebelHiDreamO1LoaderHF:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "HiDream-ai/HiDream-O1-Image-Dev",
                    "multiline": False,
                    "tooltip": (
                        "HF repo id OR local folder containing config.json, "
                        "model.safetensors.index.json + shards, and tokenizer files. "
                        "If you've downloaded the model locally, paste the folder path here."
                    ),
                }),
                "upstream_repo_path": ("STRING", {
                    "default": r"C:\Users\noahp\HiDream-O1-Image",
                    "multiline": False,
                }),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "offload": (list(OFFLOAD_PRESETS.keys()), {"default": "aggressive"}),
                "offload_folder": ("STRING", {
                    "default": r"D:\AI_Tools\hidream_offload",
                    "tooltip": (
                        "Disk offload folder for layers that don't fit in VRAM+RAM. "
                        "Will be created if missing."
                    ),
                }),
            }
        }

    RETURN_TYPES = ("HIDREAM_O1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Rebels/HiDream-O1"

    def load(self, model_path, upstream_repo_path, dtype, offload, offload_folder):
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
        preset = OFFLOAD_PRESETS[offload]

        _ensure_upstream_path(upstream_repo_path)
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        from models.pipeline import generate_image, DEFAULT_TIMESTEPS

        os.makedirs(offload_folder, exist_ok=True)

        max_memory = {
            0:     f"{preset['cuda_gb']:.1f}GiB",
            "cpu": f"{preset['cpu_gb']:.1f}GiB",
        }

        print(f"[Rebels_HiDream_O1] Loading BF16 model from: {model_path}")
        print(f"[Rebels_HiDream_O1] Memory budget: {max_memory}, disk offload: {offload_folder}")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            max_memory=max_memory,
            offload_folder=offload_folder,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = (
            processor
            if isinstance(processor, PreTrainedTokenizerBase)
            else processor.tokenizer
        )
        _add_special_tokens(tokenizer)

        return ({
            "model":             model,
            "processor":         processor,
            "tokenizer":         tokenizer,
            "generate_image":    generate_image,
            "DEFAULT_TIMESTEPS": DEFAULT_TIMESTEPS,
            "device":            "cuda",
            "dtype":             torch_dtype,
            "offload":           offload,
            "model_path":        model_path,
        },)
