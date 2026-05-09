"""
Upfront GGUF dequant ops for HiDream-O1.
Dequantizes all weights to native PyTorch tensors during load to prevent CPU bottlenecking.
"""
import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType


NATIVE_TYPES = {
    GGMLQuantizationType.F32,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
}


def _native_to_torch(tensor_data, tensor_type, target_dtype):
    if tensor_type == GGMLQuantizationType.F32:
        arr = np.frombuffer(bytes(tensor_data), dtype=np.float32)
        return torch.from_numpy(arr.copy()).to(target_dtype)
    if tensor_type == GGMLQuantizationType.F16:
        arr = np.frombuffer(bytes(tensor_data), dtype=np.float16)
        return torch.from_numpy(arr.copy()).to(target_dtype)
    if tensor_type == GGMLQuantizationType.BF16:
        arr = np.frombuffer(bytes(tensor_data), dtype=np.uint16)
        t = torch.from_numpy(arr.copy()).view(torch.bfloat16)
        return t.to(target_dtype)
    raise ValueError(f"Not a native type: {tensor_type}")


def _dequant_to_torch(tensor_data, tensor_type, shape, target_dtype):
    np_arr = gguf.quants.dequantize(np.array(tensor_data), tensor_type)
    np_arr = np_arr.reshape(shape)
    return torch.from_numpy(np_arr.astype(np.float32)).to(target_dtype)


def load_gguf(gguf_path, model, target_dtype=torch.bfloat16):
    reader = gguf.GGUFReader(gguf_path)
    
    print("[Rebels_HiDream_O1] Dequantizing GGUF to native PyTorch upfront. This will take a moment...")
    
    direct_state_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        torch_shape = list(tensor.shape)[::-1]
        
        if tensor.tensor_type not in NATIVE_TYPES:
            # Dequantize ONCE during load, not during the forward pass
            tensor_pt = _dequant_to_torch(tensor.data, tensor.tensor_type, torch_shape, target_dtype)
        else:
            tensor_pt = _native_to_torch(tensor.data, tensor.tensor_type, target_dtype)
            tensor_pt = tensor_pt.reshape(torch_shape)
            
        direct_state_dict[name] = tensor_pt

    # Load directly into the native upstream model (Bypass patch_model_for_gguf)
    missing, unexpected = model.load_state_dict(direct_state_dict, strict=False, assign=True)
    return missing, unexpected