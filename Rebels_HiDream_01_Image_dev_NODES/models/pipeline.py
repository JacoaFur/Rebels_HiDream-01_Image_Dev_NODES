import torch
import einops
import numpy as np
import tqdm
import math
from PIL import Image
import torchvision.transforms.v2 as transforms
from models.fm_solvers_unipc import FlowUniPCMultistepScheduler
from models.flash_scheduler import FlashFlowMatchEulerDiscreteScheduler
from models.utils import resize_pilimage, calculate_dimensions, get_rope_index_fix_point, find_closest_resolution

# Rebel seam smoothing module 
try:
    from .seam_smoothing import apply_seam_smoothing, SHIFT_MODES, SCHEDULES
except ImportError:
    from seam_smoothing import apply_seam_smoothing, SHIFT_MODES, SCHEDULES

TIMESTEP_TOKEN_NUM = 1
NOISE_SCALE = 8.0
T_EPS = 0.001
CONDITION_IMAGE_SIZE = 384
PATCH_SIZE = 32

# REQUIRED: Prevents ImportError in loader.py
DEFAULT_TIMESTEPS = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
]

TENSOR_TRANSFORM = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.5], [0.5]),
])

# ======================================================================
# FIXED MATH: Shift-Awareness to stop "Gray Soup"
# ======================================================================
def _sigmas_simple(num_steps, shift=1.0):
    ts = torch.linspace(999, 1, num_steps)
    # Correct Flow-Matching shift formula
    ts = (shift * ts) / (1 + (shift - 1) * ts / 1000.0)
    return ts / 1000.0

def _sigmas_karras(num_steps, shift=1.0):
    rho = 7.0
    sigma_min, sigma_max = 0.001, 0.999
    ramp = torch.linspace(0, 1, num_steps)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return (shift * sigmas) / (1 + (shift - 1) * sigmas)

SIGMA_SCHEDULE_MAP = {
    "normal": lambda n, s: None,
    "simple": _sigmas_simple,
    "karras": _sigmas_karras,
    "exponential": _sigmas_simple,
}

STOCHASTIC_SAMPLERS = {"euler_ancestral", "dpmpp_sde", "dpmpp_2m_sde"}
UNIPC_SAMPLERS = {"uni_pc", "uni_pc_bh2", "deis"}

# ======================================================================
# Logic Functions
# ======================================================================

def build_scheduler(num_inference_steps, shift, device, sampler_name="euler", scheduler_name="normal"):
    if sampler_name in UNIPC_SAMPLERS:
        sched = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)
    else:
        sched = FlashFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    
    sched.set_timesteps(num_inference_steps, device=device)
    sigma_fn = SIGMA_SCHEDULE_MAP.get(scheduler_name, _sigmas_simple)
    custom_sigmas = sigma_fn(num_inference_steps, shift)
    
    if custom_sigmas is not None:
        sched.timesteps = (custom_sigmas * 1000.0).long().clamp(1, 999).to(device)
        sched.sigmas = torch.cat([custom_sigmas, torch.tensor([0.0])]).to(device)
    return sched

def build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config):
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    
    messages = [{"role": "user", "content": prompt}]
    template_caption = (processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        + getattr(tokenizer, "boi_token", "<|boi_token|>")
                        + getattr(tokenizer, "tms_token", "<|tms_token|>") * TIMESTEP_TOKEN_NUM)
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)
    
    image_grid_thw = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)
    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)
    
    position_ids, _ = get_rope_index_fix_point(1, image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=image_grid_thw, video_grid_thw=None, attention_mask=None, skip_vision_start_token=[1])
    
    txt_seq_len = input_ids.shape[-1]
    token_types = torch.zeros((1, position_ids.shape[-1]), dtype=input_ids.dtype)
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len + image_len] = 1
    
    return {'input_ids': input_ids, 'position_ids': position_ids, 'token_types': (token_types > 0).to(token_types.dtype), 'vinput_mask': (token_types == 1)}

# ======================================================================
# Main generate_image Function
# ======================================================================

@torch.no_grad()
def generate_image(model, processor, prompt: str, ref_image_paths: list = None, height: int = 2048, width: int = 2048,
                   num_inference_steps: int = 50, guidance_scale: float = 5.0, shift: float = 3.0,
                   sampler_name: str = "euler", scheduler_name: str = "normal", seed: int = 42,
                   noise_scale_start: float = 7.5, noise_scale_end: float = 7.5, noise_clip_std: float = 2.5,
                   seam_smooth_steps: int = 0, seam_smooth_strength: float = 0.5, **kwargs):
    device = model.device
    dtype = torch.bfloat16
    w, h = find_closest_resolution(width, height)
    h_patches, w_patches = h // PATCH_SIZE, w // PATCH_SIZE
    tgt_image_len = h_patches * w_patches

    # 1. Prepare Samples
    cond_sample = build_t2i_text_sample(prompt, h, w, processor.tokenizer, processor, model.config)
    samples = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond_sample.items()}]
    if guidance_scale > 1.0:
        uncond = build_t2i_text_sample(" ", h, w, processor.tokenizer, processor, model.config)
        samples.append({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in uncond.items()})

    # 2. Setup Noise & Scheduler
    torch.manual_seed(seed + 1)
    noise = noise_scale_start * torch.randn((1, 3, h, w), device="cpu").to(device, dtype)
    z = einops.rearrange(noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)', p1=PATCH_SIZE, p2=PATCH_SIZE)
    
    sched = build_scheduler(num_inference_steps, shift, device, sampler_name, scheduler_name)
    noise_scale_schedule = np.linspace(noise_scale_start, noise_scale_end, len(sched.timesteps))

    # 3. FIXED: Signature has 3 args and result is sliced to 4096
    def forward_once(sample, z_in, t_px):
        with torch.autocast(device.type, dtype=dtype, cache_enabled=False):
            out = model(input_ids=sample['input_ids'], position_ids=sample['position_ids'], vinputs=z_in,
                        timestep=t_px.reshape(-1).to(device), token_types=sample['token_types'], use_flash_attn=False)
        return out.x_pred[0, sample['vinput_mask'][0]][-tgt_image_len:].unsqueeze(0)

    # 4. Sampling Loop
    for step_idx, step_t in enumerate(tqdm.tqdm(sched.timesteps, desc="Generating")):
        t_pixeldit = 1.0 - step_t.float() / 1000.0
        sigma = (step_t.float() / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)
        
        x_cond = forward_once(samples[0], z, t_pixeldit)
        v_cond = (x_cond.to(dtype=torch.float32) - z.to(dtype=torch.float32)) / sigma

        if len(samples) > 1:
            x_uncond = forward_once(samples[1], z, t_pixeldit)
            v_uncond = (x_uncond.to(dtype=torch.float32) - z.to(dtype=torch.float32)) / sigma
            v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v_guided = v_cond

        # 5. FIXED: Correct argument name is 's_noise'
        z = sched.step(-v_guided.float(), step_t.to(dtype=torch.float32), z.float(), 
                       s_noise=noise_scale_schedule[step_idx], # NOT s_noise_scale
                       noise_clip_std=noise_clip_std, 
                       return_dict=False)[0].to(dtype)

        # 6. Seam Smoothing
        if seam_smooth_steps > 0 and step_idx >= (len(sched.timesteps) - seam_smooth_steps):
            z_img = einops.rearrange(z, 'B (H W) C -> B C H W', H=h_patches, W=w_patches)
            shift_h, shift_w = h_patches // 2, w_patches // 2
            z_shifted = torch.roll(z_img, shifts=(shift_h, shift_w), dims=(2, 3))
            z_s = einops.rearrange(z_shifted, 'B C H W -> B (H W) C')
            x_pred_s = forward_once(samples[0], z_s.clone(), t_pixeldit)
            z_s_img = einops.rearrange(x_pred_s.to(dtype), 'B (H W) C -> B C H W', H=h_patches, W=w_patches)
            z_unshifted = einops.rearrange(torch.roll(z_s_img, shifts=(-shift_h, -shift_w), dims=(2, 3)), 'B C H W -> B (H W) C')
            z = (1.0 - seam_smooth_strength) * z + seam_smooth_strength * z_unshifted

    # 7. Final Decode
    img = (z + 1) / 2
    img = einops.rearrange(img.cpu().float(), 'B (H W) (C p1 p2) -> B C (H p1) (W p2)', 
                           H=h_patches, W=w_patches, p1=PATCH_SIZE, p2=PATCH_SIZE)
    arr = np.round(np.clip(img[0].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")
