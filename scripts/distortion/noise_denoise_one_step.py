import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

import dotenv
dotenv.load_dotenv()

# 1. Initialization
# Load the standard SDXL pipeline and swap the scheduler to DDIMScheduler.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")
pipe.vae.to(dtype=torch.float32)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.set_timesteps(1000)  # Ensure scheduler is set to 1000 steps for consistent noise scaling

def _debug_tensor_stats(name: str, tensor: torch.Tensor):
    """Print tensor health/range stats to diagnose black outputs."""
    t = tensor.detach().float()
    total = t.numel()
    finite_mask = torch.isfinite(t)
    finite_count = finite_mask.sum().item()
    nan_count = torch.isnan(t).sum().item()
    inf_count = torch.isinf(t).sum().item()

    if finite_count > 0:
        finite_vals = t[finite_mask]
        t_min = finite_vals.min().item()
        t_max = finite_vals.max().item()
        t_mean = finite_vals.mean().item()
        t_std = finite_vals.std().item()
    else:
        t_min = float("nan")
        t_max = float("nan")
        t_mean = float("nan")
        t_std = float("nan")

    near_zero_ratio = (t.abs() < 1e-6).float().mean().item() if total > 0 else 0.0

    print(
        f"[DEBUG] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, min={t_min:.6f}, max={t_max:.6f}, "
        f"mean={t_mean:.6f}, std={t_std:.6f}, "
        f"finite={finite_count}/{total}, nan={nan_count}, inf={inf_count}, "
        f"near_zero_ratio={near_zero_ratio:.4f}"
    )


def _debug_image_stats(name: str, image_np: np.ndarray):
    """Print image stats after conversion to diagnose all-black outputs."""
    arr = image_np.astype(np.float32)
    zero_ratio = float((arr <= 0).mean())
    print(
        f"[DEBUG] {name}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}, "
        f"mean={arr.mean():.6f}, std={arr.std():.6f}, zero_ratio={zero_ratio:.4f}"
    )

import torch

def tensor_health(name, t):
    t = t.detach().float()
    finite = torch.isfinite(t)
    finite_count = finite.sum().item()
    total = t.numel()
    print(
        f"{name:24s} "
        f"dtype={t.dtype} "
        f"min={t[finite].min().item() if finite_count else float('nan'):.6f} "
        f"max={t[finite].max().item() if finite_count else float('nan'):.6f} "
        f"mean={t[finite].mean().item() if finite_count else float('nan'):.6f} "
        f"std={t[finite].std().item() if finite_count else float('nan'):.6f} "
        f"nan={torch.isnan(t).sum().item()} "
        f"inf={torch.isinf(t).sum().item()} "
        f"finite={finite_count}/{total}"
    )

def inspect_vae_weights(vae):
    print("=== VAE module info ===")
    print("vae dtype:", vae.dtype)
    print("force_upcast:", getattr(vae.config, "force_upcast", None))

    bad = []
    for name, p in vae.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)

    if bad:
        print("Non-finite VAE params found:", len(bad))
        for n in bad[:20]:
            print(" -", n)
    else:
        print("All VAE params are finite.")

def inspect_posterior(pipe, image_tensor):
    # image_tensor should be in [-1,1], shape [1,3,H,W]
    with torch.no_grad():
        posterior = pipe.vae.encode(image_tensor).latent_dist
        mu = posterior.mean
        logvar = posterior.logvar
        std = posterior.std
        z = posterior.sample()

    print("=== Posterior stats ===")
    tensor_health("posterior.mean", mu)
    tensor_health("posterior.logvar", logvar)
    tensor_health("posterior.std", std)
    tensor_health("posterior.sample_z", z)


def single_step_denoise_workflow(image_path: str, scale_factor: float, noise_timestep: int = 50, debug: bool = False):
    """
    Args:
        image_path: Path to the clean real image.
        scale_factor: 1.0, 0.5, or 0.25 (Level of degradation).
        noise_timestep: The timestep at which to add noise (0 to 1000).
                        500 implicitly acts as "50% noise".
        debug: If True, prints detailed tensor stats to trace black-image root causes.
    """
    # 1. Start from a clean real image
    original_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_image.size

    # 3. Control degradation level by downscaling
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)

    # Enforce divisibility by 64 (mandatory structurally for SDXL latents + U-Net)
    new_w = new_w - (new_w % 64)
    new_h = new_h - (new_h % 64)

    if new_w <= 0 or new_h <= 0:
        raise ValueError(
            f"Downscaled size became invalid ({new_w}x{new_h}). "
            "Use a larger input image or a larger scale_factor."
        )

    downscaled_image = original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Format image to Tensor [-1.0, 1.0] for the VAE
    image_np = np.array(downscaled_image).astype(np.float32) / 127.5 - 1.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(pipe.device, torch.float32)
    if debug:
        _debug_tensor_stats("image_tensor_pre_vae", image_tensor)

    # Convert to Latents Space (VAE running in FP32 to prevent FP16 overflow NaNs)
    with torch.no_grad():
        if debug:
            inspect_vae_weights(pipe.vae)
            inspect_posterior(pipe, image_tensor)
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()
        latents = (latents * pipe.vae.config.scaling_factor).to(torch.float16)
    if debug:
        _debug_tensor_stats("latents_clean", latents)

    # 2. Add noise to clean real latent
    noise = torch.randn_like(latents)
    timestep = torch.tensor([noise_timestep], device=pipe.device)

    noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)
    if debug:
        _debug_tensor_stats("noise", noise)
        _debug_tensor_stats("latents_noisy", noisy_latents)

    # Create textual conditioning (Empty/Null prompt is usually preferred for pure image recovery)
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt="",
        device=pipe.device,
        do_classifier_free_guidance=False
    )

    # SDXL specifically expects time_ids in `added_cond_kwargs` (original_size, crops, target_size)
    add_time_ids = torch.tensor(
        [[new_h, new_w, 0, 0, new_h, new_w]],
        dtype=pipe.dtype, device=pipe.device
    )
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

    # Execute a Single Step U-Net Forward Pass (Predicting Epsilon Noise)
    with torch.no_grad():
        noise_pred = pipe.unet(
            noisy_latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]
    if debug:
        _debug_tensor_stats("noise_pred", noise_pred)

    # 2b. Extract Denoised Estimation
    # With DDIM, stepping forward yields the exact formula calculation of x_0 (clean image estimation)
    with torch.no_grad():
        step_output = pipe.scheduler.step(noise_pred, timestep.item(), noisy_latents)
        denoised_latents = step_output.pred_original_sample
    if debug:
        _debug_tensor_stats("denoised_latents_x0", denoised_latents)

    # Decode Denoised Latent back to RGB Array (Upcast latents to FP32 for VAE decoder)
    with torch.no_grad():
        denoised_latents_scaled = (denoised_latents / pipe.vae.config.scaling_factor).to(torch.float32)
        denoised_tensor = pipe.vae.decode(denoised_latents_scaled).sample
    if debug:
        _debug_tensor_stats("denoised_tensor_decoded", denoised_tensor)

    # Map tensor [-1, 1] to PIL Image [0, 255]
    denoised_tensor = (denoised_tensor / 2 + 0.5).clamp(0, 1)
    if debug:
        _debug_tensor_stats("denoised_tensor_clamped_0_1", denoised_tensor)
    denoised_tensor = denoised_tensor.cpu().permute(0, 2, 3, 1).numpy()[0]
    if debug:
        _debug_image_stats("denoised_numpy_0_1", denoised_tensor)

    denoised_tensor = np.nan_to_num(denoised_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    denoised_uint8 = (denoised_tensor * 255).astype(np.uint8)
    if debug:
        _debug_image_stats("denoised_uint8_0_255", denoised_uint8)

    denoised_result = Image.fromarray(denoised_uint8)

    return downscaled_image, denoised_result

if __name__ == "__main__":
    for scale in [0.5, 0.25]:
        print(f"Processing scale: {scale}x")
        degraded, result = single_step_denoise_workflow(
             image_path="cleaner.jpg",
             scale_factor=scale,
               noise_timestep=1, # Customize noise limit here (0 to 1000)
               debug=False
        )

        result.resize((512, 512), Image.Resampling.LANCZOS).save(f"denoised_inference_{scale}x.jpg")
