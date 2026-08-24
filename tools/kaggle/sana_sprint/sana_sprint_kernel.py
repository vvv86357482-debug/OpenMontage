"""
Sana-Sprint 1.6B image-generation kernel (Kaggle T4).

Known-good constraints — do NOT "fix" any of these, each one closed a real bug:
  - machine_shape: NvidiaTeslaT4 in kernel-metadata.json ("gpu_type" /
    "hardware_tier" are not real Kaggle API fields; they are silently ignored
    and the job defaults to P100).
  - num_inference_steps=2 exactly. This pipeline class raises ValueError on 4.
  - Transformer stays bfloat16; immediately after load run
    pipe.vae.to(torch.float32). A float16 VAE on this model produces
    all-black images.
  - torch.cuda.synchronize() before reading torch.cuda.max_memory_allocated().
  - augment_prompt_for_sana(): this pipeline has no negative_prompt argument;
    negatives are folded into "NOT X, use Y instead" clauses.
  - Hard-fail gate: reject output under 50KB or with pixel std < 5.0
    (catches black/placeholder images even when the job reports success).
Outputs go to /kaggle/working/output (kernel output), never /tmp.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

OUTPUT_DIR = "/kaggle/working/output"
MODEL_ID = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
NUM_INFERENCE_STEPS = 2
MIN_BYTES = 50_000
MIN_STD = 5.0


def log(msg):
    print(f"[sana_sprint] {msg}", flush=True)


def augment_prompt_for_sana(prompt, negative_prompt="", replacement=""):
    """SanaSprintPipeline accepts no negative_prompt parameter.

    Fold each comma-separated negative into a 'NOT X' clause, upgraded to
    'NOT X, use Y instead' when a replacement is supplied.
    """
    clauses = []
    for neg in [n.strip() for n in negative_prompt.split(",") if n.strip()]:
        if replacement.strip():
            clauses.append(f"NOT {neg}, use {replacement.strip()} instead")
        else:
            clauses.append(f"NOT {neg}")
    return prompt if not clauses else f"{prompt}. {', '.join(clauses)}."


def validate_output(path):
    """Hard-fail gate: catches black/placeholder images regardless of exit status."""
    size_bytes = os.path.getsize(path)
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    pixel_std = float(arr.std())
    passed = size_bytes >= MIN_BYTES and pixel_std >= MIN_STD
    return {
        "path": path,
        "size_bytes": size_bytes,
        "pixel_std": round(pixel_std, 2),
        "min_bytes_required": MIN_BYTES,
        "min_std_required": MIN_STD,
        "gate_passed": passed,
    }


def main():
    from diffusers import SanaSprintPipeline

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = {
        "model_id": MODEL_ID,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "transformer_dtype": "bfloat16",
        "vae_dtype": "float32",
        "seed": 0,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    log(f"GPU: {manifest['gpu_name']}")

    pipe = SanaSprintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.vae.to(torch.float32)  # fp16 VAE renders all-black images on this model
    manifest["vae_dtype_actual"] = str(next(pipe.vae.parameters()).dtype)
    manifest["transformer_dtype_actual"] = str(next(pipe.transformer.parameters()).dtype)

    scene = {
        "id": "smoke_001",
        "prompt": (
            "A weathered brass automaton head resting on archive shelves among "
            "dusty ledgers, candlelight, shallow depth of field, cinematic"
        ),
        "negative_prompt": "blurry, low quality, watermark, text",
        "replacement": "sharp focus, high detail",
    }
    final_prompt = augment_prompt_for_sana(
        scene["prompt"], scene["negative_prompt"], scene["replacement"]
    )
    log(f"augmented prompt: {final_prompt}")

    generator = torch.Generator("cuda").manual_seed(manifest["seed"])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    result = pipe(
        prompt=final_prompt,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=1024,
        width=1024,
        generator=generator,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    image_path = os.path.join(OUTPUT_DIR, f"{scene['id']}.png")
    result.images[0].save(image_path)
    check = validate_output(image_path)

    manifest.update(
        {
            "scene_id": scene["id"],
            "final_prompt": final_prompt,
            "elapsed_seconds": round(elapsed, 2),
            "peak_vram_mb": round(peak_mem_mb, 1),
            **check,
        }
    )
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"manifest: {json.dumps(manifest)}")

    if not check["gate_passed"]:
        log(
            f"HARD FAIL: size={check['size_bytes']}B std={check['pixel_std']} "
            f"(needs >={MIN_BYTES}B and std>={MIN_STD})"
        )
        sys.exit(2)
    log("GATE PASSED")


if __name__ == "__main__":
    main()
