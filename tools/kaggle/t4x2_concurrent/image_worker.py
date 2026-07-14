"""T4x2 concurrent test — SANA-Sprint image generation worker.

Pinned to cuda:0 by the parent driver via CUDA_VISIBLE_DEVICES=0.
Generates a small real batch of images and writes a per-worker manifest.
Reuses the proven SANA-Sprint singleton-loading + VAE float32 pattern.
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

import torch

DEVICE = "cuda"
OUT = Path("/kaggle/working/img_out")
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    ("img_01", "Rain-soaked Jeju street at 3 AM, taxi pulling away, monochrome noir film grain"),
    ("img_02", "Crime scene evidence board, red string connecting photos, typewriter labels, muted palette"),
    ("img_03", "Korean coastal cliff at dawn, fog, cold color grade, cinematic still"),
    ("img_04", "Police archive room, filing cabinets, manila folders, fluorescent lighting"),
    ("img_05", "Courtroom empty at night, wooden bench, shaft of light through window"),
    ("img_06", "Surveillance monitor wall, CCTV grain, blue tint, noir atmosphere"),
]


def load_pipe():
    from diffusers import SanaSprintPipeline
    pipe = SanaSprintPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipe.vae.to(torch.float32)
    return pipe


def main() -> int:
    t0 = time.time()
    pipe = load_pipe()
    load_s = round(time.time() - t0, 2)
    print(f"[IMG cuda:0] SANA loaded in {load_s}s", flush=True)

    results = []
    total_gen = 0.0
    for aid, prompt in PROMPTS:
        start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            img = pipe(
                prompt=prompt,
                height=576,
                width=1024,
                num_inference_steps=2,
                guidance_scale=4.5,
            ).images[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
        else:
            peak = 0
        dur = time.time() - start
        total_gen += dur
        p = OUT / f"{aid}.png"
        img.save(p)
        results.append({
            "id": aid, "path": str(p),
            "duration_s": round(dur, 2),
            "vram_peak_bytes": int(peak),
            "success": True,
        })
        print(f"  {aid}: {dur:.2f}s", flush=True)

    manifest = {
        "role": "image",
        "device": DEVICE,
        "model_load_s": load_s,
        "total_gen_s": round(total_gen, 2),
        "scenes": results,
    }
    (OUT / "worker_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[IMG cuda:0] DONE load={load_s}s gen={round(total_gen,2)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
