"""
Benchmark of 5 candidate image models vs SANA-Sprint 1.6B baseline (Kaggle T4).

Isolated benchmark for OpenMontage. Does NOT touch production pipeline.
Protocol: 1024x1024, batch=1, official fast settings per model, fixed seeds,
10 documentary prompts, 1 untimed warm-up image per model, peak-VRAM via
torch.cuda.max_memory_allocated after torch.cuda.synchronize().

Outputs: /kaggle/working/output/<model_key>/*.png + /kaggle/working/results.json
Statuses: OK | BLOCKED (gated / checkpoint not found) | FAILED (dependency/oom/error)
No model substitution: a blocked checkpoint is recorded as blocked, never swapped.
"""

import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

OUTPUT_ROOT = "/kaggle/working/output"
RESULTS_PATH = "/kaggle/working/results.json"
SEED_BASE = 4200
MIN_BYTES, MIN_STD = 50_000, 5.0

PROMPTS = [
    "Cold War intelligence room with analysts, maps, telephones and equipment.",
    "Dense 1980s military control panel with dozens of switches, gauges and CRT displays.",
    "Scientist operating complex laboratory control panel, realistic hands.",
    "Soviet submarine control room with gauges, pipes, switches and cables.",
    "NASA mission control with many consoles and engineers.",
    "Vintage radar console with many small controls and labels.",
    "Electronics laboratory with oscilloscopes, CRTs, circuit boards and engineers.",
    "Intelligence briefing room with maps, photographs and documents.",
    "Industrial workshop with engine, pipes, valves, tools and workers.",
    "Complex technical laboratory with many interacting objects.",
]
WARMUP_PROMPT = "A ceramic cup on a wooden table, soft window light."

RESULTS = {"benchmark": "image_models_2026_08", "environment": {}, "protocol": {}, "models": []}


def log(m):
    print(f"[bench] {m}", flush=True)


def save_results():
    with open(RESULTS_PATH, "w") as fh:
        json.dump(RESULTS, fh, indent=2)


def sh(pip_args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pip_args, check=True)


def classify_error(exc):
    name = type(exc).__name__
    msg = f"{name}: {exc}"
    if "GatedRepoError" in name or "401" in str(exc) or "403" in str(exc) or "Access denied" in str(exc):
        return "BLOCKED", "gated_or_auth", msg
    if "EntryNotFound" in name or "404" in str(exc) or "does not appear to exist" in str(exc):
        return "BLOCKED", "checkpoint_not_found", msg
    if "OutOfMemoryError" in name or "CUDA out of memory" in str(exc):
        return "FAILED", "oom", msg
    if name in ("ImportError", "ModuleNotFoundError") or "undefined symbol" in str(exc).lower():
        return "FAILED", "dependency", msg
    return "FAILED", "error", msg


def env_report():
    import diffusers
    import transformers
    info = {
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1048576, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    RESULTS["environment"] = info
    log(json.dumps(info))


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_generation(model_entry, pipe, call_kwargs_fn, img_dir):
    os.makedirs(img_dir, exist_ok=True)

    def gen_one(idx, prompt, fname):
        g = torch.Generator("cuda").manual_seed(SEED_BASE + idx)
        kw = call_kwargs_fn()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        img = pipe(prompt=prompt, generator=g, **kw).images[0]
        torch.cuda.synchronize()
        el = time.time() - t0
        path = os.path.join(img_dir, fname)
        img.save(path)
        size = os.path.getsize(path)
        std = float(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).std())
        rec = {
            "scene_index": idx, "prompt": prompt, "seed": SEED_BASE + idx,
            "file": path.replace("/kaggle/working/", ""), "elapsed_seconds": round(el, 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1),
            "bytes": size, "pixel_std": round(std, 2),
            "gate_passed": bool(size >= MIN_BYTES and std >= MIN_STD),
        }
        log(json.dumps({k: rec[k] for k in ("scene_index", "elapsed_seconds", "peak_vram_mb", "gate_passed")}))
        return rec

    warm = gen_one(-1, WARMUP_PROMPT, "warmup.png")
    items, errors = [], []
    for i, prompt in enumerate(PROMPTS):
        try:
            items.append(gen_one(i, prompt, f"scene{i:02d}.png"))
        except Exception as e:
            status, kind, msg = classify_error(e)
            errors.append({"scene_index": i, "kind": kind, "message": msg[:2000],
                           "traceback_tail": traceback.format_exc()[-1500:]})
            log(f"scene {i} {kind}: {msg[:300]}")
            if kind == "oom":
                break
        save_results()
    secs = [r["elapsed_seconds"] for r in items]
    model_entry.update({
        "warmup_seconds": warm["elapsed_seconds"],
        "images": items, "errors": errors,
        "successful_images": len(items), "failed_images": len(errors),
        "avg_sec_per_image": round(statistics.mean(secs), 2) if secs else None,
        "median_sec_per_image": round(statistics.median(secs), 2) if secs else None,
        "peak_vram_mb": max((r["peak_vram_mb"] for r in items), default=None),
    })


def finalize(entry, load_t0):
    entry["load_time_seconds"] = round(time.time() - load_t0, 1)
    RESULTS["models"].append(entry)
    save_results()


def bench_sana_baseline():
    from diffusers import SanaSprintPipeline
    repo = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers"
    entry = {"key": "sana_sprint_baseline", "repo_id": repo,
             "settings": {"steps": 2, "height": 1024, "width": 1024, "batch": 1,
                          "note": "production known-good setting; vae fp32 fix inherited"},
             "role": "baseline"}
    t0 = time.time()
    try:
        pipe = SanaSprintPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
        pipe.vae.to(torch.float32)
    except Exception as e:
        status, kind, msg = classify_error(e)
        finalize({**entry, "status": status, "blocked_reason": kind, "error": msg}, t0)
        return
    try:
        def kw():
            return {"num_inference_steps": 2, "height": 1024, "width": 1024}
        run_generation(entry, pipe, kw, os.path.join(OUTPUT_ROOT, "sana_sprint_baseline"))
    finally:
        del pipe
        free_gpu()
    finalize(entry, t0)


def bench_sd35_flash():
    repo = "stabilityai/sd3.5-flash"
    entry = {"key": "sd35_flash", "repo_id": repo,
             "settings": {"steps": 4, "guidance_scale": 1.0, "height": 1024, "width": 1024, "batch": 1,
                          "note": "official fast settings from paper (arXiv 2509.21318); no public HF weights found"},
             "role": "candidate"}
    t0 = time.time()
    pipe = None
    try:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
        entry["loader"] = "StableDiffusion3Pipeline"
    except Exception as e:
        status, kind, msg = classify_error(e)
        finalize({**entry, "status": status, "blocked_reason": kind,
                  "error": msg, "traceback_tail": traceback.format_exc()[-1500:]}, t0)
        return
    try:
        def kw():
            return {"num_inference_steps": 4, "guidance_scale": 1.0,
                    "height": 1024, "width": 1024}
        run_generation(entry, pipe, kw, os.path.join(OUTPUT_ROOT, "sd35_flash"))
    finally:
        del pipe
        free_gpu()
    finalize(entry, t0)


def bench_microsoft_lens_turbo():
    repo = "microsoft/Lens-Turbo"
    entry = {"key": "microsoft_lens_turbo", "repo_id": repo,
             "settings": {"steps": 4, "guidance_scale": 1.0, "height": 1024, "width": 1024, "batch": 1,
                          "note": "official fast settings from model card; gated repo"},
             "role": "candidate"}
    t0 = time.time()
    pipe = None
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
        entry["loader"] = "DiffusionPipeline"
    except Exception as e:
        status, kind, msg = classify_error(e)
        finalize({**entry, "status": status, "blocked_reason": kind,
                  "error": msg, "traceback_tail": traceback.format_exc()[-1500:]}, t0)
        return
    try:
        def kw():
            return {"num_inference_steps": 4, "guidance_scale": 1.0,
                    "height": 1024, "width": 1024}
        run_generation(entry, pipe, kw, os.path.join(OUTPUT_ROOT, "microsoft_lens_turbo"))
    finally:
        del pipe
        free_gpu()
    finalize(entry, t0)


def bench_amd_nitro_t():
    from transformers import AutoModelForCausalLM
    repo = "amd/Nitro-T-1.2B"
    text_encoder_repo = "meta-llama/Llama-3.2-1B"
    entry = {"key": "amd_nitro_t_1_2b", "repo_id": repo,
             "text_encoder_repo_id": text_encoder_repo,
             "settings": {"height": 1024, "width": 1024, "batch": 1,
                          "note": "official settings from model card; requires gated Llama-3.2-1B text encoder"},
             "role": "candidate"}
    t0 = time.time()
    try:
        text_encoder = AutoModelForCausalLM.from_pretrained(text_encoder_repo, torch_dtype=torch.bfloat16)
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            repo, text_encoder=text_encoder, torch_dtype=torch.bfloat16,
            trust_remote_code=True).to("cuda")
        entry["loader"] = "DiffusionPipeline trust_remote_code per model card"
    except Exception as e:
        status, kind, msg = classify_error(e)
        finalize({**entry, "status": status, "blocked_reason": kind,
                  "error": msg, "traceback_tail": traceback.format_exc()[-1500:],
                  "substitution_used": False}, t0)
        return
    try:
        def kw():
            return {"num_inference_steps": 20, "height": 1024, "width": 1024}
        run_generation(entry, pipe, kw, os.path.join(OUTPUT_ROOT, "amd_nitro_t_1_2b"))
    finally:
        del pipe
        free_gpu()
    finalize(entry, t0)


def bench_bonsai_gemlite():
    repo = "prism-ml/bonsai-image-ternary-4B-gemlite-2bit"
    entry = {"key": "bonsai_ternary_4b_gemlite", "repo_id": repo,
             "settings": {"steps": 4, "guidance_scale": 1.0, "shift": 3.0,
                          "height": 1024, "width": 1024, "batch": 1,
                          "note": "official fast settings from model card"}}
    t0 = time.time()
    pipe = None
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
        entry["loader"] = "DiffusionPipeline.from_pretrained per model card"
    except Exception as e:
        status, kind, msg = classify_error(e)
        finalize({**entry, "status": status, "blocked_reason": kind,
                  "error": msg, "traceback_tail": traceback.format_exc()[-1500:]}, t0)
        return
    try:
        def kw():
            return {"num_inference_steps": 4, "guidance_scale": 1.0,
                    "height": 1024, "width": 1024}
        run_generation(entry, pipe, kw, os.path.join(OUTPUT_ROOT, "bonsai_ternary_4b_gemlite"))
    finally:
        del pipe
        free_gpu()
    finalize(entry, t0)


def bench_sid_dit_sd35():
    from huggingface_hub import snapshot_download
    repo = "YGu1998/SiD-DiT-SD3.5-medium"
    entry = {"key": "sid_dit_sd35_medium", "repo_id": repo,
             "settings": {"steps": 4, "guidance_scale": 1.0, "time_scale": 1000,
                          "height": 1024, "width": 1024, "batch": 1,
                          "note": "official settings from model card README"}}
    t0 = time.time()
    snap = snapshot_download("YGu1998/SiD_pipelines")
    pkg_root = "/kaggle/working/SiD_pipelines_pkg"
    pkg_dir = os.path.join(pkg_root, "SiD_pipelines")
    os.makedirs(pkg_dir, exist_ok=True)
    for f in os.listdir(snap):
        if f.endswith(".py") or f == "requirements.txt":
            shutil.copy2(os.path.join(snap, f), os.path.join(pkg_dir, f))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from SiD_pipelines import SiDSD3Pipeline
    pipe = None
    offload = False
    try:
        try:
            pipe = SiDSD3Pipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
        except torch.OutOfMemoryError:
            log("cuda OOM on full .to(cuda); retrying once with enable_model_cpu_offload")
            free_gpu()
            pipe = None
            pipe = SiDSD3Pipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
            offload = True
        entry["cpu_offload"] = offload
    except Exception as e:
        status, kind, msg = classify_error(e)
        finalize({**entry, "status": status, "blocked_reason": kind,
                  "error": msg, "traceback_tail": traceback.format_exc()[-1500:]}, t0)
        return
    try:
        def kw():
            return {"num_inference_steps": 4, "guidance_scale": 1.0, "time_scale": 1000,
                    "height": 1024, "width": 1024}
        run_generation(entry, pipe, kw, os.path.join(OUTPUT_ROOT, "sid_dit_sd35_medium"))
    finally:
        del pipe
        free_gpu()
    finalize(entry, t0)


def main():
    log("installing dependencies into isolated target dir")
    VENV_DIR = "/kaggle/working/venv-libs"
    os.makedirs(VENV_DIR, exist_ok=True)
    sh(["-q", "--target", VENV_DIR,
        "transformers==4.57.1", "diffusers>=0.38.0",
        "accelerate", "sentencepiece", "hqq", "gemlite"])
    sys.path.insert(0, VENV_DIR)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    env_report()
    RESULTS["protocol"] = {
        "resolution": "1024x1024", "batch": 1, "seed_base": SEED_BASE,
        "seeds": [SEED_BASE + i for i in range(len(PROMPTS))],
        "scenes": len(PROMPTS), "warmup_images": 1,
        "quality_gate": {"min_bytes": MIN_BYTES, "min_pixel_std": MIN_STD},
        "prompts": PROMPTS,
    }
    save_results()

    order = [
        ("sd35_flash", bench_sd35_flash),
        ("microsoft_lens_turbo", bench_microsoft_lens_turbo),
        ("amd_nitro_t_1_2b", bench_amd_nitro_t),
        ("bonsai_ternary_4b_gemlite", bench_bonsai_gemlite),
        ("sid_dit_sd35_medium", bench_sid_dit_sd35),
        ("sana_sprint_baseline", bench_sana_baseline),
    ]
    for key, fn in order:
        log(f"=== {key} ===")
        try:
            fn()
        except Exception as e:
            RESULTS["models"].append({
                "key": key, "status": "FAILED", "blocked_reason": "harness_error",
                "error": f"{type(e).__name__}: {e}",
                "traceback_tail": traceback.format_exc()[-1500:]})
            log(f"harness error in {key}: {e}")
        save_results()

    summary = [{"key": m.get("key"), "status": m.get("status"),
                "avg_sec": m.get("avg_sec_per_image"), "median_sec": m.get("median_sec_per_image"),
                "peak_vram_mb": m.get("peak_vram_mb")} for m in RESULTS["models"]]
    log("SUMMARY " + json.dumps(summary))
    save_results()


if __name__ == "__main__":
    main()
