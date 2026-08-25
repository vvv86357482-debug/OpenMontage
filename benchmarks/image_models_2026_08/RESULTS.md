# Image Model Benchmark — 2026-08 — Kaggle T4

Status: RUNNING (kernel `forts845/image-bench-2026-08`)

## Protocol

- Hardware: Kaggle NVIDIA Tesla T4 16 GB (`machine_shape: NvidiaTeslaT4`)
- 1024×1024, batch=1, identical prompts and seeds (4200–4209) across all models
- 10 documentary scenes + 1 untimed warm-up per model
- Official fast settings per model card; no negative prompts; no model substitution
- Timing: `torch.cuda.synchronize()` bracketed wall clock; peak VRAM via
  `torch.cuda.max_memory_allocated()` after sync (same method as production SANA kernel)
- Quality gate per image: ≥50 KB file size AND pixel std ≥5.0 (inherited production gate)
- Baseline reference: production SANA-Sprint smoke run
  (`tools/kaggle/outputs/sana_sprint/output/manifest.json`: 9.2 s/image @2 steps,
  peak 12 684.6 MB VRAM, Tesla T4, torch 2.10.0+cu128)
- Harness: `benchmarks/image_models_2026_08/kaggle/benchmark_kernel.py`
- Production pipeline untouched; everything isolated under `benchmarks/image_models_2026_08/`

## Checkpoint resolution (verified against HF API, 2026-08-24)

| # | Candidate | Resolved repo | Gating | Resolution |
|---|-----------|---------------|--------|------------|
| 1 | SD3.5-Flash | `stabilityai/sd3.5-flash`, `stabilityai/stable-diffusion-3.5-flash` | 401 unauthenticated; no public search listing found | BLOCKED unless token |
| 2 | Microsoft Lens-Turbo 3.8B | `microsoft/Lens-Turbo` (referenced by quantizer READMEs) | 401 unauthenticated | BLOCKED unless token |
| 3 | AMD Nitro-T 1.2B | `amd/Nitro-T-1.2B` (official org, diffusers, ungated) | repo open; requires gated `meta-llama/Llama-3.2-1B` text encoder | attempt in kernel; expected BLOCKED on gating without HF_TOKEN |
| 4 | Bonsai Image Ternary 4B Gemlite | `prism-ml/bonsai-image-ternary-4B-gemlite-2bit` (ungated, Apache-2.0) | none | runnable; Flux2KleinPipeline, FlowMatchEuler 4 steps, cfg 1.0, shift 3.0, gemlite+hqq kernels |
| 5 | SiD-DiT + SD3.5 Medium | `YGu1998/SiD-DiT-SD3.5-medium` (ungated, self-contained incl. T5 encoder) | none | runnable; custom `SiDSD3Pipeline`, 4 steps, cfg 1.0, time_scale 1000 |

## Results

To be filled from kernel `results.json` after run completes.
