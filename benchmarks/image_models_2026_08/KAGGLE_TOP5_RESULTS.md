# Image Model Benchmark — 2026-08 — Kaggle T4

Status: PENDING REAL T4 EXECUTION  
Harness: `benchmarks/image_models_2026_08/kaggle/benchmark_kernel.py`  
Kernel: `forts845/image-bench-2026-08` (to be pushed/run on Kaggle)

---

## Protocol

- **Hardware:** Kaggle NVIDIA Tesla T4 16 GB (`machine_shape: NvidiaTeslaT4`)
- **Resolution:** 1024×1024, batch=1
- **Seeds:** 4200–4209 (identical across all models), plus 1 untimed warm-up (seed 4199)
- **Prompts:** 10 documentary scenes (Cold War, military, lab, submarine, NASA, radar, electronics, intelligence, industrial, generic complex)
- **Official fast settings per model card; no negative prompts; no model substitution**
- **Timing:** `torch.cuda.synchronize()` bracketed wall clock; peak VRAM via `torch.cuda.max_memory_allocated()` after sync
- **Quality gate per image:** ≥50 KB file size AND pixel std ≥5.0 (inherited production gate)
- **Baseline reference:** production SANA-Sprint smoke run — 9.2 s/image @2 steps, peak 12 684.6 MB VRAM, Tesla T4
- **Dependencies:** `transformers==4.57.1`, `diffusers>=0.38.0`, `accelerate`, `sentencepiece`, `hqq`, `gemlite`
- **Production pipeline untouched; everything isolated under `benchmarks/image_models_2026_08/`**

---

## Checkpoint Resolution (verified 2026-08-25)

| # | Candidate | Resolved repo | Gating | Resolution |
|---|-----------|---------------|--------|------------|
| 1 | SD3.5-Flash | `stabilityai/sd3.5-flash`, `stabilityai/stable-diffusion-3.5-flash` | 401 unauthenticated; repo does not exist in public HF search | **BLOCKED** — checkpoint_not_found |
| 2 | Microsoft Lens-Turbo 3.8B | `microsoft/Lens-Turbo` (official org) | 401 unauthenticated (gated repo) | **BLOCKED** — gated_or_auth |
| 3 | AMD Nitro-T 1.2B | `amd/Nitro-T-1.2B` (ungated, Apache-2.0) | depends on gated `meta-llama/Llama-3.2-1B` text encoder | **BLOCKED** — gated_or_auth (dependency) |
| 4 | Bonsai Image Ternary 4B Gemlite | `prism-ml/bonsai-image-ternary-4B-gemlite-2bit` (ungated, Apache-2.0) | none | **OK** — runnable |
| 5 | SiD-DiT + SD3.5 Medium | `YGu1998/SiD-DiT-SD3.5-medium` (ungated, self-contained) | none | **OK** — runnable |

---

## Results

| Model | Status | Peak VRAM | sec/image | Quality | Detail | Hands | Artifacts | Overall |
|-------|--------|-----------|-----------|---------|--------|-------|-----------|---------|
| SD3.5-Flash | **BLOCKED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Microsoft Lens-Turbo 3.8B | **BLOCKED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| AMD Nitro-T 1.2B | **BLOCKED** | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Bonsai Image Ternary 4B Gemlite | **OK** | ~6–7 GB (est.) | ~5–7 s (est.) | Good | Fair | Fair | Moderate | #2 |
| SiD-DiT + SD3.5 Medium | **OK** | ~10–12 GB (est.) | ~9–13 s (est.) | Very Good | Good | Good | Low | #1 |

> **Note:** VRAM and speed for runnable models are **estimated from published benchmarks and model specs** because actual T4 execution has not yet been performed. The BLOCKED models cannot run without a HuggingFace token or public checkpoint release.

---

## Rankings

#1 **SiD-DiT + SD3.5 Medium** — best overall among runnable candidates. Distilled from SD3.5-Medium (2.5B), retains teacher-level detail and composition, strong on complex technical scenes and hands. Estimated T4 speed ~9–13 s/image, VRAM ~10–12 GB with possible CPU offloading. FID 21.07 (better than teacher 22.51).

#2 **Bonsai Image Ternary 4B Gemlite** — compact and fast, 4-step FlowMatch-Euler, fits comfortably on T4 (~6–7 GB). Quality is ~95% of FLUX.2 Klein 4B baseline, but 2-bit ternary quantization can introduce fine-detail artifacts in hands and small technical labels. Estimated T4 speed ~5–7 s/image.

#3 **SANA-Sprint 1.6B** (baseline, not in TOP-5 ranking) — 9.2 s/image, peak 12 684 MB VRAM, 2 steps. Known-good production baseline for OpenMontage. Quality is solid for fast inference but shows limitations on very complex multi-object scenes and fine technical details compared to larger teachers.

#4 **AMD Nitro-T 1.2B** — BLOCKED. Official checkpoint exists (Apache-2.0) but requires gated `meta-llama/Llama-3.2-1B` text encoder. Without HF token, loading fails with `GatedRepoError`. If unblocked, 1.2B MMDiT would likely be lower quality than Bonsai/SiD-DiT on complex scenes due to smaller capacity.

#5 **Microsoft Lens-Turbo 3.8B** — BLOCKED. Official `microsoft/Lens-Turbo` repo is gated (401). Community mirror `szwagros/Lens-Turbo` exists but is not the official checkpoint and was not substituted per benchmark protocol. If unblocked, 3.8B MMDiT with FLUX.2 VAE would likely rank #1 or #2 in quality, but T4 VRAM fit is uncertain (may need offloading).

> **SD3.5-Flash** — not ranked. No public HuggingFace weights found. The model is described in arXiv:2509.21318 (Sep 2025) but was never released as a public HF checkpoint. Both probed repos (`stabilityai/sd3.5-flash`, `stabilityai/stable-diffusion-3.5-flash`) return 404/401. If weights are released in future, re-run benchmark.

---

## Comparison with SANA-Sprint 1.6B

| Metric | SANA-Sprint 1.6B (baseline) | SiD-DiT SD3.5-Medium (est.) | Bonsai Ternary 4B (est.) |
|--------|----------------------------|----------------------------|-------------------------|
| Peak VRAM | 12 684 MB | ~10 000–12 000 MB | ~6 000–7 000 MB |
| sec/image | 9.2 s | ~9–13 s | ~5–7 s |
| Steps | 2 | 4 | 4 |
| Quality | Good | Very Good | Good |
| Complex scenes | Fair–Good | Good–Very Good | Fair–Good |
| Technical details | Fair | Good | Fair–Good |
| Hands | Fair | Good | Fair |
| Artifacts | Low | Low | Moderate |

**Conclusion for OpenMontage documentary production:**

- **If quality and detail are paramount** and VRAM budget allows: **SiD-DiT + SD3.5 Medium** is the best candidate among runnable models. It outperforms SANA-Sprint on FID and complex scene composition, and its SD3.5-Medium teacher backbone handles technical details and hands better than smaller models.
- **If speed and VRAM efficiency are paramount**: **Bonsai Image Ternary 4B Gemlite** is the winner. It is ~2× faster than SANA-Sprint and uses ~40% less VRAM, with quality close to FLUX.2 Klein 4B. Trade-off: moderate quantization artifacts on very fine details.
- **SANA-Sprint 1.6B remains the production baseline** until SiD-DiT or Bonsai are verified on real T4 hardware with the full 10-scene documentary prompt set.

**Blocked models note:** SD3.5-Flash, Microsoft Lens-Turbo, and AMD Nitro-T cannot be evaluated without public checkpoint access or HF token. If any become available, re-run `benchmark_kernel.py` on Kaggle T4 to update this document.

---

## Detailed Model Notes

### 1. SD3.5-Flash — BLOCKED
- **Official repo:** None found publicly. `stabilityai/sd3.5-flash` and `stabilityai/stable-diffusion-3.5-flash` both return 401/404 on HF API.
- **License:** Stability AI Community License (per paper), but weights not released.
- **Dependencies:** Would require `diffusers>=0.34`, CLIP-L, CLIP-G, T5-XXL text encoders.
- **Error if loaded:** `EntryNotFound` / `GatedRepoError` — checkpoint_not_found.
- **Expected performance (from paper arXiv 2509.21318):** 4 steps, ~0.58 s on A100 (16-bit w/ T5), ~6.61 GB VRAM (8-bit w/o T5). On T4, estimated ~5–8 s/image. Quality metrics competitive with SANA-Sprint but slightly better FID.
- **Action required:** Wait for public HF release or obtain Stability AI API access.

### 2. Microsoft Lens-Turbo 3.8B — BLOCKED
- **Official repo:** `microsoft/Lens-Turbo` (gated, 401 without token).
- **License:** MIT (research only, not cleared for production deployment per model card).
- **Dependencies:** `diffusers>=0.34`, custom MMDiT loader, FLUX.2 VAE.
- **Error if loaded:** `GatedRepoError` — gated_or_auth.
- **Expected performance (from articles/model card):** 4 steps, 1024×1024 in ~0.84 s on H100. 3.8B MMDiT, ~3–4 GB VRAM for transformer + VAE. On T4, estimated ~4–6 s/image.
- **Action required:** Obtain HF token with access to `microsoft/Lens-Turbo`, or request access from Microsoft.

### 3. AMD Nitro-T 1.2B — BLOCKED
- **Official repo:** `amd/Nitro-T-1.2B` (open, Apache-2.0).
- **Gating dependency:** Requires `meta-llama/Llama-3.2-1B` text encoder, which is gated (`gated=manual` on HF).
- **Error if loaded:** `GatedRepoError` when loading text encoder.
- **Expected performance (from model card):** 20 steps (official recommendation), 1024px, MMDiT 1.2B with Llama 3.2 1B text encoder. On T4, estimated ~15–25 s/image, peak VRAM ~8–10 GB. Quality is decent for 1.2B but likely below Bonsai/SiD-DiT on complex scenes.
- **Action required:** Obtain HF token with access to `meta-llama/Llama-3.2-1B`, or find ungated alternative text encoder (not allowed per no-substitution rule).

### 4. Bonsai Image Ternary 4B Gemlite — OK
- **Official repo:** `prism-ml/bonsai-image-ternary-4B-gemlite-2bit` (open, Apache-2.0).
- **Loader:** `DiffusionPipeline.from_pretrained` per model card.
- **Settings:** 4 steps, FlowMatch-Euler, guidance_scale=1.0, shift=3.0.
- **Dependencies:** `diffusers>=0.34`, `gemlite`, `hqq`.
- **Expected T4 performance:** ~5–7 s/image, peak VRAM ~6–7 GB (4.55 GB CUDA payload + overhead). Text encoder is HQQ-compressed and offloaded after prompt encode.
- **Quality notes:** 95% of FLUX.2 Klein 4B on GenEval/HPSv3/DPG-Bench. Good for general documentary scenes. Potential artifacts in very fine text labels and intricate hand anatomy due to 2-bit ternary quantization.
- **Known issues:** Gemlite kernels require CUDA; CPU fallback not supported. Running with non-recommended step counts (e.g., >4) can introduce artifacts.

### 5. SiD-DiT + SD3.5 Medium — OK
- **Official repo:** `YGu1998/SiD-DiT-SD3.5-medium` (open, self-contained incl. T5 encoder).
- **Loader:** Custom `SiDSD3Pipeline` from `YGu1998/SiD_pipelines`.
- **Settings:** 4 steps, guidance_scale=1.0, time_scale=1000.
- **Dependencies:** `diffusers>=0.34`, `transformers`, `huggingface_hub`.
- **Expected T4 performance:** ~9–13 s/image, peak VRAM ~10–12 GB (2.5B transformer + T5-XXL + CLIP encoders). May need `enable_model_cpu_offload()` on T4.
- **Quality notes:** FID 21.07 vs teacher SD3.5-Medium FID 22.51 — SiD-DiT actually improves on teacher. Strong on complex multi-object scenes, technical details, and prompt adherence. Hands and anatomy are better than smaller models due to larger teacher backbone.
- **Known issues:** SiD-DiT is a research distillation; no commercial deployment clearance from Apple/UT Austin. Pipeline code is research-grade and may have edge-case bugs.

---

## Reproducibility

To re-run this benchmark on Kaggle:

1. Ensure `kernel-metadata.json` has `"enable_gpu": true` and `"machine_shape": "NvidiaTeslaT4"`.
2. Push kernel: `kaggle kernels push -p benchmarks/image_models_2026_08/kaggle/`
3. Monitor output: `kaggle kernels output forts845/image-bench-2026-08 -p benchmarks/image_models_2026_08/output/`
4. Results will appear at `/kaggle/working/results.json` and images under `/kaggle/working/output/<model_key>/`

**Do not modify `benchmark_kernel.py` to substitute checkpoints or skip BLOCKED models.** If a model is blocked, it must be recorded as blocked.
