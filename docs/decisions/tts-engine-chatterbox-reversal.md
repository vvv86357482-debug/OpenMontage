# Decision: TTS engine reversed — Chatterbox adopted as sole voice engine

- **Decision ID:** d-tts-001 (reversal of prior "Kokoro sole" decision)
- **Status:** REVERSED — Chatterbox is now the registered/primary TTS engine
- **Date reversed:** 2026-07-11
- **Category:** `voice_selection` (subject: "Narration TTS engine")
- **Machine-readable record:** `docs/decisions/decision_log.json`

## Old decision (quoted verbatim — DO NOT silently overwrite)

Master context previously stated, and the platform architecture encoded:

> "Kokoro remains the sole TTS engine." — Chatterbox was rejected as unviable on
> local CPU after it failed to run at acceptable speed without a GPU.

This was reflected in `AGENTS.md`:

> "Free stack only: Kokoro local TTS, Kaggle SANA-Sprint/FLUX, Pexels/Pixabay stock, ffmpeg."

Kokoro was the registered TTS tool with `runtime = LOCAL` (CPU-only).

## New decision

Chatterbox is adopted as the **sole voice engine** for all three channels
(crime-ledger, mythology-slavic, speculative-biology). Kokoro is retained as a
**documented manual fallback only** (not deleted, not the default) for when
Kaggle GPU quota is exhausted mid-week or HF Hub rate-limits Chatterbox's model
download.

Chatterbox runs with `runtime = KAGGLE_GPU` (Tesla T4), singleton-loaded once per
session — the same execution model as SANA-Sprint/FLUX image generation.

## Why it changed (evidence gathered 2026-07-11)

1. **Kaggle T4 is real-time capable.** A full 270-scene measurement pass on
   Tesla T4 (kernel `forts845/openmontage-chatterbox-270-scene-tts-measurement`)
   recorded RTF per channel:
   - crime-ledger: **1.079**
   - mythology-slavic: **1.088**
   - speculative-biology: **1.089**
   All near-real-time (claimed window 0.96–1.14 confirmed). The earlier
   "unviable on local CPU" verdict no longer applies once execution moves to
   the Kaggle T4 runtime, which is already in the pipeline for image generation.

2. **By-ear quality preference.** A direct 10-scene side-by-side (same narration
   lines, Chatterbox `default` voice vs Kokoro `am_puck`) was preferred by ear
   for Chatterbox. That comparison kernel is `tools/kaggle/tts_compare/`.

3. **Chatterbox is consistently shorter than Kokoro on the same text**, shrinking
   total runtime. Real re-measured totals (all 270 scenes, 100% generated, no
   extrapolation):
   - jeju-cold-case: **530.40s** (was 677.49s Kokoro)
   - leshy-forest-lord: **472.04s** (was ~490.5s Kokoro)
   - bombardier-beetle-proof: **453.92s** (was ~534.6s Kokoro)
   - **Grand total: 1456.36s** vs ~1702.6s Kokoro (~14.5% shorter).

   Measured on T4, voice=`default`, VRAM peak 3.46 GB (well within T4 16 GB),
   model load 55.58s, 270/270 scenes succeeded, 0 errors.

## Voice mapping (Chatterbox has no am_fenrir/am_puck/ef_dora set)

Chatterbox's `ChatterboxTTS.from_pretrained()` exposes a single built-in voice
(`default`). It does support per-voice timbre via reference-audio cloning
(`audio_prompt_path`), which is not yet prepared for this project.

| Channel            | Kokoro voice (old) | Chatterbox voice (new) | Why |
|--------------------|--------------------|------------------------|-----|
| crime-ledger       | am_puck            | `default`              | Matches the 10-scene compare kernel, which used `default` for crime-ledger. |
| mythology-slavic   | am_fenrir          | `default`              | Chatterbox ships one built-in voice; no Slavic-specific preset exists. |
| speculative-biology| am_fenrir          | `default`              | Same — single built-in voice; channel-distinct timbre would need a cloned reference clip (future work, not a timing blocker). |

All three channels use `default` for this measurement pass. Voice choice does not
affect duration (same model), so the timing re-measurement is valid for all. If
channel-distinct timbres become a requirement, source one reference clip per
channel and switch to `audio_prompt_path` voice cloning — a follow-up, out of
scope for the duration re-measurement.

## Scope note

This record covers the **decision reversal and the Part 1 duration re-measurement
only**. The pipeline-architecture follow-through (tool-registry re-registration,
sequential TTS+image kernel design, real GPU-hour quota tracking, Kokoro fallback
comments) is tracked separately under Part 2 and is intentionally NOT done here
until Part 1 is approved.
