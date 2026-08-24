# Kaggle Kernels

Remote GPU jobs (T4) for narration and still-image generation.

## Prerequisites (Codespace secrets, never pasted in chat/terminal)

The pinned CLI here is `kaggle` >= 2.x, whose auth is a single API token
(classic `KAGGLE_USERNAME`/`KAGGLE_KEY` pairs are 1.x-only and ignored).

Add repo/user secrets:
- `KAGGLE_USERNAME` - plain account name (used for the kernel `id` prefix)
- `KAGGLE_API_TOKEN` - from kaggle.com -> Settings -> API -> Generate New Token

New secrets only appear after stopping + starting the Codespace.

## Layout

| Path | Purpose |
|------|---------|
| `sana_sprint/` | Sana-Sprint 1.6B image kernel (`machine_shape: NvidiaTeslaT4`, steps=2, fp32 VAE fix, 50KB/std>=5 gate) |
| `omnivoice/`   | OmniVoice voice-clone TTS kernel (simon_evers.flac CC0 ref, ffprobe 24kHz/16-bit PCM gate) |
| `outputs/<kernel>/` | Pulled smoke-test artifacts (manifest.json, media) — tracked in git |

## Usage

```bash
tools/kaggle/push.sh sana_sprint    # or omnivoice
kaggle kernels status <user>/<slug>
kaggle kernels output <user>/<slug> -p tools/kaggle/outputs/sana_sprint
```

Known-good constraints are documented at the top of each kernel script.
Do not "fix" them; each one closed a real bug (P100 fallback, ValueError on
steps=4, all-black images with fp16 VAE, missing-audio-stream ModuleNotFoundError).
