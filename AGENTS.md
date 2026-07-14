# OpenMontage

**MANDATORY: Read `AGENT_GUIDE.md` before responding to ANY user message.**
It contains routing rules that determine your first action based on what the user asked.

## Project Architecture

```
Codespace (agent skills) → scene_plan.json → Kaggle GPU (SANA-Sprint/FLUX + Chatterbox TTS, sequential in one session) → assets/ + audio/
                        → script.json → Chatterbox TTS (Kaggle T4) → audio/   [Kokoro = manual fallback only]
                        → video_compose.py (ffmpeg) → final render
                        → run_pre_publish_checklist.py → 9 hard gates → publish
```

> **TTS engine reversal (2026-07-11):** Chatterbox is now the sole voice engine,
> replacing Kokoro. This **reverses** the earlier "Kokoro remains the sole TTS
> engine" decision. The prior decision is quoted and the rationale for the change
> is logged in `docs/decisions/tts-engine-chatterbox-reversal.md`
> (machine-readable: `docs/decisions/decision_log.json`). Kokoro is kept installed
> as a documented manual fallback — not deleted, not the default. Part 2
> (registry re-registration, sequential kernel, GPU-hour quota tracking) is
> pending Part 1 approval.

- **Free stack only:** Chatterbox TTS (Kaggle T4 GPU), Kaggle SANA-Sprint/FLUX, Pexels/Pixabay stock, ffmpeg. Kokoro kept as manual fallback only (local CPU, when Kaggle GPU quota is out or HF Hub rate-limits Chatterbox).
- **No API-first tools:** ElevenLabs, FAL.ai, Google Cloud, etc. are NOT the default path.
- **3 channels:** dark-annals (am_fenrir), crime-ledger (am_puck), mind-tactics (ef_dora).
- **Playbooks:** `styles/{channel}.yaml` — single source of truth for voice, prompts, negatives, output settings.
- **Chatterbox voice:** single built-in `default` voice for all channels (no am_fenrir/am_puck/ef_dora set); channel-distinct timbre via reference-audio cloning is future work, not a timing blocker.

## Hard-Fail Philosophy

- **No silent placeholders.** If a tool cannot complete its task, it must fail loudly with a specific error message.
- **No fabricated metrics.** Memory/duration values in manifests must be real hardware measurements. The pre-publish checklist (publish_9) catches sequential-digit patterns, round GB boundaries, and sub-50KB image files.
- **No fallback substitution.** An asset that fails generation is not silently replaced with a generic alternative.
- **No hardcoded device_map='auto'.** Always use `device_map='cuda'` on CUDA, `None` on CPU.

## Singleton Model Loading

All model/pipeline loading uses the singleton pattern — load once per session, reuse for all items.
Never reload inside a per-item loop. See `tools/kaggle/kernel.ipynb` cell 3 for the pattern.

## Credential Handling Rules

- Never commit real API keys, tokens, or passwords.
- Use `${{ secrets.X }}` in GitHub Actions, `os.environ.get()` in Python.
- Kaggle Secrets: read via `UserSecretsClient.get_secret()`, never hardcoded.
- `.env` files are gitignored — never paste credentials into YAML, notebooks, or Python source.
- If a secret fetch fails, report it clearly. Do not retry silently.

## STOP-Checkpoint Workflow

Every PART ends with a STOP checkpoint showing:
1. Files changed (paths)
2. Current task state
3. Explicit approval request

Sections are approved incrementally. Do not proceed to the next PART until the current one is approved.

## Known Blockers

- **T4 GPU allocation:** Kaggle's free GPU scheduler intermittently assigns P100 instead of T4.
  The kernel has a hard gate that aborts on CC<7 (P100) so mis-provisioned runs fail fast.
- **HF_TOKEN secrets service:** Can return `ConnectionError` transiently. SANA-Sprint is public and
  does not need HF_TOKEN; FLUX.1 Schnell does. Retry later if the service is down.
- **diffusers not installed locally:** Model loading code runs on Kaggle, not in the Codespace.
  Cannot verify `from_pretrained` signatures locally — verify on Kaggle or in docs.
- **ffprobe not available:** Some environments only have ffmpeg. Checklist uses ffmpeg -i stderr
  as fallback for audio stream detection.

## Response Format

Use the 3-line summary convention for PART reports:
```
═══════════════════════════════════════════
PART X — NAME
STATUS: [APPROVED / BLOCKED / NEEDS APPROVAL]
═══════════════════════════════════════════

[Summary ≤3 lines]

[Evidence if needed]

STOP — [specific question]
```

No preamble, no backstory, no postamble. Direct findings only.
