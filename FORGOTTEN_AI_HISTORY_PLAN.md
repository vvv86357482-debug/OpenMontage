# Forgotten History of AI — Execution Plan

> Provenance: this file was reconstructed on 2026-08-24 from verified session
> evidence (commands, manifests, commit hashes) because no prior plan file ever
> existed in the repo. Items marked **TODO(owner)** are unknowns only the
> project owner can supply — do not invent values for them.

## Channel Identity

- **Channel:** Forgotten History of AI
- **Format:** Archive-first historical documentary about AI's origins, 1943-2012
- **Scope guard:** NOT AI news, NOT current model coverage — anything post-2012 belongs to another channel
- **Style playbook:** `styles/forgotten-ai-history.yaml` (validated, committed `6e659e0`)
  - Period visual anchor: 1950s-1980s lab photography (Bell Labs / Stanford / MIT era), silver-gelatin monochrome, punch cards, oscilloscope traces; AI-reconstructed scenes must match period aesthetics, enforced via `asset_generation.image_negative_prompt` (augment_prompt_for_sana-compatible)
  - `forbidden_words`: revolutionary, genius, changed everything, changed the world
  - Closing rule: end on real historical consequence or open question — never a tidy wrap-up

## Voice

- **Reference:** `tools/kaggle/outputs/omnivoice/ref_voices/simon_evers.flac`
  - CC0, from OwenTyme/voice-zero (`voices/simon_evers.flac`), sha256 `3260517de0220d583890db9da0815bd0d53987b68df4d9301ff04c20a5bb64da`
  - 10.53s, FLAC 44.1kHz mono 16-bit
  - Smoke-tested via OmniVoice kernel v2: output pcm_s16le @24000 Hz, 6.95s narration
- Reuse as-is for now. **TODO(owner):** decide when/whether to clone a dedicated channel voice.

## Verified Infrastructure State (as of 2026-08-24)

| Component | State | Evidence |
|---|---|---|
| Sana-Sprint 1.6B Kaggle kernel | WORKING | kaggle `forts845/sana-sprint` v1 COMPLETE: Tesla T4, 2 steps, bf16 transformer + fp32 VAE, image 1,524,274 B / std 49.72 (gate: 50 KB / 5.0), 9.2 s, peak 12,684 MB |
| OmniVoice TTS Kaggle kernel | WORKING | kaggle `forts845/omnivoice` v2 COMPLETE: pcm_s16le @24000 Hz, 6.95 s audio, 15.79 s gen, peak 4,537 MB |
| Kernel source of truth | COMMITTED | `tools/kaggle/{sana_sprint,omnivoice}/`, smoke artifacts under `tools/kaggle/outputs/`, commit `6755bb1` |
| Style playbook | COMMITTED | `styles/forgotten-ai-history.yaml`, jsonschema PASS, accessibility pass=True, commit `6e659e0` |
| video_compose guards | PRESENT | phash dedup (`imagehash.phash`, threshold 3), gradient fallback for missing assets, `-shortest`/apad mux sync — commit `bad54ea` |
| Kilo Code config | WORKING (corrected) | OpenRouter direct with model "Ox Alpha" (`~/.local/share/kilo/auth.json` holds an openrouter key — verified Phase 0). Earlier project docs assumed "OmniRoute"/model auto — that was wrong; this row records the real setup. |
| Git remote | FORK | work happens on `vvv86357482-debug/OpenMontage` main; codespace token is denied write on `i3478421-hub/OpenMontage` |

### Known-good Kaggle constraints (do NOT re-litigate)

- `machine_shape: "NvidiaTeslaT4"` in kernel-metadata.json (real CLI field; confirmed Tesla T4 in run manifests)
- Sana Sprint: `num_inference_steps=2` exactly (4 raises ValueError); VAE forced to float32 (fp16 renders all-black); hard gate rejects <50 KB or pixel std <5.0
- OmniVoice: exact install line `pip install -q omnivoice imagehash librosa soundfile`; NumPy >=2.0 compatibility required (`np.ptp`, not `.ptp()` — v1 died on this); ffprobe gate enforces pcm_s16le @24000
- Outputs always land in tracked repo paths, never `/tmp`
- Kaggle auth: CLI 2.x reads `KAGGLE_API_TOKEN` (+ `KAGGLE_USERNAME` for id prefix); classic username/key pair is dead

## Phase Log

- **Phase 0 — Infrastructure Audit: DONE.** Repo audited against known-good claims: video_compose patterns present, 24 addyosmani skills present-but-untracked, zero cathy-swartz/genealogy leftovers, no plan file existed.
- **Phase 1 — Kaggle rebuild + smoke tests: DONE.** Both kernels green on first-or-second version (OmniVoice needed one diagnosed fix: NumPy 2.0 `ptp`). Commit `6755bb1`.
- **Phase 2 — Channel scaffolding: DONE.** Playbook created, validated, pushed (`6e659e0`). Response-style convention added to AGENTS.md (`1948a59`).
- **Phase 3 — Episode 1: ELIZA (IN PROGRESS).** Topic locked by owner: ELIZA (Weizenbaum, 1966) — chosen over Dartmouth 1956 as the stronger hook for episode one. Current step: primary-source research + hook/closing-line draft, awaiting owner approval before full script.

- **Phase 4 — Asset generation (IN PROGRESS, blocked 2026-08-24 on Kaggle auth).** Owner decisions recorded: runtime option (c) ~50 min @135wpm / 6,739 words, no padding; pipeline bound to documentary-montage. Script + scene_plan artifacts schema-valid and mirrored under `docs/forgotten-history-of-ai/episode-01/`. Prepared and committed: Sana v3 kernel (13 generate-slots, prompts pre-augmented via augment_prompt_for_sana, per-image 50KB/std>=5 gates) and OmniVoice v3 (7 narration segments, sentence-boundary chunking, per-segment pcm_s16le@24000 ffprobe gate). Source-archive audit: 4/7 slots rights-clear (Bundesarchiv CC-BY-SA Berlin; PD Detroit Harris-Ewing; Weizenbaum portraits PD + CC-BY-SA; CPHR jacket CC-BY-SA 4.0); 3 flagged slots RESOLVED by owner decision 2026-08-24: journal/Science cover plates -> typographic citation cards (provided text_cards, no fabricated art); CTSS console -> AI reconstruction (generate). Archive images downloaded with ATTRIBUTION.md under projects/forgotten-history-of-ai/assets/images/archive/. Sana batch regenerated at 14 prompts. RESUME STEPS after Codespace restart with secrets: (1) `tools/kaggle/push.sh sana_sprint`, poll COMPLETE, `kaggle kernels output forts845/sana-sprint -p projects/forgotten-history-of-ai/assets/images`, verify manifest sizes/std locally; (2) commit image batch; (3) same for omnivoice into assets/audio + local ffprobe; (4) commit; (5) STOP — owner reviews first/last/2-random images before edit_decisions.

## Open Budget Items

- **Kaggle GPU quota:** 29h44m of 30h weekly reported by owner this session, before accounting for the ~2 T4 smoke-test runs already consumed.
- **GitHub Codespaces budget:** check github.com/settings/billing manually — not exposed to this codespace's token (gh api `/settings/billing/usage` → 403, no user codespaces billing endpoint).

## Session Conventions

- Report format: see Response Style section in `AGENTS.md`.
- Always read `AGENT_GUIDE.md` first; production goes through pipelines (Rule Zero).
- Stage only files belonging to the task; keep the 24 untracked `.agents/skills/` folders out of unrelated commits until intentionally committed.
