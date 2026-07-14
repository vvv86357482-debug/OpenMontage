# Pre-Publish Checklist

Hard gates that must ALL pass before any video is published/uploaded.
Any single failure blocks publish — no silent pass-through.

## Audit basis

These gates are grounded in real failure modes observed during this project's development:

| Gate | Real failure mode it prevents |
|------|-------------------------------|
| audio_sample_rate | Three separate ffmpeg calls missing `-ar 44100` in PART 1; chipmunk pitch-shift regression from prior session |
| audio_stream_present | Silent audio stream drops when codec copy fails on mismatched source |
| no_placeholder_assets | Prior session had silent placeholder image substitution that still reported checklist PASS |
| real_file_sizes | Sub-50KB images indicate failed generation or empty placeholder writes |
| manifest_sanity | Round/non-varying memory and duration values indicate fabricated metrics |
| anti_anachronism_check | Generated images with modern elements (cars, phones, electric light) breaking period-channel content |
| tts_voice_correct | Wrong Kokoro voice for channel (e.g. fenrir on crime-ledger) |
| script_closing_style | Scripts missing required closing line per channel voice_system |
| script_forbidden_words | Scripts containing channel-banned terms (e.g. "incredible" on dark-annals) |
| synthetic_disclosure | YouTube requires synthetic content disclosure — missing toggle causes post-upload demonetization/removal |
| model_loading_verified | Kaggle runs that never loaded the model weights (silent import failures, cache misses on gated repos) |
| gpu_requirement_met | P100 assigned instead of T4, causing silent fallback to wrong pipeline |

## Gate descriptions

### publish_1: Hook promise within 15 seconds
- **Check:** The script's first 15 seconds of narration must contain a concrete, specific promise or claim — not a generic intro, not "in this video we'll learn about..."
- **Automated:** Parse script segments with `start_seconds <= 15`, verify at least one sentence contains a specific noun phrase (date, name, number, location) rather than meta-commentary.
- **Manual review trigger:** If automated check is inconclusive, flag for human review with the first 15s transcript excerpt.

### publish_2: Hook cadence (new moment every ~60s)
- **Check:** A new hook-moment, surprising fact, or structural shift appears at least every 60 seconds. Flag gaps > 90 seconds.
- **Automated:** Walk script segments; for each 90-second window starting at t=0, verify at least one segment has `enhancement_cue != null` or `narrative_role` in {`build_tension`, `deliver_payload`, `evidence`}.

### publish_3: Source traceability
- **Check:** Every major factual claim in the script must trace to a `source_ref` from the research_brief.
- **Automated:** For each sentence in the script containing a statistic, date, name, or specific assertion, verify a matching `source_ref` exists in `research_brief.data_points` or `expert_voices`.

### publish_4: Style differentiation
- **Check:** Visual style and pacing of this video must differ meaningfully from the channel's last 2-3 published videos.
- **Automated:** Compare `style_playbook` field and at least 2 `texture_keywords` or `motion.pacing_rules` values against prior scene_plan artifacts for the same channel. Flag if identical.
- **Note:** This requires prior scene_plan artifacts to exist. If this is the channel's first video, skip with a `style_diff_not_applicable` note.

### publish_5: Kokoro TTS voice verification
- **Check:** Narration must use Kokoro (synthetic), not an unavailable/API-based TTS.
- **Automated:** Verify `script.metadata.tts_provider == "kokoro"` and `script.metadata.tts_voice` matches the channel playbook's `audio.tts_voice`.

### publish_6: Closing line present
- **Check:** Video ends with the closing line style defined in the channel playbook's `voice_system.closing_style`.
- **Automated:** Verify `script.metadata.closing_line` is non-empty and `script.metadata.closing_style` matches `playbook.voice_system.closing_style`.

### publish_7: Synthetic content disclosure
- **Check:** Output a visible, audible, or metadata-level disclosure that the content is AI-generated/synthetic.
- **Automated:** Verify `edit_decisions.synthetic_disclosure` is `true` and the disclosure text is present in the final render's metadata or as an on-screen text overlay in the composition.
- **YouTube requirement:** "Enable Altered or synthetic content disclosure toggle in YouTube Studio before uploading." This message must appear in the publish output.

### publish_8: Upload cadence
- **Check:** Flag if this would be a 4th+ upload in a rolling 7-day window for this channel.
- **Automated:** Read channel's upload history (from `projects/<id>/history/` or external tracker), count uploads in the last 7 days. If count >= 3 and this would be the 4th, flag as `upload_cadence_warning`.
- **Note:** This is a warning, not a hard block — the agent should surface it but not refuse to publish without explicit user instruction.

### publish_9: Asset integrity (CRITICAL — prevents silent placeholder substitution)
- **Check:** Every asset listed in the manifest must exist on disk, be non-zero-byte, and have a realistic file size.
- **Automated:**
  - For every asset in `asset_manifest.assets`:
    - `path` must exist on filesystem
    - `size_bytes >= 50_000` (50KB minimum — catches empty writes and placeholder stubs)
    - Image assets: `size_bytes >= 50_000` and file extension matches declared type
    - Audio assets: `duration_seconds > 0` and matches script segment duration within ±10%
  - For the final render:
    - Video file exists, duration > 0
    - Audio stream exists (ffprobe confirms at least one audio stream)
    - Audio sample rate is 44100 Hz
- **Hard fail:** If ANY asset fails, abort publish and write `fail_job()` with the specific asset ID and failure reason. This is the gate that catches the silent placeholder-substitution bug from prior sessions.
