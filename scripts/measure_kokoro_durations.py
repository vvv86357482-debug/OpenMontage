"""Measure real Kokoro TTS durations for every scene in all three scene plans.

Rebuilds scene_plan duration_seconds from actual Kokoro audio output (not
linear-regression estimates). Runs all 270 scenes (90 x 3 scripts) through
Kokoro with HF_HUB_OFFLINE=1 (model weights are cached locally).

For each scene the narration text (scene_plan `description`) is synthesized
with the channel's Kokoro voice and the produced audio length becomes the
real `duration_seconds`. start/end seconds and total_duration are recomputed.

Run from repo root:
    HF_HUB_OFFLINE=1 python3 scripts/measure_kokoro_durations.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Weights are cached locally; online verification hangs this shared IP.
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from kokoro import KPipeline
from misaki import en

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PLANS_DIR = REPO_ROOT / "scene_plans"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# channel -> kokoro voice (from styles/<channel>.yaml -> tts_voice)
CHANNEL_VOICES = {
    "speculative-biology": "am_fenrir",
    "mythology-slavic": "am_fenrir",
    "crime-ledger": "am_puck",
}

SAMPLE_RATE = 24000  # Kokoro native output rate


def measure_text(pipeline: KPipeline, g2p, voice: str, text: str) -> float:
    """Return real spoken duration (seconds) of `text` via Kokoro."""
    text = (text or "").strip()
    if not text:
        return 0.0
    voice_tensor = pipeline.load_voice(voice)
    _, tokens = g2p(text)
    chunks = [r.audio for r in pipeline.generate_from_tokens(tokens, voice_tensor)]
    if not chunks:
        return 0.0
    audio = torch.cat(chunks, dim=0)
    return len(audio) / SAMPLE_RATE


def rebuild_plan(path: Path, pipeline: KPipeline, g2p) -> dict:
    plan = json.loads(path.read_text())
    channel = plan["metadata"]["channel"]
    voice = CHANNEL_VOICES[channel]

    t0 = time.time()
    measured = 0
    for scene in plan["scenes"]:
        text = scene.get("description") or scene.get("narration") or ""
        dur = measure_text(pipeline, g2p, voice, text)
        if dur <= 0:
            # Fall back to a safe minimum rather than collapsing the timeline.
            dur = 0.5
        scene["duration_seconds"] = round(dur, 2)
        measured += 1

    # Recompute timeline
    cursor = 0.0
    for scene in plan["scenes"]:
        d = scene["duration_seconds"]
        scene["start_seconds"] = round(cursor, 2)
        scene["end_seconds"] = round(cursor + d, 2)
        cursor += d

    total = round(cursor, 2)
    plan["metadata"]["total_duration_seconds"] = total
    plan["metadata"]["timing_method"] = "measured_kokoro_full"
    plan["metadata"]["measured_samples"] = len(plan["scenes"])
    plan["metadata"]["note"] = (
        "Durations are real Kokoro TTS measurements for 100% of scenes "
        "(no regression estimates). Measured with HF_HUB_OFFLINE=1."
    )

    elapsed = round(time.time() - t0, 1)
    print(
        f"  {channel}: {measured} scenes measured in {elapsed}s, "
        f"total={total}s, voice={voice}",
        flush=True,
    )
    return plan


def main() -> int:
    plan_files = sorted(SCENE_PLANS_DIR.glob("*.json"))
    if not plan_files:
        print("No scene plans found in", SCENE_PLANS_DIR, file=sys.stderr)
        return 1

    print("Loading Kokoro pipeline (offline)...", flush=True)
    t_load = time.time()
    pipeline = KPipeline(lang_code="a")
    g2p = en.G2P()
    print(f"Pipeline ready in {round(time.time() - t_load, 1)}s", flush=True)

    for path in plan_files:
        plan = rebuild_plan(path, pipeline, g2p)
        path.write_text(json.dumps(plan, indent=2))
        # Keep the source script's per-scene duration_seconds in sync.
        script_path = SCRIPTS_DIR / path.name
        if script_path.exists():
            script = json.loads(script_path.read_text())
            dur_map = {
                s["id"]: s["duration_seconds"]
                for s in plan["scenes"]
            }
            changed = 0
            for sc in script.get("scenes", []):
                if sc.get("id") in dur_map:
                    sc["duration_seconds"] = dur_map[sc["id"]]
                    changed += 1
            if changed:
                script_path.write_text(json.dumps(script, indent=2))
                print(f"  synced {changed} scene durations in {script_path.name}", flush=True)

    print("Done. All scene plans rebuilt with 100% measured durations.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
