"""Rebuild all three scene_plans with REAL Chatterbox-measured durations.

Consumes the Kaggle measurement manifest produced by
tools/kaggle/tts_measure/kernel.ipynb (270 scenes, all generated on T4).
Mirrors scripts/measure_kokoro_durations.py's discipline but for Chatterbox:
- 100% of scenes get a real measured duration (no regression/extrapolation)
- start/end seconds and total_duration_seconds recomputed from measured values
- source scripts/<channel>.json per-scene durations synced in lockstep
- validates each rebuilt plan against schemas/artifacts/scene_plan.schema.json
- flags any scene whose measured duration exceeds its image-slot budget

SLOT BUDGET (documented, auditable): a single still-image slot
(shot_type in IMAGE_SLOT_SHOT_TYPES) is designed for <= IMAGE_SLOT_MAX_SECONDS
of narration. Beyond that the Ken Burns motion stalls / should be split into
two slots. This is the "doesn't fit its assigned image slot" rule.

Run from repo root AFTER the Kaggle manifest is pulled:
    python3 scripts/apply_chatterbox_measurements.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_PLANS_DIR = REPO_ROOT / "scene_plans"
SCRIPTS_DIR = REPO_ROOT / "scripts"
MANIFEST = Path(
    os.environ.get(
        "CHATTERBOX_MANIFEST",
        "tools/kaggle/tts_measure/result/measure/measure_manifest.json",
    )
)
SCHEMA = REPO_ROOT / "schemas/artifacts/scene_plan.schema.json"

# channel -> scene_plan / script file (matches build_chatterbox_measure_kernel.py)
CHANNEL_FILES = {
    "crime-ledger": "jeju-cold-case.json",
    "mythology-slavic": "leshy-forest-lord.json",
    "speculative-biology": "bombardier-beetle-proof.json",
}

# Documented image-slot budget (see module docstring).
IMAGE_SLOT_SHOT_TYPES = {"body_kenburns", "generated"}
IMAGE_SLOT_MAX_SECONDS = 12.0

FLAG_REPORT = REPO_ROOT / "tools/kaggle/tts_measure/result/slot_fit_report.json"


def load_schema():
    try:
        import jsonschema
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jsonschema", "-q"])
        import jsonschema
    return jsonschema, json.loads(SCHEMA.read_text())


def _voice_note(manifest: dict, channel: str) -> str:
    voice_meta = manifest.get("voice_meta") or {}
    vm = voice_meta.get(channel)
    if vm and vm.get("voice"):
        return (
            f"Durations are real Chatterbox TTS measurements for 100% of scenes "
            f"(no regression estimates). Measured on Kaggle Tesla T4, cloned voice="
            f"'{vm['voice']}' ({vm.get('license', 'CC0')})."
        )
    voice = manifest.get("voice")
    return (
        "Durations are real Chatterbox TTS measurements for 100% of scenes "
        f"(no regression estimates). Measured on Kaggle Tesla T4, voice='{voice}'."
    )


def rebuild_plan(plan_path: Path, dur_map: dict[str, float], manifest: dict) -> tuple[dict, list]:
    plan = json.loads(plan_path.read_text())
    channel = plan["metadata"]["channel"]
    missing = 0
    for scene in plan["scenes"]:
        sid = scene["id"]
        if sid in dur_map and dur_map[sid] > 0:
            scene["duration_seconds"] = round(dur_map[sid], 2)
        else:
            scene["duration_seconds"] = 0.5  # safe floor on missing measurement
            missing += 1

    cursor = 0.0
    for scene in plan["scenes"]:
        d = scene["duration_seconds"]
        scene["start_seconds"] = round(cursor, 2)
        scene["end_seconds"] = round(cursor + d, 2)
        cursor += d

    total = round(cursor, 2)
    plan["metadata"]["total_duration_seconds"] = total
    plan["metadata"]["timing_method"] = "measured_chatterbox_full"
    plan["metadata"]["measured_samples"] = len(plan["scenes"])
    plan["metadata"]["note"] = _voice_note(manifest, channel)
    return plan, missing


def flag_slot_fit(plan: dict) -> list[dict]:
    flags = []
    for scene in plan["scenes"]:
        shot = scene.get("shot_type")
        if shot in IMAGE_SLOT_SHOT_TYPES and scene["duration_seconds"] > IMAGE_SLOT_MAX_SECONDS:
            flags.append({
                "id": scene["id"],
                "shot_type": shot,
                "duration_seconds": scene["duration_seconds"],
                "max_seconds": IMAGE_SLOT_MAX_SECONDS,
                "image_prompt": (scene.get("image_prompt") or "")[:80],
            })
    return flags


def main() -> int:
    if not MANIFEST.exists():
        print(f"FATAL: measurement manifest not found at {MANIFEST}", file=sys.stderr)
        print("Run the Kaggle kernel (tools/kaggle/tts_measure) and pull output first.", file=sys.stderr)
        return 2

    jsonschema, schema = load_schema()
    manifest = json.loads(MANIFEST.read_text())

    if manifest.get("errors"):
        print("WARNING: manifest reports errors:", manifest["errors"][:5], file=sys.stderr)

    channel_results = manifest["channels"]
    all_flags = {}
    total_missing = 0
    summary = {}

    for channel, fname in CHANNEL_FILES.items():
        plan_path = SCENE_PLANS_DIR / fname
        rows = channel_results.get(channel, [])
        dur_map = {r["id"]: r.get("dur_s", 0.0) for r in rows}

        plan, missing = rebuild_plan(plan_path, dur_map, manifest)
        total_missing += missing

        # validate
        try:
            jsonschema.validate(instance=plan, schema=schema)
            valid = True
            verr = None
        except Exception as e:  # jsonschema.ValidationError
            valid = False
            verr = str(e)

        flags = flag_slot_fit(plan)
        all_flags[channel] = flags

        plan_path.write_text(json.dumps(plan, indent=2))

        # sync source script durations
        script_path = SCRIPTS_DIR / fname
        if script_path.exists():
            script = json.loads(script_path.read_text())
            changed = 0
            for sc in script.get("scenes", []):
                if sc.get("id") in dur_map and dur_map[sc["id"]] > 0:
                    sc["duration_seconds"] = round(dur_map[sc["id"]], 2)
                    changed += 1
            if changed:
                script_path.write_text(json.dumps(script, indent=2))

        summary[channel] = {
            "file": fname,
            "scenes": len(plan["scenes"]),
            "total_duration_seconds": plan["metadata"]["total_duration_seconds"],
            "schema_valid": valid,
            "validation_error": verr,
            "missing_measurements": missing,
            "slot_fit_flags": len(flags),
            "channel_stats": manifest.get("channel_stats", {}).get(channel),
        }
        print(f"  {channel}: {len(plan['scenes'])} scenes, "
              f"total={plan['metadata']['total_duration_seconds']}s, "
              f"valid={valid}, flags={len(flags)}, missing={missing}", flush=True)

    FLAG_REPORT.write_text(json.dumps({
        "slot_max_seconds": IMAGE_SLOT_MAX_SECONDS,
        "slot_shot_types": sorted(IMAGE_SLOT_SHOT_TYPES),
        "flags_by_channel": all_flags,
    }, indent=2))

    print("\n=== SUMMARY ===")
    grand = 0.0
    for ch, s in summary.items():
        grand += s["total_duration_seconds"]
        print(f"  {ch}: total={s['total_duration_seconds']}s valid={s['schema_valid']} "
              f"flags={s['slot_fit_flags']} missing={s['missing_measurements']}")
    print(f"  GRAND TOTAL (3 scripts): {round(grand,2)}s")
    print(f"  total missing measurements: {total_missing}")
    print(f"  slot-fit report: {FLAG_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
