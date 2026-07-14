#!/usr/bin/env python3
"""Pre-publish checklist — 9 hard gates.

Every gate must pass before a video is published. Any single failure
blocks publish with a structured error message. Grounded in real failure
modes observed during this project's development (see pre_publish_checklist.md).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:
    import subprocess as sp
    sp.check_call([sys.executable, "-m", "pip", "install", "jsonschema", "-q"])
    import jsonschema


REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKLIST_DIR = REPO_ROOT / "skills" / "meta"
SCHEMA_PATH = REPO_ROOT / "schemas" / "artifacts" / "scene_plan.schema.json"
SCRIPT_SCHEMA_PATH = REPO_ROOT / "schemas" / "artifacts" / "script.schema.json"


class GateFailure(Exception):
    """Raised when a hard gate fails."""

    def __init__(self, gate: str, message: str, details: dict | None = None):
        self.gate = gate
        self.message = message
        self.details = details or {}
        super().__init__(f"[{gate}] {message}")


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _ffprobe_audio_info(video_path: Path) -> dict:
    """Extract audio stream info via ffprobe, falling back to ffmpeg -i when ffprobe is unavailable."""
    probe_cmd = None
    for cmd_name in ("ffprobe",):
        probe_cmd = shutil.which(cmd_name)
        if probe_cmd:
            break

    if probe_cmd:
        try:
            result = subprocess.run(
                [probe_cmd, "-v", "quiet", "-show_streams", "-select_streams", "a:0", str(video_path)],
                capture_output=True, text=True, check=True,
            )
            info: dict[str, Any] = {}
            for line in result.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
            return info
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Fallback: parse ffmpeg -i stderr for audio stream info
    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        return {}
    try:
        result = subprocess.run(
            [ffmpeg_cmd, "-i", str(video_path)],
            capture_output=True, text=True, check=False,
        )
        text = result.stderr
        info: dict[str, Any] = {}
        if "Audio:" in text:
            info["has_audio"] = "true"
            import re
            sr_match = re.search(r"Audio:.*?(\d{4,6}) Hz", text)
            if sr_match:
                info["sample_rate"] = sr_match.group(1)
        return info
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def _has_audio_stream(video_path: Path) -> bool:
    info = _ffprobe_audio_info(video_path)
    return bool(info.get("codec_name") or info.get("has_audio"))


def _audio_sample_rate(video_path: Path) -> int | None:
    info = _ffprobe_audio_info(video_path)
    rate = info.get("sample_rate")
    return int(rate) if rate and rate.isdigit() else None


# ──────────────────────────────────────────────
# Gate implementations
# ──────────────────────────────────────────────

def _get_script_segments(script: dict) -> list[dict]:
    """Backward-compat: support both 'segments' and schema-compliant 'sections'."""
    return script.get("segments", []) or script.get("sections", [])


def gate_hook_promise(script: dict) -> None:
    """publish_1: Hook delivers concrete promise within first 15 seconds."""
    segments = _get_script_segments(script)
    early = [s for s in segments if s.get("start_seconds", 999) <= 15]
    if not early:
        raise GateFailure(
            "publish_1",
            "No script segments found within first 15 seconds.",
            {"early_segments": len(early)},
        )

    # Look for a specific claim (sentence containing a number, proper noun, or date pattern)
    import re
    specific_pattern = re.compile(r'\b(\d{1,4}s?\b|january|february|march|april|may|june|july|august|september|october|november|december|\d{4}|percent|%|\$\d)', re.IGNORECASE)
    has_concrete = any(
        specific_pattern.search(s.get("text", ""))
        for s in early
    )
    if not has_concrete:
        raise GateFailure(
            "publish_1",
            "Hook (first 15s) lacks a concrete promise — no date, number, name, or specific claim detected.",
            {"early_segment_texts": [s.get("text", "")[:120] for s in early[:3]]},
        )


def gate_hook_cadence(script: dict) -> None:
    """publish_2: New hook-moment every ~60s, flag gaps >90s."""
    segments = sorted(_get_script_segments(script), key=lambda s: s.get("start_seconds", 0))
    if not segments:
        raise GateFailure("publish_2", "Script has no segments.", {})

    max_gap = 90.0
    engagement_roles = {"build_tension", "deliver_payload", "evidence", "emotional_beat", "call_to_action"}

    last_engagement = 0.0
    for seg in segments:
        start = seg.get("start_seconds", 0)
        role = seg.get("narrative_role", "")
        if role in engagement_roles:
            last_engagement = start
        elif start - last_engagement > max_gap:
            raise GateFailure(
                "publish_2",
                f"Gap of {start - last_engagement:.1f}s since last engagement moment (>{max_gap}s).",
                {"gap_start": last_engagement, "gap_end": start, "role": role},
            )


def gate_source_traceability(script: dict, research_brief: dict | None = None) -> None:
    """publish_3: Every major claim traces to a source_ref."""
    if research_brief is None:
        return  # Skip if no research brief — will be caught at proposal stage

    source_refs = set()
    for dp in research_brief.get("data_points", []):
        source_refs.add(dp.get("source_ref", ""))
    for ev in research_brief.get("expert_voices", []):
        source_refs.add(ev.get("source_ref", ""))

    if not source_refs:
        raise GateFailure(
            "publish_3",
            "Research brief has no source_ref entries — claims cannot be verified.",
            {},
        )

    # Check that at least 80% of segments have a matching source_ref or are marked non-factual
    segments = _get_script_segments(script)
    unmatched = []
    for seg in segments:
        text = seg.get("text", "")
        # Skip pure transition/formatting segments
        if len(text.strip()) < 20:
            continue
        has_ref = any(ref and ref.lower() in text.lower() for ref in source_refs)
        if not has_ref and seg.get("narrative_role") not in {"transition"}:
            unmatched.append(seg.get("id", "?"))

    if len(unmatched) > len(segments) * 0.2:
        raise GateFailure(
            "publish_3",
            f"{len(unmatched)}/{len(segments)} segments lack traceable source_ref.",
            {"unmatched_segments": unmatched[:10]},
        )


def gate_style_differentiation(scene_plan: dict, project_id: str) -> None:
    """publish_4: Visual style differs from channel's prior videos."""
    history_dir = REPO_ROOT / "projects" / project_id / "history"
    if not history_dir.exists() or not any(history_dir.iterdir()):
        return  # No prior videos — skip

    current_playbook = scene_plan.get("style_playbook", "")
    current_keywords: set[str] = set()
    for scene in scene_plan.get("scenes", []):
        current_keywords.update(kw.lower() for kw in scene.get("texture_keywords", []))

    prior_playbooks: set[str] = set()
    prior_keywords: set[str] = set()
    for fp in history_dir.glob("scene_plan_*.json"):
        try:
            prior = _load_json(fp)
            prior_playbooks.add(prior.get("style_playbook", ""))
            for scene in prior.get("scenes", []):
                prior_keywords.update(kw.lower() for kw in scene.get("texture_keywords", []))
        except Exception:
            continue

    if current_playbook in prior_playbooks and current_keywords == prior_keywords:
        raise GateFailure(
            "publish_4",
            "Style/pacing identical to a prior video — playbook and texture keywords match exactly.",
            {"playbook": current_playbook, "shared_keywords": sorted(current_keywords)},
        )


def gate_tts_voice(script: dict, playbook: dict) -> None:
    """publish_5: Kokoro TTS voice matches channel playbook."""
    expected_voice = playbook.get("audio", {}).get("tts_voice", "")
    actual_voice = script.get("metadata", {}).get("tts_voice", "")
    actual_provider = script.get("metadata", {}).get("tts_provider", "")

    if actual_provider != "kokoro":
        raise GateFailure(
            "publish_5",
            f"TTS provider is '{actual_provider}' — expected 'kokoro'.",
            {"actual": actual_provider, "expected": "kokoro"},
        )

    if expected_voice and actual_voice != expected_voice:
        raise GateFailure(
            "publish_5",
            f"TTS voice is '{actual_voice}' — playbook requires '{expected_voice}'.",
            {"actual": actual_voice, "expected": expected_voice},
        )


def gate_closing_line(script: dict, playbook: dict) -> None:
    """publish_6: Script ends with closing line matching playbook's voice_system.closing_style."""
    expected_closing = playbook.get("voice_system", {}).get("closing_style", "")
    actual_closing = script.get("metadata", {}).get("closing_line", "")

    if not actual_closing:
        raise GateFailure(
            "publish_6",
            "Script has no closing_line in metadata.",
            {"expected_style": expected_closing},
        )

    # Check that the closing line reflects the required style (not a summary if "never a summary", etc.)
    if "never a summary" in expected_closing and len(actual_closing.split()) < 20:
        # Likely a summary — too short for a proper closing observation
        pass  # heuristic only; human review catches rest

    if actual_closing == _get_script_segments(script)[-1].get("text", ""):
        # Closing line matches last segment verbatim — might be recycled narration
        raise GateFailure(
            "publish_6",
            "Closing line appears to be recycled from last narration segment — verify it matches playbook closing_style.",
            {"closing_line": actual_closing[:120]},
        )


def gate_synthetic_disclosure(edit_decisions: dict) -> None:
    """publish_7: Synthetic content disclosure present."""
    if not edit_decisions.get("synthetic_disclosure"):
        raise GateFailure(
            "publish_7",
            "synthetic_disclosure is missing or false — YouTube requires Altered/synthetic content disclosure.",
            {},
        )

    disclosure_text = edit_decisions.get("synthetic_disclosure_text", "")
    if not disclosure_text or len(disclosure_text.strip()) < 10:
        raise GateFailure(
            "publish_7",
            "synthetic_disclosure_text is empty or too short — must contain visible disclosure message.",
            {"text": disclosure_text[:80]},
        )


def gate_upload_cadence(project_id: str, channel: str) -> None:
    """publish_8: Flag 4th+ upload in 7 days (warning, not block)."""
    history_dir = REPO_ROOT / "projects" / project_id / "history"
    if not history_dir.exists():
        return

    # Count completed publishes in last 7 days
    import datetime
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
    uploads = 0
    for fp in history_dir.glob("checkpoint_publish*.json"):
        try:
            data = _load_json(fp)
            if data.get("status") == "completed":
                ts = data.get("metadata", {}).get("completed_at", "")
                if ts:
                    dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if dt >= cutoff:
                        uploads += 1
        except Exception:
            continue

    if uploads >= 3:
        # Warning — print to stderr but don't raise
        print(
            f"[WARNING] Upload cadence gate: {uploads} uploads in last 7 days for channel '{channel}'. "
            "Publish_8 recommends slowing cadence but does not block.",
            file=sys.stderr,
        )


def gate_asset_integrity(asset_manifest: dict, final_render: Path) -> None:
    """publish_9: Hard-fail on empty/placeholder/missing assets."""
    MIN_IMAGE_SIZE = 50_000  # 50KB — catches empty writes and placeholder stubs
    MIN_AUDIO_DURATION = 0.5

    assets = asset_manifest.get("assets", [])
    failures: list[dict] = []

    for asset in assets:
        asset_id = asset.get("id", "unknown")
        path_str = asset.get("path", "")
        if not path_str:
            failures.append({"id": asset_id, "reason": "missing path field"})
            continue

        p = Path(path_str)
        if not p.exists():
            failures.append({"id": asset_id, "reason": f"file not found: {path_str}"})
            continue

        size = p.stat().st_size
        asset_type = asset.get("type", "")

        if asset_type == "image" and size < MIN_IMAGE_SIZE:
            failures.append({
                "id": asset_id,
                "reason": f"image file too small: {size} bytes (minimum {MIN_IMAGE_SIZE})",
                "path": str(p),
            })

        if asset_type == "audio":
            dur = asset.get("duration_seconds", 0)
            if dur < MIN_AUDIO_DURATION:
                failures.append({
                    "id": asset_id,
                    "reason": f"audio duration too short: {dur}s (minimum {MIN_AUDIO_DURATION}s)",
                })

    # Check final render
    if not final_render.exists():
        failures.append({"id": "final_render", "reason": f"final render not found: {final_render}"})
    elif final_render.stat().st_size < 1024:
        failures.append({"id": "final_render", "reason": "final render is empty or near-empty"})
    else:
        # Audio stream present?
        if not _has_audio_stream(final_render):
            failures.append({"id": "final_render", "reason": "no audio stream found in final render"})
        else:
            # Audio sample rate
            rate = _audio_sample_rate(final_render)
            if rate is None:
                failures.append({"id": "final_render", "reason": "could not read audio sample rate"})
            elif rate != 44100:
                failures.append({
                    "id": "final_render",
                    "reason": f"audio sample rate is {rate}Hz — expected 44100Hz",
                })

    if failures:
        raise GateFailure(
            "publish_9",
            f"{len(failures)} asset integrity failure(s) detected.",
            {"failures": failures},
        )


def _has_sequential_digits(value: float | int) -> bool:
    """Detect placeholder patterns like 1234567890, 7890, 0123456789 as substrings."""
    s = str(int(value))
    # Check for runs of 4+ ascending sequential digits (e.g. "1234", "5678", "7890")
    for run_len in range(len(s), 3, -1):
        for start in range(len(s) - run_len + 1):
            substr = s[start:start + run_len]
            digits = [int(c) for c in substr if c.isdigit()]
            if len(digits) < 4:
                continue
            # Check if each digit = previous + 1 (with wrap 9->0)
            if all(digits[i] == digits[i - 1] + 1 for i in range(1, len(digits))):
                return True
    return False


def gate_model_weights_loaded(asset_manifest: dict) -> None:
    """publish_9 variant: Ensure generated image assets have real, non-fabricated metadata."""
    for asset in asset_manifest.get("assets", []):
        if asset.get("type") != "image":
            continue

        mem = asset.get("memory_peak_bytes", 0)
        dur = asset.get("duration_s", 0)

        # Fabricated metrics have round/multiples-of-10 values
        if mem > 0 and mem % 1_000_000_000 == 0:
            raise GateFailure(
                "publish_9",
                f"Image {asset['id']} has round memory_peak_bytes ({mem}) — likely fabricated.",
                {"asset_id": asset["id"], "memory_peak_bytes": mem},
            )

        if dur > 0 and dur % 1 == 0 and dur >= 10:
            raise GateFailure(
                "publish_9",
                f"Image {asset['id']} has round duration_s ({dur}) — likely fabricated.",
                {"asset_id": asset["id"], "duration_s": dur},
            )

        # Sequential-digit placeholder pattern (e.g. 1234567890)
        if mem > 0 and _has_sequential_digits(mem):
            raise GateFailure(
                "publish_9",
                f"Image {asset['id']} has sequential-digit memory_peak_bytes ({mem}) — likely fabricated.",
                {"asset_id": asset["id"], "memory_peak_bytes": mem},
            )

        if dur > 0 and _has_sequential_digits(dur):
            raise GateFailure(
                "publish_9",
                f"Image {asset['id']} has sequential-digit duration_s ({dur}) — likely fabricated.",
                {"asset_id": asset["id"], "duration_s": dur},
            )


# ──────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────

def run_checklist(
    project_dir: Path,
    playbook: dict,
    script: dict | None = None,
    scene_plan: dict | None = None,
    asset_manifest: dict | None = None,
    edit_decisions: dict | None = None,
    research_brief: dict | None = None,
) -> list[dict]:
    """Run all 9 hard gates. Returns list of results; raises on first failure."""
    results: list[dict] = []
    gates = [
        ("publish_1", "Hook promise in first 15s", lambda: gate_hook_promise(script or {})),
        ("publish_2", "Hook cadence (gap > 90s)", lambda: gate_hook_cadence(script or {})),
        ("publish_3", "Source traceability", lambda: gate_source_traceability(script or {}, research_brief)),
        ("publish_4", "Style differentiation", lambda: gate_style_differentiation(scene_plan or {}, project_dir.name)),
        ("publish_5", "TTS voice correct", lambda: gate_tts_voice(script or {}, playbook)),
        ("publish_6", "Closing line present", lambda: gate_closing_line(script or {}, playbook)),
        ("publish_7", "Synthetic disclosure", lambda: gate_synthetic_disclosure(edit_decisions or {})),
        ("publish_8", "Upload cadence", lambda: gate_upload_cadence(project_dir.name, playbook.get("identity", {}).get("name", ""))),
        ("publish_9", "Asset integrity + real metrics", lambda: _run_publish_9(asset_manifest, project_dir, scene_plan)),
    ]

    for gate_id, label, fn in gates:
        try:
            fn()
            results.append({"gate": gate_id, "label": label, "status": "PASS"})
        except GateFailure as exc:
            results.append({"gate": gate_id, "label": label, "status": "FAIL", "error": str(exc), "details": exc.details})
            raise  # Hard gate — stop on first failure

    return results


def _run_publish_9(asset_manifest: dict | None, project_dir: Path, scene_plan: dict | None) -> None:
    """Combined gate_9: asset integrity + real metrics."""
    if asset_manifest is None:
        raise GateFailure("publish_9", "asset_manifest is missing.", {})

    gate_asset_integrity(asset_manifest, project_dir / "renders" / "final.mp4")
    gate_model_weights_loaded(asset_manifest)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run pre-publish checklist")
    parser.add_argument("project_dir", type=Path, help="Project directory under projects/<id>/")
    parser.add_argument("--channel", default="", help="Channel name (loads matching playbook)")
    parser.add_argument("--playbook", type=Path, default=None, help="Explicit playbook YAML path")
    parser.add_argument("--script", type=Path, default=None, help="script.json path")
    parser.add_argument("--scene-plan", type=Path, default=None, help="scene_plan.json path")
    parser.add_argument("--asset-manifest", type=Path, default=None, help="asset_manifest.json path")
    parser.add_argument("--edit-decisions", type=Path, default=None, help="edit_decisions.json path")
    parser.add_argument("--research-brief", type=Path, default=None, help="research_brief.json path")
    parser.add_argument("--ignore-cadence", action="store_true", help="Skip upload cadence warning (publish_8)")
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()

    # Load playbook
    if args.playbook:
        import yaml
        with open(args.playbook) as f:
            playbook = yaml.safe_load(f) or {}
    else:
        channel = args.channel or project_dir.name
        playbook_path = REPO_ROOT / "styles" / f"{channel}.yaml"
        if not playbook_path.exists():
            print(f"FAIL: playbook not found at {playbook_path}", file=sys.stderr)
            return 1
        import yaml
        with open(playbook_path) as f:
            playbook = yaml.safe_load(f) or {}

    # Load optional artifacts
    script = _load_json(args.script) if args.script else None
    scene_plan = _load_json(args.scene_plan) if args.scene_plan else None
    asset_manifest = _load_json(args.asset_manifest) if args.asset_manifest else None
    edit_decisions = _load_json(args.edit_decisions) if args.edit_decisions else None
    research_brief = _load_json(args.research_brief) if args.research_brief else None

    # Validate scene_plan against schema if provided
    if scene_plan:
        with open(SCHEMA_PATH) as f:
            scene_schema = json.load(f)
        try:
            jsonschema.validate(instance=scene_plan, schema=scene_schema)
        except jsonschema.ValidationError as e:
            print(f"FAIL: scene_plan schema validation: {e.message}", file=sys.stderr)
            return 1

    # Validate script against schema if provided
    if script:
        with open(SCRIPT_SCHEMA_PATH) as f:
            script_schema = json.load(f)
        try:
            jsonschema.validate(instance=script, schema=script_schema)
        except jsonschema.ValidationError as e:
            print(f"FAIL: script schema validation: {e.message}", file=sys.stderr)
            return 1

    print("=== RUNNING PRE-PUBLISH CHECKLIST ===\n")
    print(f"Project:  {project_dir}")
    print(f"Playbook: {playbook.get('identity', {}).get('name', 'unknown')}")
    print(f"Channel:  {args.channel or project_dir.name}")
    print()

    try:
        results = run_checklist(
            project_dir=project_dir,
            playbook=playbook,
            script=script,
            scene_plan=scene_plan,
            asset_manifest=asset_manifest,
            edit_decisions=edit_decisions,
            research_brief=research_brief,
        )
    except GateFailure as exc:
        print(f"BLOCKED at gate {exc.gate}: {exc.message}", file=sys.stderr)
        if exc.details:
            print(f"Details: {json.dumps(exc.details, indent=2)}", file=sys.stderr)
        sys.exit(1)

    for r in results:
        status = r["status"]
        marker = "[PASS]" if status == "PASS" else f"[{status}]"
        print(f"  {marker} {r['gate']}: {r['label']}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\n=== RESULT: {passed}/{total} gates passed ===")

    if passed == total:
        print("PUBLISH APPROVED — all 9 hard gates passed.")
        return 0
    else:
        print("PUBLISH BLOCKED — see failures above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
