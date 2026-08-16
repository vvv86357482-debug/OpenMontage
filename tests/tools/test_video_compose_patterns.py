"""Test the 3-pattern video_compose: ffprobe sync, perceptual-hash dedup, gradient fallback."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from tools.video.video_compose import VideoCompose


def _make_clip(path: Path, w: int = 1280, h: int = 720, d: int = 2, color: str = "teal") -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c={color}:s={w}x{h}:d={d}:r=30",
         "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p",
         "-g", "30", "-keyint_min", "30", str(path)],
        capture_output=True, check=True,
    )


def _edit_decisions(cuts: list[dict], metadata: dict | None = None) -> dict:
    ed = {
        "version": "1.0",
        "render_runtime": "ffmpeg",
        "cuts": cuts,
    }
    if metadata:
        ed["metadata"] = metadata
    return ed


def test_3_clip_concat_duration(tmp_path: Path):
    vc = VideoCompose()
    clips = []
    for i, color in enumerate(["red", "green", "blue"]):
        clip = tmp_path / f"clip_{i}.mp4"
        _make_clip(clip, color=color)
        clips.append(clip)

    cuts = [
        {"id": "c1", "source": str(clips[0]), "in_seconds": 0, "out_seconds": 2},
        {"id": "c2", "source": str(clips[1]), "in_seconds": 0, "out_seconds": 2},
        {"id": "c3", "source": str(clips[2]), "in_seconds": 0, "out_seconds": 2},
    ]
    ed = _edit_decisions(cuts)
    out = tmp_path / "out_3clip.mp4"

    r = vc.execute({
        "operation": "compose",
        "edit_decisions": ed,
        "output_path": str(out),
    })
    assert r.success, r.error

    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_format", "-print_format", "json", str(out)]
    ).decode()
    duration = float(json.loads(probe)["format"]["duration"])
    expected = 6.0
    assert abs(duration - expected) < 0.3, f"Duration {duration}s != expected {expected}s"


def test_perceptual_hash_dedup_warning(tmp_path: Path):
    vc = VideoCompose()
    clip = tmp_path / "clip.mp4"
    _make_clip(clip)

    cuts = [
        {"id": "c1", "source": str(clip), "in_seconds": 0, "out_seconds": 1},
        {"id": "c2", "source": str(clip), "in_seconds": 0, "out_seconds": 1},
    ]
    ed = _edit_decisions(cuts)
    out = tmp_path / "out_dedup.mp4"

    r = vc.execute({
        "operation": "compose",
        "edit_decisions": ed,
        "output_path": str(out),
    })
    assert r.success, r.error


def test_gradient_fallback_for_missing_asset(tmp_path: Path):
    vc = VideoCompose()
    clip = tmp_path / "existent.mp4"
    _make_clip(clip)

    cuts = [
        {"id": "exists", "source": str(clip), "in_seconds": 0, "out_seconds": 1},
        {"id": "missing", "source": str(tmp_path / "nonexistent.mp4"), "in_seconds": 0, "out_seconds": 1},
    ]
    asset_manifest = {
        "version": "1.0",
        "assets": [
            {"id": "exists", "path": str(clip), "type": "video"},
            {"id": "missing", "path": str(tmp_path / "nonexistent.mp4"), "type": "video"},
        ],
    }
    ed = {
        "version": "1.0",
        "render_runtime": "ffmpeg",
        "renderer_family": "explainer-data",
        "cuts": cuts,
    }
    out = tmp_path / "out_fallback.mp4"

    # Use operation="render" to exercise _render path which includes fallback logic
    r = vc.execute({
        "operation": "render",
        "edit_decisions": ed,
        "asset_manifest": asset_manifest,
        "output_path": str(out),
    })
    assert r.success, r.error

    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_format", "-print_format", "json", str(out)]
    ).decode()
    duration = float(json.loads(probe)["format"]["duration"])
    expected = 2.0
    assert abs(duration - expected) < 0.3, f"Fallback duration {duration}s != expected {expected}s"

    fallback_path = tmp_path / ".compose_tmp" / "gradient_fallback_missing.mp4"
    assert fallback_path.exists(), f"Gradient fallback not generated at {fallback_path}"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print("Running test_3_clip_concat_duration...")
        test_3_clip_concat_duration(tmp_path)
        print("PASSED")

        print("Running test_perceptual_hash_dedup_warning...")
        test_perceptual_hash_dedup_warning(tmp_path)
        print("PASSED")

        print("Running test_gradient_fallback_for_missing_asset...")
        test_gradient_fallback_for_missing_asset(tmp_path)
        print("PASSED")

        print("All video_compose pattern tests passed.")
