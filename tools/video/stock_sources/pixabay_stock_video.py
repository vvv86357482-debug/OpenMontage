"""Pixabay stock video tool — BaseTool contract (fallback source).

Wraps the Pixabay Video API with channel-aware anti-anachronism filtering
and mandatory audio stripping (-an) on all downloaded clips.
Used as fallback when Pexels returns no usable results.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


_CHANNEL_SEARCH_TERMS: dict[str, list[str]] = {
    "dark-annals": [
        "candle flame",
        "old parchment",
        "burning torch",
        "gothic cathedral dark",
        "ancient stone wall",
        "medieval manuscript",
        "castle ruins fog",
    ],
    "crime-ledger": [
        "vintage typewriter",
        "file cabinet",
        "rain window night",
        "detective desk lamp",
        "old photograph darkroom",
        "police evidence",
        "black and white noir",
    ],
    "mind-tactics": [
        "laboratory clean",
        "data chart abstract",
        "scientist notes",
        "brain scan",
        "clean office",
        "research paper",
        "microscope",
    ],
}

_MODERN_TOURIST_BLACKLIST = [
    "tourist",
    "travel",
    "modern",
    "city tour",
    "selfie",
    "smartphone",
    "sunglasses",
    "vlog",
]


class PixabayStockVideo(BaseTool):
    name = "pixabay_stock_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_search"
    provider = "pixabay"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["cmd:ffmpeg"]
    install_instructions = (
        "Set PIXABAY_API_KEY in .env to enable Pixabay Video search "
        "(free key at https://pixabay.com/api/docs/)."
    )
    fallback_tools = []
    agent_skills = []

    capabilities = [
        "search_videos",
        "download_video",
        "trim_to_duration",
        "apply_channel_color_grade",
        "anti_anachronism_filter",
    ]
    supports = {
        "offline": False,
        "audio_strip": True,
        "color_grade": True,
    }
    best_for = [
        "free stock video fallback when Pexels is exhausted",
        "broad general-purpose footage",
        "community-contributed clips",
    ]
    not_good_for = [
        "specialized historical footage (skews recent/lifestyle)",
        "audio-sensitive compositions without manual review",
    ]

    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Search terms for Pixabay Video API"},
            "orientation": {"type": "string", "default": "landscape", "enum": ["landscape", "portrait", "square"]},
            "max_duration": {"type": "number", "default": 10.0, "description": "Maximum clip duration in seconds"},
            "min_duration": {"type": "number", "default": 2.0, "description": "Minimum clip duration in seconds"},
            "channel": {"type": "string", "description": "Channel name for anti-anachronism filtering and color grading"},
            "output_path": {"type": "string", "description": "Required for download_video and trim_to_duration operations"},
            "duration": {"type": "number", "default": 5.0, "description": "Target duration for trim_to_duration"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=500, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout", "api_error"])
    idempotency_key_fields = ["query", "orientation", "channel", "output_path", "duration"]
    side_effects = ["writes video file to output_path", "calls Pixabay API"]
    user_visible_verification = [
        "Verify downloaded clip content matches channel aesthetic (no modern tourist content)",
        "Check ffprobe confirms -an (no audio stream) and target duration",
    ]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if os.environ.get("PIXABAY_API_KEY") else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("PIXABAY_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="PIXABAY_API_KEY not set. Get a free key at https://pixabay.com/api/docs/ and add it to .env.",
            )

        operation = inputs.get("operation", "search_videos")
        channel = inputs.get("channel", "")

        if operation == "search_videos":
            return self._search_videos(inputs)
        elif operation == "download_video":
            return self._download_video(inputs)
        elif operation == "trim_to_duration":
            return self._trim_to_duration(inputs)
        elif operation == "apply_channel_color_grade":
            return self._apply_channel_color_grade(inputs)
        else:
            return ToolResult(success=False, error=f"Unknown operation: {operation}")

    def _search_videos(self, inputs: dict[str, Any]) -> ToolResult:
        import requests

        query = inputs.get("query", "")
        if not query:
            return ToolResult(success=False, error="query is required for search_videos")

        max_duration = inputs.get("max_duration", 10.0)
        min_duration = inputs.get("min_duration", 2.0)
        channel = inputs.get("channel", "")

        params = {
            "key": os.environ["PIXABAY_API_KEY"],
            "q": query,
            "per_page": min(max(inputs.get("per_page", 20), 3), 200),
            "page": max(inputs.get("page", 1), 1),
            "safesearch": "true",
        }
        if max_duration is not None:
            params["max_duration"] = int(max_duration)
        if min_duration is not None:
            params["min_duration"] = int(min_duration)

        r = requests.get("https://pixabay.com/api/videos/", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        hits = data.get("hits", []) or []

        candidates: list[dict[str, Any]] = []
        for h in hits:
            videos = h.get("videos", {})
            rend = _pick_rendition(videos, min_width=0)
            if rend is None:
                continue

            duration = float(h.get("duration", 0) or 0)
            tags = (h.get("tags", "") or "").lower()
            user = (h.get("user", "") or "").lower()

            if not self._passes_anachronism_filter(tags, user, channel):
                continue

            candidates.append({
                "id": str(h.get("id")),
                "source": "pixabay_video",
                "url": h.get("pageURL", ""),
                "download_url": rend["url"],
                "duration": duration,
                "width": rend["width"],
                "height": rend["height"],
                "creator": h.get("user", ""),
                "thumbnail": h.get("userImageURL", "") or videos.get("tiny", {}).get("thumbnail", ""),
                "tags": h.get("tags", ""),
                "channel": channel,
            })

        return ToolResult(
            success=True,
            data={"candidates": candidates, "count": len(candidates), "query": query, "channel": channel},
        )

    def _download_video(self, inputs: dict[str, Any]) -> ToolResult:
        import requests

        download_url = inputs.get("download_url") or inputs.get("url")
        output_path = Path(inputs.get("output_path", ""))
        if not download_url:
            return ToolResult(success=False, error="download_url or url is required")
        if not output_path:
            return ToolResult(success=False, error="output_path is required")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(download_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)

        if not output_path.exists():
            return ToolResult(success=False, error="Download failed — output file missing")

        return ToolResult(
            success=True,
            data={"output": str(output_path), "size_bytes": output_path.stat().st_size},
            artifacts=[str(output_path)],
        )

    def _trim_to_duration(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = Path(inputs.get("input_path", ""))
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_trimmed"))))
        duration = float(inputs.get("duration", 5.0))

        if not input_path.exists() or not input_path.is_file():
            return ToolResult(success=False, error=f"Input video not found: {input_path}")

        ffmpeg = _find_ffmpeg()
        cmd = [
            ffmpeg, "-y",
            "-ss", "0",
            "-t", str(duration),
            "-i", str(input_path),
            "-an",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        _run(cmd)

        if not output_path.exists():
            return ToolResult(success=False, error="FFmpeg trim produced no output")

        return ToolResult(
            success=True,
            data={"output": str(output_path), "duration_target": duration},
            artifacts=[str(output_path)],
        )

    def _apply_channel_color_grade(self, inputs: dict[str, Any]) -> ToolResult:
        input_path = Path(inputs.get("input_path", ""))
        output_path = Path(inputs.get("output_path", str(input_path.with_stem(f"{input_path.stem}_graded"))))
        channel = inputs.get("channel", "")

        if not input_path.exists() or not input_path.is_file():
            return ToolResult(success=False, error=f"Input video not found: {input_path}")

        ffmpeg = _find_ffmpeg()
        vf = _channel_color_filter(channel)
        cmd = [
            ffmpeg, "-y",
            "-i", str(input_path),
            "-an",
            "-vf", vf,
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        _run(cmd)

        if not output_path.exists():
            return ToolResult(success=False, error="FFmpeg color grade produced no output")

        return ToolResult(
            success=True,
            data={"output": str(output_path), "channel": channel, "filter": vf},
            artifacts=[str(output_path)],
        )

    def _passes_anachronism_filter(self, tags: str, user: str, channel: str) -> bool:
        text = f"{tags} {user}"
        for banned in _MODERN_TOURIST_BLACKLIST:
            if banned in text:
                return False
        return True


def _pick_rendition(
    videos: dict[str, Any],
    min_width: int = 0,
) -> dict | None:
    preference = ["large", "medium", "small", "tiny"]
    for tier in preference:
        rend = videos.get(tier)
        if not rend or not rend.get("url"):
            continue
        w = int(rend.get("width") or 0)
        h = int(rend.get("height") or 0)
        if w >= min_width:
            return {"url": rend["url"], "width": w, "height": h, "size": rend.get("size")}
    return None


def _channel_color_filter(channel: str) -> str:
    palettes = {
        "dark-annals": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131:0:sepia=0.4:saturation=0.7",
        "crime-ledger": "hue=s=0:gain_r=0.4:gain_b=0.1:gain_g=0.4:contrast=1.3:brightness=-0.1",
        "mind-tactics": "colorchannelmixer=.5:.5:.5:0:.5:.5:.5:0:.5:.5:.5:0:saturation=0.3:contrast=1.05",
    }
    return palettes.get(channel, "null")


def _find_ffmpeg() -> str:
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg to use stock video tools.")
    return ffmpeg


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)
    def _trim_and_color_grade(self, inputs: dict[str, Any]) -> ToolResult:
        """Trim and color-grade in a single ffmpeg pass (eliminates one CRF re-encode)."""
        import requests

        download_url = inputs.get("download_url") or inputs.get("url")
        output_path = Path(inputs.get("output_path", ""))
        in_s = float(inputs.get("in_seconds", 0))
        out_s = float(inputs.get("out_seconds", 0))
        duration = out_s - in_s
        channel = inputs.get("channel", "")

        if not download_url:
            return ToolResult(success=False, error="download_url or url is required")
        if not output_path:
            return ToolResult(success=False, error="output_path is required")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp_dl.mp4")

        r = requests.get(download_url, stream=True, timeout=120)
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)

        vf_parts: list[str] = []
        color_filter = _channel_color_filter(channel)
        if color_filter and color_filter != "null":
            vf_parts.append(color_filter)
        vf_parts.extend(["setsar=1", "fps=30"])

        ffmpeg = _find_ffmpeg()
        cmd = [
            ffmpeg, "-y",
            "-ss", str(in_s),
            "-t", str(duration),
            "-i", str(tmp),
            "-vf", ",".join(vf_parts),
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path),
        ]
        _run(cmd)
        tmp.unlink(missing_ok=True)

        if not output_path.exists():
            return ToolResult(success=False, error="Combined trim+color produced no output")

        return ToolResult(
            success=True,
            data={"output": str(output_path), "duration_target": duration, "channel": channel},
            artifacts=[str(output_path)],
        )
