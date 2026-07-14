"""Pexels stock video tool — BaseTool contract.

Wraps the Pexels video API with channel-aware anti-anachronism filtering
and mandatory audio stripping (-an) on all downloaded clips.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import time
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
        "candle flame close up",
        "old parchment texture",
        "burning torch fire",
        "gothic cathedral interior dark",
        "ancient stone wall texture",
        "medieval manuscript pages",
        "castle ruins fog",
    ],
    "crime-ledger": [
        "vintage typewriter close up",
        "file cabinet drawer",
        "rain on window night",
        "detective office desk lamp",
        "old photograph darkroom",
        "police evidence bag",
        "black and white film noir",
    ],
    "mind-tactics": [
        "laboratory clean modern",
        "data chart abstract",
        "scientist notes writing",
        "brain scan medical",
        "clean office minimal",
        "research paper documents",
        "microscope close up",
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


class PexelsStockVideo(BaseTool):
    name = "pexels_stock_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_search"
    provider = "pexels"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["cmd:ffmpeg"]
    install_instructions = (
        "Set PEXELS_API_KEY in .env to enable Pexels stock search "
        "(free key at https://www.pexels.com/api/)."
    )
    fallback_tools = ["pixabay_stock_video"]
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
        "free stock video for hook segments",
        "texture/object-based historical footage",
        "channel-specific visual tone matching",
    ]
    not_good_for = [
        "AI-generated video (use kaggle_image or flux_image)",
        "long-form narrative footage",
    ]

    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "description": "Search terms for Pexels video API"},
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
    side_effects = ["writes video file to output_path", "calls Pexels API"]
    user_visible_verification = [
        "Verify downloaded clip content matches channel aesthetic (no modern tourist content)",
        "Check ffprobe confirms -an (no audio stream) and target duration",
    ]

    def get_status(self) -> ToolStatus:
        return ToolStatus.AVAILABLE if os.environ.get("PEXELS_API_KEY") else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("PEXELS_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="PEXELS_API_KEY not set. Get a free key at https://www.pexels.com/api/ and add it to .env.",
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

        orientation = inputs.get("orientation", "landscape")
        max_duration = inputs.get("max_duration", 10.0)
        min_duration = inputs.get("min_duration", 2.0)
        channel = inputs.get("channel", "")

        params = {
            "query": query,
            "per_page": inputs.get("per_page", 20),
            "page": inputs.get("page", 1),
        }
        if orientation:
            params["orientation"] = orientation

        headers = {"Authorization": os.environ["PEXELS_API_KEY"]}
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        videos = data.get("videos", []) or []

        candidates: list[dict[str, Any]] = []
        for v in videos:
            duration = float(v.get("duration", 0) or 0)
            if duration < min_duration or duration > max_duration:
                continue
            if not self._is_modern_tourist_content(v):
                continue

            rend = _pick_video_rendition(v.get("video_files", []) or [])
            if rend is None:
                continue

            user = v.get("user") or {}
            candidates.append({
                "id": str(v.get("id")),
                "source": "pexels",
                "url": v.get("url", ""),
                "download_url": rend.get("link", ""),
                "duration": duration,
                "width": int(rend.get("width") or v.get("width") or 0),
                "height": int(rend.get("height") or v.get("height") or 0),
                "creator": user.get("name", ""),
                "thumbnail": v.get("image", ""),
                "tags": _slug_tags_from_url(v.get("url", "") or ""),
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
        output_path.parent.mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _is_modern_tourist_content(clip_metadata: dict[str, Any]) -> bool:
        title = (clip_metadata.get("url", "") or "").lower()
        tags = (clip_metadata.get("tags", "") or "").lower()
        user = (clip_metadata.get("user", {}) or {}).get("name", "").lower()
        photographer = (clip_metadata.get("photographer", "") or "").lower()
        text_blob = f"{title} {tags} {user} {photographer}"

        for banned in _MODERN_TOURIST_BLACKLIST:
            if banned in text_blob:
                return False

        if any(word in text_blob for word in ["selfie stick", "travel blog", "vacation"]):
            return False

        return True


def _pick_video_rendition(
    video_files: list[dict],
    min_width: int = 0,
    max_width: int = 1920,
) -> dict | None:
    candidates = [
        f for f in video_files
        if (f.get("file_type", "") or "").startswith("video/")
        and min_width <= int(f.get("width") or 0) <= max_width
        and f.get("link")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda f: int(f.get("width") or 0), reverse=True)
    return candidates[0]


def _slug_tags_from_url(url: str) -> str:
    if not url:
        return ""
    tail = url.rstrip("/").rsplit("/", 1)
    if len(tail) != 2:
        return ""
    slug = tail[1]
    parts = slug.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        slug = parts[0]
    return slug.replace("-", " ").strip()


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
