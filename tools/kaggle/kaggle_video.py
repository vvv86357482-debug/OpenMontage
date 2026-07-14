"""Video generation via Kaggle GPU kernel.

Placeholder for future video generation support. Currently delegates to
image generation or fails with a clear message.
"""

from __future__ import annotations

import json
import os
import tempfile
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


class KaggleVideo(BaseTool):
    name = "kaggle_video"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "video_generation"
    provider = "kaggle"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = ["cmd:kaggle"]
    install_instructions = (
        "Install the Kaggle CLI: pip install kaggle\n"
        "Set KAGGLE_USERNAME and KAGGLE_KEY in your environment."
    )
    fallback_tools = ["ltx_video_local", "comfyui_video"]
    agent_skills = []

    capabilities = ["generate_video", "batch_generation"]
    supports = {
        "offline": False,
        "video_output": True,
    }
    best_for = ["GPU-accelerated video generation via Kaggle kernels"]
    not_good_for = ["local CPU-only environments"]

    input_schema = {
        "type": "object",
        "required": ["prompt", "output_path"],
        "properties": {
            "prompt": {"type": "string"},
            "output_path": {"type": "string"},
            "duration_seconds": {"type": "number", "default": 5.0},
            "fps": {"type": "integer", "default": 24},
            "width": {"type": "integer", "default": 1024},
            "height": {"type": "integer", "default": 576},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=2048, vram_mb=0, disk_mb=10000, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["kernels_timeout"])
    idempotency_key_fields = ["prompt", "output_path", "duration_seconds"]
    side_effects = ["pushes Kaggle kernel", "downloads video output"]
    user_visible_verification = ["Verify output video exists and has correct duration"]

    def get_status(self) -> ToolStatus:
        import shutil as _shutil
        return ToolStatus.AVAILABLE if _shutil.which("kaggle") else ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=False,
            error=(
                "Kaggle video generation is not yet implemented. "
                "Use ltx_video_local, comfyui_video, or other video generation "
                "tools for now."
            ),
        )
