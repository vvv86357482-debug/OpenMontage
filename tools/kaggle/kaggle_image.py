"""Image generation via Kaggle GPU kernel.

Submits a single kernel push per project. The kernel runs SANA or FLUX.1
depending on the detected GPU, generates all requested images, and writes
a manifest.json to /kaggle/working/outputs/.

This tool never falls back to placeholders. If the kernel reports failure,
the tool returns a failed ToolResult.
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


class KaggleImage(BaseTool):
    name = "kaggle_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "kaggle"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.SEEDED
    runtime = ToolRuntime.LOCAL_GPU

    dependencies = ["cmd:kaggle"]
    install_instructions = (
        "Install the Kaggle CLI: pip install kaggle\n"
        "Set KAGGLE_USERNAME and KAGGLE_KEY in your environment.\n"
        "Get your key from https://www.kaggle.com/settings"
    )
    fallback_tools = ["flux_image", "dashscope_image"]
    agent_skills = ["flux-best-practices"]

    capabilities = [
        "generate_image",
        "batch_generation",
        "offline_via_kaggle",
    ]
    supports = {
        "negative_prompt": True,
        "seed": True,
        "custom_size": True,
        "offline": True,
    }
    best_for = [
        "free GPU image generation",
        "batch scene assets",
        "no API keys required beyond Kaggle",
    ]
    not_good_for = [
        "real-time generation",
        "single-image quick tests (kernel overhead)",
    ]

    input_schema = {
        "type": "object",
        "required": ["assets"],
        "properties": {
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "prompt"],
                    "properties": {
                        "id": {"type": "string"},
                        "prompt": {"type": "string"},
                        "width": {"type": "integer", "default": 1024},
                        "height": {"type": "integer", "default": 576},
                        "negative_prompt": {"type": "string", "default": ""},
                        "seed": {"type": "integer"},
                    },
                },
                "description": "List of image assets to generate in one kernel run",
            },
            "project_id": {"type": "string", "description": "Unique project identifier"},
            "channel": {"type": "string", "description": "Channel style for prompt context"},
            "kernel_slug": {"type": "string", "description": "Kaggle kernel slug (user/notebook-name)"},
            "title": {"type": "string", "description": "Human-readable kernel title"},
            "timeout_seconds": {"type": "integer", "default": 1800},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=2, ram_mb=2048, vram_mb=0, disk_mb=5000, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=1, retryable_errors=["kernels_timeout"])
    idempotency_key_fields = ["project_id", "kernel_slug", "assets"]
    side_effects = ["pushes Kaggle kernel", "downloads outputs to output_path"]
    user_visible_verification = [
        "Verify manifest.json in output_path shows all assets generated",
        "Check each asset file exists and is > 1KB",
    ]

    def get_status(self) -> ToolStatus:
        if shutil.which("kaggle"):
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0  # Free Kaggle GPU quota

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        import shutil as _shutil

        kaggle_cli = _shutil.which("kaggle")
        if not kaggle_cli:
            return ToolResult(
                success=False,
                error="Kaggle CLI not found. Install with: pip install kaggle",
            )

        assets = inputs.get("assets", [])
        if not assets:
            return ToolResult(success=False, error="No assets provided")

        project_id = inputs.get("project_id", "default")
        kernel_slug = inputs.get("kernel_slug")
        title = inputs.get("title", f"openmontage-{project_id}")
        timeout = inputs.get("timeout_seconds", 1800)
        output_path = Path(inputs.get("output_path", f"projects/{project_id}/assets/images"))

        # Build notebook path
        notebook_path = Path(__file__).resolve().parent / "kernel" / "kernel.ipynb"
        if not notebook_path.exists():
            return ToolResult(success=False, error=f"Kernel notebook not found: {notebook_path}")

        # Embed JOB_TASKS into the notebook before push
        job_tasks = {
            "project_id": project_id,
            "channel": inputs.get("channel", "unknown"),
            "assets": [
                {
                    "id": a["id"],
                    "type": "image",
                    "prompt": a["prompt"],
                    "width": a.get("width", 1024),
                    "height": a.get("height", 576),
                    "negative_prompt": a.get("negative_prompt", ""),
                    "seed": a.get("seed"),
                }
                for a in assets
            ],
        }

        try:
            notebook_data = json.loads(notebook_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return ToolResult(success=False, error=f"Failed to read kernel notebook: {exc}")

        # Find and update the JOB_TASKS cell
        job_tasks_updated = False
        for cell in notebook_data.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                if "JOB_TASKS = {" in source and "PLACEHOLDER" in source:
                    cell["source"] = [
                        f"JOB_TASKS = {json.dumps(job_tasks, indent=4)}\n"
                    ]
                    job_tasks_updated = True
                    break

        if not job_tasks_updated:
            return ToolResult(
                success=False,
                error="Could not find JOB_TASKS placeholder cell in kernel notebook",
            )

        # Write modified notebook to temp
        tmp_dir = Path(tempfile.mkdtemp(prefix="kaggle_"))
        modified_notebook = tmp_dir / f"{project_id}_kernel.ipynb"
        modified_notebook.write_text(
            json.dumps(notebook_data, indent=2), encoding="utf-8"
        )

        start = time.time()
        try:
            # Push kernel
            push_result = self._run_kaggle([
                "kaggle", "kernels", "push",
                "-p", str(modified_notebook),
            ])
            print(f"Kernel push result: {push_result}")

            # Extract kernel slug if not provided
            if not kernel_slug:
                kernel_slug = self._extract_kernel_slug(push_result, project_id)

            # Wait for completion
            status = self._poll_kernel(kernel_slug, timeout)
            if status.get("status", "").lower() not in ("complete", "success"):
                return ToolResult(
                    success=False,
                    error=f"Kernel {kernel_slug} did not complete successfully: {status}",
                )

            # Download outputs
            output_path.mkdir(parents=True, exist_ok=True)
            download_dir = tmp_dir / "downloads"
            download_dir.mkdir(exist_ok=True)
            self._run_kaggle([
                "kaggle", "kernels", "output",
                kernel_slug,
                "-p", str(download_dir),
            ])

            # Move outputs to canonical location using shutil.move (cross-filesystem safe)
            for src in download_dir.rglob("*"):
                if src.is_file():
                    rel = src.relative_to(download_dir)
                    dst = output_path / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))

            # Verify manifest
            manifest_path = output_path / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                generated = [
                    a for a in manifest.get("assets", [])
                    if a.get("success") and Path(a.get("path", "")).exists()
                ]
                failed = [
                    a for a in manifest.get("assets", [])
                    if not a.get("success")
                ]
                
                if failed:
                    return ToolResult(
                        success=False,
                        error=(
                            f"Kernel reported failures: {failed}. "
                            "No silent placeholder substitution performed."
                        ),
                    )
                
                return ToolResult(
                    success=True,
                    data={
                        "project_id": project_id,
                        "kernel_slug": kernel_slug,
                        "generated_count": len(generated),
                        "total_count": len(manifest.get("assets", [])),
                        "output_path": str(output_path),
                        "manifest": manifest,
                    },
                    artifacts=[str(output_path)],
                    duration_seconds=round(time.time() - start, 2),
                )
            else:
                return ToolResult(
                    success=False,
                    error=(
                        "Kernel completed but manifest.json not found in outputs. "
                        "Hard-fail: cannot verify asset generation."
                    ),
                )

        except Exception as exc:
            return ToolResult(success=False, error=f"Kaggle image generation failed: {exc}")

    def _run_kaggle(self, cmd: list[str]) -> str:
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle CLI error: {result.stderr}")
        return result.stdout.strip()

    def _extract_kernel_slug(self, push_output: str, project_id: str) -> str:
        try:
            data = json.loads(push_output)
            return data.get("slug", f"default/{project_id}_kernel")
        except Exception:
            return f"default/{project_id}_kernel"

    def _poll_kernel(self, kernel_slug: str, timeout: int) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            try:
                output = self._run_kaggle([
                    "kaggle", "kernels", "status", kernel_slug
                ])
                data = json.loads(output)
                status = data.get("status", "").lower()
                if status in ("complete", "error", "failed"):
                    return data
            except Exception:
                pass
            time.sleep(30)
        return {"status": "timeout", "error": f"Timeout after {timeout}s"}
