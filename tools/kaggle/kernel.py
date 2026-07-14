"""Kaggle kernel lifecycle helpers.

Wraps the Kaggle API for:
- kernel push (submit notebook as kernel)
- kernel status polling
- output download
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.base_tool import BaseTool


class KaggleKernel(BaseTool):
    name = "kaggle_kernel"
    version = "0.1.0"
    tier = BaseTool.tier
    capability = "corpus_population"
    provider = "kaggle"
    stability = BaseTool.stability
    execution_mode = BaseTool.execution_mode
    determinism = BaseTool.determinism
    runtime = BaseTool.runtime

    dependencies = ["cmd:kaggle"]
    install_instructions = (
        "Install the Kaggle CLI: pip install kaggle\n"
        "Set KAGGLE_USERNAME and KAGGLE_KEY in your environment.\n"
        "Get your key from https://www.kaggle.com/settings"
    )

    def push_kernel(self, notebook_path: Path, kernel_slug: str, title: str) -> dict[str, Any]:
        """Push a notebook as a Kaggle kernel."""
        if not shutil.which("kaggle"):
            raise RuntimeError("Kaggle CLI not found. Install with: pip install kaggle")
        
        result = subprocess.run(
            [
                "kaggle", "kernels", "push",
                "-p", str(notebook_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle kernel push failed: {result.stderr}")
        
        return json.loads(result.stdout) if result.stdout.strip() else {"status": "pushed"}

    def poll_kernel(self, kernel_slug: str, timeout_seconds: int = 1800) -> dict[str, Any]:
        """Poll kernel status until completion or timeout."""
        start = time.time()
        last_status = None
        
        while time.time() - start < timeout_seconds:
            result = subprocess.run(
                [
                    "kaggle", "kernels", "status",
                    kernel_slug,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.returncode == 0:
                try:
                    status_data = json.loads(result.stdout)
                    last_status = status_data.get("status", "").lower()
                    if last_status in ("complete", "error", "failed"):
                        return status_data
                except json.JSONDecodeError:
                    pass
            
            time.sleep(30)
        
        return {"status": "timeout", "error": f"Kernel {kernel_slug} did not complete within {timeout_seconds}s"}

    def download_outputs(self, kernel_slug: str, output_dir: Path) -> Path:
        """Download kernel outputs to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "kaggle", "kernels", "output",
                kernel_slug,
                "-p", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")
        
        return output_dir
