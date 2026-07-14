#!/usr/bin/env python3
"""Poll the v2 final-voice measurement Kaggle kernel and pull output."""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

SLUG = "forts845/openmontage-tts-measure-v2-final-voices"
OUT = Path("/workspaces/OpenMontage/tools/kaggle/tts_measure_v2/result")
OUT.mkdir(parents=True, exist_ok=True)
DONE = OUT / "POLL_DONE.txt"
LOG = OUT / "poll.log"
DEADLINE = time.time() + 90 * 60


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def main() -> int:
    with LOG.open("a") as log:
        log.write(f"[{time.strftime('%H:%M:%S')}] monitor start\n"); log.flush()
        while time.time() < DEADLINE:
            out = run(["kaggle", "kernels", "status", SLUG]).stdout.strip()
            log.write(f"[{time.strftime('%H:%M:%S')}] {out}\n"); log.flush()
            low = out.lower()
            if "complete" in low or "error" in low or "failed" in low:
                log.write("terminal -> pulling\n"); log.flush()
                pr = run(["kaggle", "kernels", "output", SLUG, "-p", str(OUT)])
                log.write(f"pull rc={pr.returncode}\n{pr.stdout}\n{pr.stderr}\n"); log.flush()
                DONE.write_text(out)
                return 0
            time.sleep(60)
        log.write("DEADLINE\n"); return 2


if __name__ == "__main__":
    raise SystemExit(main())
