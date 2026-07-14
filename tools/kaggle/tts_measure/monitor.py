#!/usr/bin/env python3
"""Poll a Kaggle kernel until complete, then pull its output locally."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SLUG = "forts845/openmontage-chatterbox-270-scene-tts-measurement"
OUT_DIR = Path("/workspaces/OpenMontage/tools/kaggle/tts_measure/result")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DONE = OUT_DIR / "POLL_DONE.txt"
LOG = OUT_DIR / "poll.log"

DEADLINE = time.time() + 80 * 60  # 80 min cap


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def status() -> str:
    r = run(["kaggle", "kernels", "status", SLUG])
    return r.stdout.strip() + ("\nERR:" + r.stderr.strip() if r.returncode else "")


def main() -> int:
    with LOG.open("a") as log:
        log.write(f"[{time.strftime('%H:%M:%S')}] monitor start, deadline 80m\n")
        while time.time() < DEADLINE:
            out = status()
            log.write(f"[{time.strftime('%H:%M:%S')}] {out}\n")
            log.flush()
            low = out.lower()
            if "complete" in low or "error" in low or "failed" in low:
                log.write(f"[{time.strftime('%H:%M:%S')}] terminal state -> pulling output\n")
                log.flush()
                pr = run(["kaggle", "kernels", "output", SLUG, "-p", str(OUT_DIR)])
                log.write(f"[{time.strftime('%H:%M:%S')}] pull rc={pr.returncode}\n{pr.stdout}\n{pr.stderr}\n")
                DONE.write_text(out)
                return 0
            time.sleep(60)
        log.write("DEADLINE reached without terminal state\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
