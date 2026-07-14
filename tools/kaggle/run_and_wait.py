"""Run a Kaggle kernel and wait for completion using the official Kaggle Python API."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from kaggle import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import KernelWorkerStatus

KERNEL_FOLDER = Path(__file__).resolve().parent
OUTPUT_DIR = Path("/tmp/kaggle_real_output")
POLL_INTERVAL_S = 30
TIMEOUT_S = 1800

def main() -> int:
    api = KaggleApi()
    api.authenticate()
    print("Authenticated successfully")

    print(f"Pushing kernel from folder: {KERNEL_FOLDER}")
    push_result = api.kernels_push(str(KERNEL_FOLDER))

    slug = push_result.ref if hasattr(push_result, 'ref') else None
    if slug and slug.startswith("/code/"):
        slug = slug[len("/code/"):]
    print(f"Push result: ref={getattr(push_result, 'ref', None)} "
          f"version={getattr(push_result, 'versionNumber', None)} "
          f"error={getattr(push_result, 'error', None)}")
    print(f"Kernel slug: {slug}")

    if getattr(push_result, "error", None):
        print("Push reported an error; aborting.")
        return 1

    print("Polling status...")
    start = time.time()
    final_status = None
    final_status_obj = None
    while time.time() - start < TIMEOUT_S:
        status_obj = api.kernels_status(slug)
        status_enum = status_obj.status
        st = status_enum.name.lower() if isinstance(status_enum, KernelWorkerStatus) else str(status_enum).lower()
        print(f"Status: {st}")
        final_status = st
        final_status_obj = status_obj
        if st in ("complete", "success"):
            break
        if st in ("error", "failed", "cancelled"):
            print("Kernel failed/cancelled.")
            break
        time.sleep(POLL_INTERVAL_S)
    else:
        print("Timed out waiting for kernel completion.")
        return 1

    if final_status not in ("complete", "success"):
        # Try to get failure message
        if final_status_obj and hasattr(final_status_obj, '_failure_message'):
            failure_msg = final_status_obj._failure_message
            print(f"Kernel did not complete successfully (status={final_status}). Failure message: {failure_msg}")
        else:
            print(f"Kernel did not complete successfully (status={final_status}).")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading outputs to {OUTPUT_DIR}")
    api.kernels_output(slug, str(OUTPUT_DIR))

    print("\n=== OUTPUT FILES ===")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            print(f"{f.relative_to(OUTPUT_DIR)}: {f.stat().st_size} bytes")

    manifest = OUTPUT_DIR / "manifest.json"
    print("\n=== MANIFEST.JSON (verbatim) ===")
    if manifest.exists():
        print(manifest.read_text())
    else:
        print("NOT FOUND")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())