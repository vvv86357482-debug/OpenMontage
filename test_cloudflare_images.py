#!/usr/bin/env python3
"""Test Cloudflare Workers AI image models through all credential pairs.

Credentials (only these env names):
  CLOUDFLARE_TOKEN_1..10, CLOUDFLARE_ACCOUNT_ID_1..10

Protocol:
1. Probe every pair for real Workers AI access and actual daily Neuron usage.
   Tokens sharing one account ID = ONE quota (deduplicated by account tag).
2. For every WORKING distinct account: one 1024x1024 test generation through
   an available image model (cheapest-first ladder), measuring wall time,
   HTTP status, reported/estimated Neurons; PNG is saved; errors/429 logged.

No token or account ID value is ever printed. Results land in
benchmarks/cloudflare_workers_ai_2026_08/.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

from cloudflare_image_provider import MODELS, CloudflareImageProvider  # noqa: E402

OUT_DIR = os.path.join("benchmarks", "cloudflare_workers_ai_2026_08")
IMAGES_DIR = os.path.join(OUT_DIR, "images")

# Cheapest-first smoke ladder: confirm each working account with minimal spend.
SMOKE_LADDER = [
    "@cf/black-forest-labs/flux-2-klein-4b",        # ~104 neurons @1024²
    "@cf/black-forest-labs/flux-1-schnell",         # ~58 neurons @1024²
    "@cf/bytedance/stable-diffusion-xl-lightning",  # price not published
]

PROMPT = (
    "Documentary-style photograph of an antique brass mechanical computer on a "
    "wooden desk, warm lamplight, dust motes in the air, shallow depth of field, "
    "highly detailed"
)


def main() -> int:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    provider = CloudflareImageProvider(pairs=10)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"== Stage A: probing {provider.total_credentials} credential pairs "
          f"({provider.distinct_accounts} distinct accounts) ==")
    report = provider.probe()
    for r in report:
        extra = ""
        if isinstance(r.get("image_models"), list) and r["image_models"]:
            extra += f" | {len(r['image_models'])} image models"
        q = r.get("neurons_today")
        if isinstance(q, dict) and "used" in q:
            extra += f" | neurons today: {q['used']}/10000 free"
        elif isinstance(q, dict) and q.get("error"):
            extra += f" | quota check: {q['error']}"
        print(f"  ACCOUNT_ID_{r['pair']} -> {r['status']}{extra}")
    print(f"\n  Quota groups:\n    {provider.status_line()}")

    results: dict = {
        "timestamp_utc": stamp,
        "credentials_total": provider.total_credentials,
        "distinct_accounts": provider.distinct_accounts,
        "probe": report,
        "generations": [],
    }

    working = provider.probe and provider.healthy_groups()
    if not working:
        print("\n== Stage B skipped: no working account ==")
    else:
        print(f"\n== Stage B: one 1024x1024 generation per working account "
              f"({len(working)} accounts, cheapest-model ladder) ==")
        for g in working:
            got = None
            for mid in SMOKE_LADDER:
                out = os.path.join(
                    IMAGES_DIR, f"group{g.label}_{mid.split('/')[-1].replace('.', '_')}.png")
                res = provider.generate(mid, PROMPT, width=1024, height=1024,
                                        output_path=out, restrict_group=g.tag)
                row = {
                    "account_group": res.account_group,
                    "token_pair": res.account_idx if res.success else g.label,
                    "model_id": res.model_id,
                    "success": res.success,
                    "http_status": res.http_status,
                    "time_s": res.elapsed_s,
                    "neurons_reported": res.neurons_reported,
                    "neurons_estimated": res.neurons_estimated,
                    "cost_usd_estimated": res.cost_usd_estimated,
                    "output_path": res.output_path if res.success else None,
                    "error": "" if res.success else res.error,
                    "attempts": res.attempts,
                }
                if res.success:
                    got = row
                    break
                # model may be unavailable on this account: try next in ladder
                print(f"     [{g.label}] {MODELS[mid]['label']}: {row['error'][:110]}")
            if got:
                results["generations"].append(got)
                n = got["neurons_reported"] or got["neurons_estimated"]
                print(f"     [{got['account_group']}] OK model={got['model_id']} "
                      f"http={got['http_status']} {got['time_s']}s ~{n} neurons")
            else:
                results["generations"].append(row)

    out_json = os.path.join(OUT_DIR, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    ok = sum(1 for x in results["generations"] if x.get("success"))
    total_neurons = sum(
        x.get("neurons_reported") or x.get("neurons_estimated") or 0
        for x in results["generations"])
    print(f"\n== Summary: working accounts confirmed by generation: {ok}; "
          f"~{total_neurons:.1f} neurons spent total ==")
    print(f"Results JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
