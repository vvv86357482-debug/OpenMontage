# Cloudflare Workers AI — Image Generation Test Results

Date: 2026-08-25 · Harness: `test_cloudflare_images.py` · Provider: `tools/cloudflare_image_provider.py`
Credentials: `CLOUDFLARE_TOKEN_1..10` + `CLOUDFLARE_ACCOUNT_ID_1..10` (renamed set, re-audited same day).
Docs verified live before testing: model catalog (updated 2026-08-12) and pricing (updated 2026-08-18).

## STATUS: BLOCKED — 0 of 10 credential pairs can access Workers AI

No image was generated and **zero Neurons were spent**. The harness correctly
skipped Stage B instead of burning quota against invalid credentials.

Second-pass audit of the RENAMED credentials (`CLOUDFLARE_TOKEN_N`):

- All 10 `CLOUDFLARE_ACCOUNT_ID_N` are now well-formed 32-hex tags and all
  distinct → 10 real quota groups (dedup by account tag works as designed).
- All 10 tokens: uniform length 53, clean charset `[A-Za-z0-9_-]`, no stray
  quotes/whitespace, no useful base64/hex decoding.
- `GET /user/tokens/verify` → HTTP 401 (all 10).
- Account-owned path `POST /accounts/{id}/tokens/verify` → HTTP 401, error
  code 1000 "Authentication error" (all 10; route valid this time).
- Cross-matrix key×account from the previous set also empty; current set uses
  per-pair probing with group dedup.
- Conclusion: the token VALUES themselves are rejected by Cloudflare. Uniform
  53-char shape suggests they are not raw Cloudflare API tokens (those are
  typically ~40 chars) — possibly wrapped/proxied secrets from another system.

## Credential audit (all 10 pairs)

| Pair | Token verify (`/user/tokens/verify`) | Account ID format (32-hex) | Verdict |
|------|--------------------------------------|----------------------------|---------|
| 01 | HTTP 401 Authentication error | malformed (30 chars) | unusable |
| 02 | HTTP 401 Authentication error | malformed (31 chars) | unusable |
| 03 | HTTP 401 Authentication error | malformed (31 chars) | unusable |
| 04 | HTTP 401 Authentication error | malformed (31 chars, non-hex char) | unusable |
| 05 | HTTP 401 Authentication error | malformed (31 chars) | unusable |
| 06 | HTTP 401 Authentication error | valid 32-hex | token rejected by Cloudflare |
| 07 | HTTP 401 Authentication error | malformed (33 chars) | unusable |
| 08 | HTTP 401 Authentication error | malformed (31 chars) | unusable |
| 09 | HTTP 401 Authentication error | malformed (30 chars) | unusable |
| 10 | HTTP 401 Authentication error | malformed (31 chars) | unusable |

Per-account re-validation (2026-08-25, second pass):

```
ACCOUNT_ID_01 -> INVALID -> LOCAL    id length 30 != 32
ACCOUNT_ID_02 -> INVALID -> LOCAL    id length 31 != 32
ACCOUNT_ID_03 -> INVALID -> LOCAL    id length 31 != 32
ACCOUNT_ID_04 -> INVALID -> LOCAL    id length 31 != 32
ACCOUNT_ID_05 -> INVALID -> LOCAL    id length 31 != 32
ACCOUNT_ID_06 -> INVALID -> HTTP 401 token rejected by Cloudflare (code 1000)
ACCOUNT_ID_07 -> INVALID -> LOCAL    id length 33 != 32
ACCOUNT_ID_08 -> INVALID -> LOCAL    id length 31 != 32
ACCOUNT_ID_09 -> INVALID -> LOCAL    id length 30 != 32
ACCOUNT_ID_10 -> INVALID -> LOCAL    id length 31 != 32
```

Additional checks performed before declaring failure:

- Base64 / URL-safe-base64 / hex decode hypotheses for every key → still 401.
- Account-owned token path (`/accounts/{id}/tokens/verify`) checked for pair 06 → still rejected.
- Cross-matrix (every key × every valid-format account ID, 10 requests) → no working combination; a shifted key↔account pairing is ruled out.
- Only pair 06 has a syntactically valid Cloudflare account tag; its key is rejected.
- Quota (GraphQL `aiInferenceAdaptiveGroups`, neurons today) could not be read for
  any pair — requires a working token. Actual per-account quota is therefore unknown.
- Conclusion: the provided secrets are not valid Cloudflare API tokens as-is
  (truncated, rotated, or from a different system).

## Generation results (requested table)

| Model | Success | Time | Neurons | Cost | Quality | Artifacts |
|-------|---------|------|---------|------|---------|-----------|
| FLUX.2 Klein 4B (`@cf/black-forest-labs/flux-2-klein-4b`) | FAIL — no valid credentials | — | — | — | not assessable | none |
| FLUX.1 Schnell (`@cf/black-forest-labs/flux-1-schnell`) | FAIL — no valid credentials | — | — | — | not assessable | none |
| FLUX.2 Dev (`@cf/black-forest-labs/flux-2-dev`) | FAIL — no valid credentials | — | — | — | not assessable | none |
| Leonardo Lucid Origin (`@cf/leonardo/lucid-origin`) | FAIL — no valid credentials | — | — | — | not assessable | none |
| Leonardo Phoenix 1.0 (`@cf/leonardo/phoenix-1.0`) | FAIL — no valid credentials | — | — | — | not assessable | none |
| SDXL-Lightning (`@cf/bytedance/stable-diffusion-xl-lightning`) | FAIL — no valid credentials | — | — | — | not assessable | none |

Machine-readable log: `benchmarks/cloudflare_workers_ai_2026_08/results.json`.

## Model IDs & pricing verified from official docs (2026-08-25)

All six models exist in the current catalog. Pricing ($0.011 / 1000 Neurons;
10,000 free Neurons/day per account):

| Model | ID | Neuron price (from docs) | Est. cost @1024×1024 |
|-------|----|--------------------------|----------------------|
| FLUX.2 Klein 4B | `@cf/black-forest-labs/flux-2-klein-4b` | 5.37 / input tile + 26.05 / output tile (512²) | ~104 N ≈ $0.00115 |
| FLUX.1 Schnell | `@cf/black-forest-labs/flux-1-schnell` | 4.80 / tile + 9.60 / step | ~58 N ≈ $0.00063 (4 steps) |
| FLUX.2 Dev | `@cf/black-forest-labs/flux-2-dev` | 18.75 / in-tile·step + 37.50 / out-tile·step | ~600 N ≈ $0.0066 (4 steps) |
| Leonardo Lucid Origin | `@cf/leonardo/lucid-origin` | 636 / tile + 12 / step | ~2688 N ≈ $0.0296 |
| Leonardo Phoenix 1.0 | `@cf/leonardo/phoenix-1.0` | 530 / tile + 10 / step | ~2240 N ≈ $0.0246 |
| SDXL-Lightning | `@cf/bytedance/stable-diffusion-xl-lightning` | **not in current pricing table** (beta legacy) | unknown |

Notes:

- Free tier = 10,000 Neurons/day/account. A single Lucid Origin 1024² render
  consumes ≈27% of a free daily allocation — "10 keys" would NOT mean 10× Lucid
  quota even if all were valid.
- SDXL-Lightning is absent from the Aug 2026 pricing page while still listed as
  beta in the catalog; treat its availability/billing as unstable.
- Estimates are marked as such in the provider; reported usage from the API
  response takes precedence when present.

## What was built (ready to run once credentials are fixed)

- `tools/cloudflare_image_provider.py` — provider with:
  - round-robin across healthy accounts (rotation starts after last success);
  - automatic failover: 429/quota → 90 s cooldown + next account; 401/403 →
    account disabled for session; 5xx/model-missing → next account;
  - local pre-validation (account tag must be 32-hex) so malformed creds fail
    fast without network calls;
  - secret hygiene: keys/IDs never logged or persisted; API error strings
    sanitized (hex runs redacted); probe output shows pair index only.
- `test_cloudflare_images.py` — Stage A probes all pairs incl. actual daily
  Neuron usage via GraphQL Analytics; Stage B runs one 1024×1024 generation
  per model with timing, neuron accounting, PNG saving, error capture.
- SANA-Sprint (`tools/kaggle/sana_sprint/`) and the BFL FLUX.2 Klein pipeline
  were not touched.

## NEXT (requires user action)

1. Re-issue the tokens as **Cloudflare API Tokens** created in
   dash.cloudflare.com → My Profile → API Tokens (or Account → API Tokens),
   template/permission: **Account › Workers AI › Edit**. Copy the value
   exactly as shown by Cloudflare — no wrapping, no renaming.
2. Keep the new `CLOUDFLARE_TOKEN_N` / `CLOUDFLARE_ACCOUNT_ID_N` names; the
   provider reads only these.
3. Re-run: `python3 test_cloudflare_images.py` — it re-probes, dedupes quota
   groups, then runs one 1024×1024 smoke generation per WORKING account via a
   cheapest-first model ladder and reports HTTP status / time / Neurons.
