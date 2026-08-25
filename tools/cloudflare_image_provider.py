"""Cloudflare Workers AI image generation provider with multi-account failover.

Security contract (hard requirements):
- Tokens and account IDs are NEVER logged, printed, or written to disk.
- Error strings from the Cloudflare API are sanitized before storage/logging
  (long hex runs that could be account IDs are replaced).
- Nothing in this module writes secrets anywhere.

Credential source (only these names):
  CLOUDFLARE_TOKEN_1..CLOUDFLARE_TOKEN_10
  CLOUDFLARE_ACCOUNT_ID_1..CLOUDFLARE_ACCOUNT_ID_10

Quota model:
- Credentials sharing the SAME account ID form one QUOTA GROUP. Multiple
  tokens of one account are backups inside that group, never extra quota.
- Round-robin rotates across DISTINCT accounts (groups). A 429/quota error
  puts the whole group on cooldown; an auth failure disables only that token.

Model IDs verified against developers.cloudflare.com/workers-ai on 2026-08-25.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

BASE_URL = "https://api.cloudflare.com/client/v4"

# --- Model registry (verified against official docs 2026-08-25) -------------
# Pricing source: https://developers.cloudflare.com/workers-ai/platform/pricing/
# (page last updated 2026-08-18). Prices: $0.011 per 1000 Neurons.
NEURONS_PER_USD = 1000.0 / 0.011

TILE = 512


def _tiles(w: int, h: int) -> int:
    return max(1, math.ceil(w / TILE)) * max(1, math.ceil(h / TILE))


MODELS: dict[str, dict[str, Any]] = {
    "@cf/black-forest-labs/flux-2-klein-4b": {
        "label": "FLUX.2 Klein 4B",
        # 5.37 neurons per input 512x512 tile + 26.05 per output tile.
        "estimate": lambda w, h, steps: _tiles(w, h) * 26.05,
        "default_steps": None,
    },
    "@cf/black-forest-labs/flux-1-schnell": {
        "label": "FLUX.1 Schnell",
        # 4.80 neurons per 512x512 tile + 9.60 per step.
        "estimate": lambda w, h, steps: _tiles(w, h) * 4.80 + steps * 9.60,
        "default_steps": 4,
    },
    "@cf/black-forest-labs/flux-2-dev": {
        "label": "FLUX.2 Dev",
        # 37.50 neurons per output tile per step (t2i has no input tiles).
        "estimate": lambda w, h, steps: _tiles(w, h) * 37.50 * steps,
        "default_steps": 4,
    },
    "@cf/leonardo/lucid-origin": {
        "label": "Leonardo Lucid Origin",
        # 636 neurons per 512x512 tile + 12 per step.
        "estimate": lambda w, h, steps: _tiles(w, h) * 636.0 + steps * 12.0,
        "default_steps": 12,
    },
    "@cf/leonardo/phoenix-1.0": {
        "label": "Leonardo Phoenix 1.0",
        # 530 neurons per 512x512 tile + 10 per step.
        "estimate": lambda w, h, steps: _tiles(w, h) * 530.0 + steps * 10.0,
        "default_steps": 12,
    },
    "@cf/bytedance/stable-diffusion-xl-lightning": {
        "label": "SDXL-Lightning",
        # Present in the model catalog (beta) but absent from the current
        # pricing table; neuron cost unknown -> estimated as None.
        "estimate": lambda w, h, steps: None,
        "default_steps": 4,
    },
}

HEX_RUN = re.compile(r"[0-9a-fA-F]{16,}")
ACCT_TAG = re.compile(r"^[0-9a-f]{32}$", re.I)


def sanitize(text: str) -> str:
    """Remove anything resembling an account ID / key fragment from text."""
    return HEX_RUN.sub("<redacted>", str(text))[:300]


def group_tag(account_id: str) -> str:
    """Stable non-reversible tag identifying one Cloudflare account/quota."""
    return hashlib.sha256(account_id.encode()).hexdigest()[:8]


@dataclass
class AccountState:
    idx: str  # credential index ("7"), the ONLY thing we log
    token: str = field(repr=False)
    account_id: str = field(repr=False)
    group: str = ""          # group_tag of the owning account
    healthy: bool = True     # token-level health (auth)
    reason: str = ""
    cooldown_until: float = 0.0
    requests_ok: int = 0
    requests_failed: int = 0
    neurons_spent: float = 0.0

    def token_available(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return self.healthy and self.cooldown_until <= now


@dataclass
class QuotaGroup:
    """One Cloudflare account (= one daily Neuron quota)."""
    tag: str                       # group_tag, safe to log
    label: str                     # first credential index seen, e.g. "3"
    members: list[AccountState] = field(default_factory=list)
    cooldown_until: float = 0.0    # quota/rate-limit state shared by all tokens
    requests_ok: int = 0
    requests_failed: int = 0
    neurons_spent: float = 0.0

    def available(self, now: float) -> bool:
        return self.cooldown_until <= now and any(
            m.token_available(now) for m in self.members
        )

    def note(self, now: float | None = None) -> str:
        now = now if now is not None else time.time()
        live = sum(1 for m in self.members if m.token_available(now))
        if not any(m.healthy for m in self.members):
            return f"group {self.label}#{self.tag}: all {len(self.members)} token(s) failed auth"
        if not self.available(now):
            until = max(0, int(self.cooldown_until - now))
            return f"group {self.label}#{self.tag}: quota/rate cooldown ({until}s left)"
        return f"group {self.label}#{self.tag}: {live}/{len(self.members)} token(s) usable"


@dataclass
class GenerationResult:
    success: bool
    model_id: str = ""
    label: str = ""
    account_idx: str = ""       # credential index only, never the value
    account_group: str = ""     # group tag + first-index label
    width: int = 0
    height: int = 0
    elapsed_s: float = 0.0
    http_status: Optional[int] = None
    neurons_reported: Optional[float] = None
    neurons_estimated: Optional[float] = None
    cost_usd_estimated: Optional[float] = None
    output_path: str = ""
    error: str = ""
    attempts: list = field(default_factory=list)


class CloudflareImageProvider:
    """Round-robin Workers AI image generation across distinct CF accounts."""

    QUOTA_COOLDOWN_S = 90.0
    REQUEST_TIMEOUT_S = 180.0

    def __init__(self, pairs: int = 10):
        entries: list[AccountState] = []
        for i in range(1, pairs + 1):
            token = os.environ.get(f"CLOUDFLARE_TOKEN_{i}", "").strip()
            acct = os.environ.get(f"CLOUDFLARE_ACCOUNT_ID_{i}", "").strip()
            if not token and not acct:
                continue
            ok_format = bool(token) and bool(ACCT_TAG.fullmatch(acct))
            entries.append(AccountState(
                idx=str(i), token=token, account_id=acct,
                group=group_tag(acct) if ACCT_TAG.fullmatch(acct) else "",
                healthy=ok_format,
                reason="" if ok_format else (
                    "missing credential" if (not token or not acct)
                    else "malformed account id (not 32-hex)"
                ),
            ))
        # Merge credentials that point at the SAME account into one group.
        self.groups: list[QuotaGroup] = []
        by_tag: dict[str, QuotaGroup] = {}
        malformed: list[AccountState] = []
        for e in entries:
            if not e.group:
                malformed.append(e)
                continue
            g = by_tag.get(e.group)
            if g is None:
                g = QuotaGroup(tag=e.group, label=e.idx)
                by_tag[e.group] = g
                self.groups.append(g)
            g.members.append(e)
        self.malformed = malformed
        self._rr = -1  # last successful group position

    # -- introspection -------------------------------------------------------

    @property
    def total_credentials(self) -> int:
        return sum(len(g.members) for g in self.groups) + len(self.malformed)

    @property
    def distinct_accounts(self) -> int:
        return len({g.tag for g in self.groups})

    def healthy_groups(self, now: float | None = None) -> list[QuotaGroup]:
        now = now if now is not None else time.time()
        return [g for g in self.groups if g.available(now)]

    def status_line(self) -> str:
        now = time.time()
        parts = []
        for g in self.groups:
            parts.append(g.note(now))
        for e in self.malformed:
            parts.append(f"credential #{e.idx}: skipped ({e.reason})")
        return "; ".join(parts)

    # -- probing -------------------------------------------------------------

    def probe(self) -> list[dict]:
        """Check every credential + group: access to Workers AI, daily usage."""
        report: list[dict] = []
        for e in self.malformed:
            report.append({"pair": e.idx, "status": f"INVALID_FORMAT ({e.reason})"})
        for g in self.groups:
            # token-level check on the first usable member proves AI access
            row: dict[str, Any] = {"pair": g.label, "group": g.tag,
                                   "tokens_in_group": len(g.members)}
            member = next((m for m in g.members if m.token_available()), None)
            if member is None:
                row["status"] = "no usable token in group"
                g.requests_failed += 1
                report.append(row)
                continue
            code, body = self._request(member, models_search=True)
            if code == 200 and body.get("success"):
                img_models = sorted(
                    m.get("name") for m in (body.get("result") or [])
                    if "text-to-image" in ((m.get("task") or {}).get("name", "")).lower()
                    or "image" in (m.get("output_modalities") or [])
                )
                row["status"] = "OK"
                row["image_models"] = img_models
                member.requests_ok += 1
                g.requests_ok += 1
                row["neurons_today"] = self._neurons_today(g, member)
            else:
                row["status"] = self._err(code, body)
                if code in (401, 403):
                    member.healthy = False
                    member.reason = "auth failed"
                g.requests_failed += 1
            report.append(row)
        return report

    def _neurons_today(self, g: QuotaGroup, member: AccountState) -> Any:
        gql = {
            "query": (
                "query($a:String!,$s:Time!){viewer{accounts(filter:{accountTag:$a}){"
                "aiInferenceAdaptiveGroups(limit:10000,filter:{datetime_geq:$s})"
                "{sum{neurons}}}}}"
            ),
            "variables": {
                "a": member.account_id,
                "s": time.strftime("%Y-%m-%dT00:00:00Z", time.gmtime()),
            },
        }
        code, body = self._request(member, payload=gql, graphql=True)
        if code != 200:
            return {"error": f"GQL_HTTP_{code}"}
        try:
            groups = body["data"]["viewer"]["accounts"][0]["aiInferenceAdaptiveGroups"]
            used = sum(x["sum"]["neurons"] for x in groups)
            return {"used": used, "free_daily_limit": 10000,
                    "remaining_estimate": max(0, 10000 - used)}
        except (KeyError, IndexError, TypeError):
            return {"error": "GQL_UNAVAILABLE"}

    # -- core HTTP -----------------------------------------------------------

    def _request(self, member: AccountState, payload: Optional[dict] = None,
                 graphql: bool = False, model_id: str = "",
                 models_search: bool = False):
        if graphql:
            url = f"{BASE_URL}/graphql"
        elif model_id:
            url = f"{BASE_URL}/accounts/{member.account_id}/ai/run/{model_id}"
        elif models_search:
            url = f"{BASE_URL}/accounts/{member.account_id}/ai/models/search?per_page=1000"
        else:
            raise ValueError("target required")
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, data=data,
                                     method="POST" if data is not None else "GET")
        req.add_header("Authorization", f"Bearer {member.token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT_S) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raw = b""
            try:
                raw = e.read()
            except Exception:
                pass
            try:
                return e.code, json.loads(raw.decode())
            except Exception:
                return e.code, {}
        except Exception as e:
            return 0, {"errors": [{"message": sanitize(str(e))}]}

    @staticmethod
    def _err(code: int, body: dict) -> str:
        msgs = "; ".join(
            e.get("message", "") for e in (body.get("errors") or [])
            if isinstance(e, dict)
        )
        return f"HTTP {code}: {sanitize(msgs)}" if msgs else f"HTTP {code}"

    # -- public API ----------------------------------------------------------

    def generate(
        self,
        model_id: str,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        guidance: Optional[float] = None,
        seed: Optional[int] = None,
        output_path: str = "",
        restrict_group: Optional[str] = None,
    ) -> GenerationResult:
        """Generate one image. Rotation covers distinct accounts; credentials
        sharing an account are tried as backups, never counted twice.

        restrict_group: group tag to force generation on ONE account only.
        """
        meta = MODELS.get(model_id)
        res = GenerationResult(model_id=model_id,
                               label=meta["label"] if meta else model_id,
                               width=width, height=height)
        if meta is None:
            res.error = f"unknown model {model_id}; known: {sorted(MODELS)}"
            return res

        payload: dict[str, Any] = {"prompt": prompt, "width": width, "height": height}
        eff_steps = steps if steps is not None else meta["default_steps"]
        if eff_steps is not None and any(
            k in model_id for k in ("schnell", "dev", "lucid", "phoenix", "lightning")
        ):
            payload["steps"] = eff_steps
        if guidance is not None and "leonardo" in model_id:
            payload["guidance"] = guidance
        if seed is not None:
            payload["seed"] = seed

        order = self._rotation(restrict_group)
        t0 = time.perf_counter()
        last_http: Optional[int] = None
        for g in order:
            now = time.time()
            if not g.available(now):
                continue
            group_done = False
            for member in [m for m in g.members if m.token_available(time.time())]:
                code, body = self._request(member, payload=payload, model_id=model_id)
                last_http = code or last_http
                entry = {"account_pair": member.idx, "group": g.label,
                         "http": code}
                if code == 200 and body.get("success") and body.get("result"):
                    ok, err = self._save_image(body, output_path)
                    if not ok:
                        entry["error"] = sanitize(err)
                        res.attempts.append(entry)
                        continue
                    res.success = True
                    res.account_idx = member.idx
                    res.account_group = f"{g.label}#{g.tag}"
                    res.http_status = code
                    res.elapsed_s = round(time.perf_counter() - t0, 3)
                    usage = (body.get("result") or {}).get("usage") or {}
                    rep = usage.get("neurons")
                    res.neurons_reported = float(rep) if rep is not None else None
                    est = meta["estimate"](width, height, eff_steps or 1)
                    res.neurons_estimated = est
                    n = res.neurons_reported if res.neurons_reported is not None else est
                    res.cost_usd_estimated = round(n / NEURONS_PER_USD, 6) if n else None
                    res.output_path = output_path
                    member.requests_ok += 1
                    member.neurons_spent += n or 0
                    g.requests_ok += 1
                    g.neurons_spent += n or 0
                    self._rr = self.groups.index(g)
                    res.attempts.append(entry)
                    return res

                entry["error"] = self._err(code, body)
                res.attempts.append(entry)
                res.error = entry["error"]
                member.requests_failed += 1
                if code in (401, 403):
                    member.healthy = False
                    member.reason = "auth failed"
                    continue  # try next token of SAME account
                if code == 429 or "quota" in entry["error"].lower():
                    g.cooldown_until = time.time() + self.QUOTA_COOLDOWN_S
                    break  # quota is per-account: stop trying its other tokens
                if code == 400 and ("model" in entry["error"].lower()
                                    or "not found" in entry["error"].lower()):
                    break  # model unavailable on this account entirely
                break  # 5xx/network: move to next account
            if group_done or restrict_group:
                break
        res.http_status = res.http_status or last_http
        res.elapsed_s = round(time.perf_counter() - t0, 3)
        if not res.success and not res.error:
            res.error = "no healthy account available"
        return res

    def _rotation(self, restrict_group: Optional[str]) -> list[QuotaGroup]:
        pool = [g for g in self.groups if not restrict_group or g.tag == restrict_group]
        if restrict_group:
            return pool
        n = len(pool)
        if n == 0:
            return []
        start = (self._rr + 1) % n
        return [pool[(start + i) % n] for i in range(n)]

    @staticmethod
    def _save_image(body: dict, output_path: str) -> tuple[bool, str]:
        result = body.get("result") or {}
        b64 = result.get("image")
        if not b64 and isinstance(result.get("images"), list):
            b64 = result["images"][0]
        if not b64:
            return False, f"no image payload in response (keys: {list(result)})"
        try:
            raw = base64.b64decode(b64)
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(raw)
            return True, ""
        except Exception as e:
            return False, f"decode/save failed: {e}"
