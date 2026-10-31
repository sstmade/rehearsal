"""The only way this project talks to the Meta Model API.

Every call is (1) served from a disk cache when the identical request was made before (free),
otherwise (2) checked against a hard budget by reserving a worst-case estimate under a file lock,
(3) sent with retries, and (4) recorded in an append-only ledger with its real usage and cost.

Budget files live in $REHEARSAL_BUDGET_DIR (default: <repo>/budget), shared by every process and
git worktree that points at the same directory:
    state.json     {"spent": $, "reserved": $, "cap": $}
    ledger.jsonl   one line per paid call
    cache/         one JSON file per request hash

Data privacy: callers pass data="public" for synthetic renders and public-domain/CC photos, which
routes Muse Spark to Contributor mode (cheaper; content may be used to train Meta models).
Anything else, including the user's own photos, must use data="private" (Standard mode, default).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAP = 15.0

# USD per 1M tokens (input, cached input, output) — https://dev.meta.ai/docs/pricing-rate-limits
TOKEN_PRICES = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.002, 0.20),
    "muse-spark-1.2-contributor": (0.10, 0.002, 0.20),
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
}
SAM_PER_IMAGE = 2.50 / 1000
IMAGE_TOKENS_EST = 2500        # worst-case input tokens per attached image
MAX_OUTPUT_EST = 16000         # worst-case output (incl. reasoning) tokens for a reservation


class BudgetExceeded(RuntimeError):
    pass


def budget_dir() -> Path:
    d = Path(os.environ.get("REHEARSAL_BUDGET_DIR", ROOT / "budget"))
    (d / "cache").mkdir(parents=True, exist_ok=True)
    return d


def cap() -> float:
    return float(os.environ.get("REHEARSAL_BUDGET_CAP", DEFAULT_CAP))


@contextmanager
def _locked_state():
    d = budget_dir()
    with open(d / "state.lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        p = d / "state.json"
        st = json.loads(p.read_text()) if p.exists() else {"spent": 0.0, "reserved": 0.0}
        yield st
        st["cap"] = cap()
        p.write_text(json.dumps(st))
        fcntl.flock(lk, fcntl.LOCK_UN)


def status() -> dict:
    with _locked_state() as st:
        return {"spent": round(st["spent"], 4), "reserved": round(st["reserved"], 4), "cap": cap(),
                "remaining": round(cap() - st["spent"] - st["reserved"], 4)}


def _reserve(amount: float, caller: str):
    with _locked_state() as st:
        if st["spent"] + st["reserved"] + amount > cap():
            raise BudgetExceeded(f"{caller}: would exceed ${cap():.2f} cap "
                                 f"(spent ${st['spent']:.3f}, reserved ${st['reserved']:.3f}, need ${amount:.3f})")
        st["reserved"] += amount


def _settle(reserved: float, actual: float, entry: dict):
    with _locked_state() as st:
        st["reserved"] = max(0.0, st["reserved"] - reserved)
        st["spent"] += actual
    with open(budget_dir() / "ledger.jsonl", "a") as f:
        f.write(json.dumps({**entry, "cost": round(actual, 6), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


def _key(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _cache_get(key: str):
    p = budget_dir() / "cache" / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _cache_put(key: str, value):
    (budget_dir() / "cache" / f"{key}.json").write_text(json.dumps(value))


_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        from . import load_env
        load_env()
        _client = OpenAI(base_url=BASE_URL, api_key=os.environ["MODEL_API_KEY"], max_retries=0, timeout=300)
    return _client


def _with_retries(fn, caller: str):
    delay = 1.0
    for attempt in range(4):
        try:
            return fn()
        except Exception as e:  # rate limits, 5xx, transient network errors
            status_code = getattr(e, "status_code", None)
            if status_code is not None and status_code < 500 and status_code != 429:
                raise
            if attempt == 3:
                raise
            time.sleep(delay + random.random())
            delay *= 2


def _estimate_input_tokens(messages) -> tuple[int, int]:
    chars, images = 0, 0
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            chars += len(content)
            continue
        for part in content:
            if part.get("type") in ("image_url", "input_image"):
                images += 1
            else:
                chars += len(part.get("text", ""))
    return chars // 3 + images * IMAGE_TOKENS_EST, images


def muse_model(data: str, base: str = "muse-spark-1.3") -> str:
    if data == "public" and base.startswith("muse-spark") and not base.endswith("-contributor"):
        return base + "-contributor"
    return base


def chat_json(messages: list[dict], schema: dict, name: str, *, caller: str, data: str = "private",
              model: str = "muse-spark-1.3", effort: str | None = None) -> dict:
    """Chat Completions with a strict JSON schema. Returns the parsed JSON object."""
    assert data in ("public", "private")
    model = muse_model(data, model)
    req = {"model": model, "messages": messages,
           "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}}
    if effort:
        req["reasoning_effort"] = effort
    key = _key(req)
    hit = _cache_get(key)
    if hit is not None:
        return hit["content"]

    p_in, _, p_out = TOKEN_PRICES[model]
    in_tok, images = _estimate_input_tokens(messages)
    reserve = (in_tok * p_in + MAX_OUTPUT_EST * p_out) / 1e6
    _reserve(reserve, caller)
    actual = 0.0
    entry = {"caller": caller, "model": model, "kind": "chat", "images": images}
    try:
        r = _with_retries(lambda: client().chat.completions.create(**req), caller)
        u = r.usage
        cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
        p_in, p_cached, p_out = TOKEN_PRICES[model]
        actual = ((u.prompt_tokens - cached) * p_in + cached * p_cached + u.completion_tokens * p_out) / 1e6
        entry.update(prompt_tokens=u.prompt_tokens, cached_tokens=cached, completion_tokens=u.completion_tokens)
        content = json.loads(r.choices[0].message.content)
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        # A failed request may still have been billed; without usage we charge the reservation.
        if actual == 0.0 and not isinstance(e, json.JSONDecodeError):
            actual = reserve if getattr(e, "status_code", 500) >= 500 else 0.0
        _settle(reserve, actual, entry)
        raise
    _settle(reserve, actual, entry)
    _cache_put(key, {"content": content, "model": model, "caller": caller})
    return content


def sam(prompt: str, image_url: str, *, caller: str) -> str:
    """SAM 3.1 on one image (concept or box/point prompt). Returns the raw output_text."""
    req = {"model": "sam-3.1", "prompt": prompt, "image": hashlib.sha256(image_url.encode()).hexdigest()}
    key = _key(req)
    hit = _cache_get(key)
    if hit is not None:
        return hit["text"]
    _reserve(SAM_PER_IMAGE, caller)

    def call():
        text = ""
        with client().responses.stream(model="sam-3.1", input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_url}]}]) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    text += event.delta
        return text

    entry = {"caller": caller, "model": "sam-3.1", "kind": "sam", "images": 1}
    try:
        text = _with_retries(call, caller)
    except Exception as e:
        entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        _settle(SAM_PER_IMAGE, SAM_PER_IMAGE, entry)
        raise
    _settle(SAM_PER_IMAGE, SAM_PER_IMAGE, entry)
    _cache_put(key, {"text": text, "caller": caller})
    return text


if __name__ == "__main__":
    print(json.dumps(status(), indent=1))
