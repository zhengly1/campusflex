"""LLM access layer with the common guardrails of Section 3.4:

* structured JSON output: the JSON schema is sent with every request and the reply is
  checked against it locally (``schema_errors``) before any role sees it;
* model name/version recorded per call (the served model and, for DeepSeek, the
  ``system_fingerprint`` are read from the response);
* responses cached by prompt hash so batch runs and replays are reproducible: the key covers
  the model, the prompt, the schema and the client's generation settings, and a cached entry
  replays the latency of the original call so the step budget decides identically on replay;
* per-step call budget (≤3 calls) and latency limit (20 s, cumulative over the step) before
  fallback; ``call`` enforces the remaining step budget as a wall-clock deadline.

Providers (``CAMPUSFLEX_PROVIDER``):

* ``deepseek`` (default) – DeepSeek's OpenAI-compatible Chat Completions API through the
  ``openai`` SDK, ``base_url=https://api.deepseek.com``, model ``deepseek-flash``,
  JSON mode (``response_format={"type": "json_object"}``) and **temperature 0** as the PDF
  asks. Credentials: ``DEEPSEEK_API_KEY``.
* ``anthropic`` – Claude through the ``anthropic`` SDK with server-side JSON-schema output
  (sampling parameters such as temperature are not accepted by current Claude models).

``CAMPUSFLEX_MODEL`` overrides the model id. ``ScriptedClient`` replays fixture outputs
(E5, tests) and never contacts an API.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Protocol

PROVIDER_MODELS = {"deepseek": "deepseek-flash", "anthropic": "claude-opus-5-5"}
DEFAULT_PROVIDER = os.environ.get("CAMPUSFLEX_PROVIDER", "deepseek").strip().lower()
if DEFAULT_PROVIDER not in PROVIDER_MODELS:
    raise ValueError(f"CAMPUSFLEX_PROVIDER must be one of {sorted(PROVIDER_MODELS)}, got {DEFAULT_PROVIDER!r}")
DEFAULT_MODEL = os.environ.get("CAMPUSFLEX_MODEL") or PROVIDER_MODELS[DEFAULT_PROVIDER]
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CACHE_DIR = Path(os.environ.get("CAMPUSFLEX_LLM_CACHE", Path(__file__).resolve().parents[2] / ".cache" / "llm"))

# In-loop roles have a hard 20 s budget, so they run at lower effort by default
# (used by the anthropic provider; DeepSeek chat has no effort setting).
DEFAULT_EFFORT = {"L1": "medium", "L2": "low", "L3": "low", "L4": "medium", "L5": "high"}

# Output-token caps per role for DeepSeek. A reply that needs more is truncated (finish_reason
# "length" -> error "max_tokens" -> fallback); the caps keep a runaway JSON-mode generation from
# running far past the 20 s step budget. L3 (a few adjustments) and L4 (3-6 sentences) are short.
DEFAULT_MAX_TOKENS = {"L1": 1200, "L2": 2500, "L3": 800, "L4": 600, "L5": 1500, "preflight": 50}
FALLBACK_MAX_TOKENS = 2000
REASONER_MIN_MAX_TOKENS = 8000          # deepseek-reasoner counts its chain of thought in max_tokens
# DeepSeek V4 models (deepseek-flash, deepseek-v4-pro) think before answering unless told not to; the
# thinking tokens count toward max_tokens and latency. The design needs short JSON replies within the
# 20 s step budget at temperature 0, so thinking is disabled unless CAMPUSFLEX_DEEPSEEK_THINKING=1.
DEEPSEEK_THINKING = os.environ.get("CAMPUSFLEX_DEEPSEEK_THINKING", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LLMResponse:
    role: str
    data: Any | None
    text: str = ""
    model: str = ""
    cached: bool = False
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    error: str | None = None
    prompt_hash: str = ""
    request_id: str | None = None
    system_fingerprint: str | None = None
    cache_hit_tokens: int = 0        # DeepSeek context caching: input tokens served from the prefix cache

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None

    def transcript(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("data", None)
        d["data"] = self.data
        return d


_RESPONSE_FIELDS = {f.name for f in fields(LLMResponse)}


class LLMClient(Protocol):
    model: str

    def complete_json(
        self, *, role: str, system: str, user: str, schema: dict[str, Any], effort: str | None = None
    ) -> LLMResponse: ...


def prompt_hash(model: str, role: str, system: str, user: str, schema: dict[str, Any],
                settings: dict[str, Any] | None = None) -> str:
    """Cache key. ``settings`` are the generation settings that change the answer (provider,
    temperature, max_tokens, effort, ...); they are part of the key whenever they are given."""
    obj: dict[str, Any] = {"model": model, "role": role, "system": system, "user": user, "schema": schema}
    if settings:
        obj["settings"] = settings
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _settings_of(client: Any, role: str, effort: str | None) -> dict[str, Any]:
    fn = getattr(client, "generation_settings", None)
    return fn(role, effort) if callable(fn) else {}


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} is not valid JSON")


def loads_strict(text: str | bytes) -> Any:
    """``json.loads`` that rejects NaN/Infinity (Python accepts them by default; JSON does not)."""
    return json.loads(text, parse_constant=_reject_constant)


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,), "array": (list,), "string": (str,), "boolean": (bool,),
    "integer": (int,), "number": (int, float), "null": (type(None),),
}


def schema_errors(data: Any, schema: dict[str, Any], path: str = "$", limit: int = 10) -> list[str]:
    """Local JSON-schema check for the subset the role schemas use (type, properties,
    required, additionalProperties, items, enum, anyOf). Returns human-readable errors."""
    errs: list[str] = []

    def walk(x: Any, s: dict[str, Any], p: str) -> None:
        if len(errs) >= limit or not isinstance(s, dict):
            return
        if "anyOf" in s:
            if not any(not schema_errors(x, alt, p, 1) for alt in s["anyOf"]):
                errs.append(f"{p}: matches no anyOf branch")
            return
        t = s.get("type")
        if t is not None:
            types = t if isinstance(t, list) else [t]
            ok = False
            for name in types:
                py = _JSON_TYPES.get(name, ())
                if isinstance(x, bool) and name in ("integer", "number"):
                    continue  # JSON booleans are not numbers
                if isinstance(x, float) and name in ("integer", "number") and not math.isfinite(x):
                    continue  # NaN/Infinity are not JSON numbers
                if name == "integer" and isinstance(x, float) and x.is_integer():
                    ok = True
                    break
                if isinstance(x, py):
                    ok = True
                    break
            if not ok:
                errs.append(f"{p}: expected {t}, got {type(x).__name__}")
                return
        if "enum" in s and x not in s["enum"]:
            errs.append(f"{p}: {x!r} not in {s['enum']}")
        if isinstance(x, dict):
            props = s.get("properties", {})
            for k in s.get("required", []):
                if k not in x:
                    errs.append(f"{p}: missing required {k!r}")
            for k, v in x.items():
                if k in props:
                    walk(v, props[k], f"{p}.{k}")
                elif s.get("additionalProperties") is False:
                    errs.append(f"{p}: unexpected key {k!r}")
                elif isinstance(s.get("additionalProperties"), dict):
                    walk(v, s["additionalProperties"], f"{p}.{k}")
        if isinstance(x, list) and isinstance(s.get("items"), dict):
            for i, v in enumerate(x):
                walk(v, s["items"], f"{p}[{i}]")

    walk(data, schema, path)
    return errs[:limit]


def json_instruction(schema: dict[str, Any]) -> str:
    """Appended to the system prompt for providers without server-side schema enforcement."""
    return ("\n\nOutput format: reply with one JSON object only (no prose, no code fences) that "
            "conforms to this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False, separators=(",", ":")))


class DeepSeekJSONClient:
    """DeepSeek via its OpenAI-compatible API (``openai`` SDK) in JSON mode, temperature 0.

    DeepSeek's JSON mode guarantees syntactically valid JSON but not schema conformance, so
    the schema is placed in the system prompt and the reply is checked locally; a reply that
    violates the schema is returned as an error (``schema: ...``) and the caller falls back.

    The SDK ``timeout`` is a per-read timeout: DeepSeek keeps slow non-streaming requests alive
    with blank lines, so it does not bound the wall-clock time of a call. The step deadline is
    enforced by :func:`call`.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout_s: float = 20.0,
        max_tokens: int | dict[str, int] | None = None,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
        enforce_schema: bool = True,
        effort: dict[str, str] | None = None,  # accepted for interface parity; unused
        client: Any = None,
        thinking: bool | None = None,
    ):
        import openai  # optional dependency

        self._openai = openai
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if client is None and not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        # max_retries=0: a retry would break the 20 s per-step latency budget; the planner falls back instead.
        self.client = client or openai.OpenAI(api_key=key, base_url=base_url or DEEPSEEK_BASE_URL,
                                              timeout=timeout_s, max_retries=0)
        self.model = model or (DEFAULT_MODEL if DEFAULT_PROVIDER == "deepseek" else PROVIDER_MODELS["deepseek"])
        if isinstance(max_tokens, int):
            self.max_tokens: dict[str, int] = {}
            self.default_max_tokens = int(max_tokens)
        else:
            self.max_tokens = {**DEFAULT_MAX_TOKENS, **(max_tokens or {})}
            self.default_max_tokens = FALLBACK_MAX_TOKENS
        self.temperature = temperature
        self.enforce_schema = enforce_schema
        self.thinking = DEEPSEEK_THINKING if thinking is None else bool(thinking)

    @property
    def _reasoner(self) -> bool:
        return self.model.startswith("deepseek-reasoner")

    def max_tokens_for(self, role: str) -> int:
        n = int(self.max_tokens.get(role, self.default_max_tokens))
        return max(n, REASONER_MIN_MAX_TOKENS) if self._reasoner else n

    def generation_settings(self, role: str, effort: str | None = None) -> dict[str, Any]:
        return {"provider": "deepseek", "json_mode": True,
                "temperature": None if self._reasoner else self.temperature,
                "max_tokens": self.max_tokens_for(role), "enforce_schema": self.enforce_schema,
                "thinking": True if self._reasoner else self.thinking}

    def complete_json(self, *, role: str, system: str, user: str, schema: dict[str, Any],
                      effort: str | None = None) -> LLMResponse:
        o = self._openai
        h = prompt_hash(self.model, role, system, user, schema, self.generation_settings(role, effort))
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system + json_instruction(schema)},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens_for(role),
        }
        if not self._reasoner:
            kwargs["temperature"] = self.temperature  # the reasoner ignores sampling parameters
            kwargs["extra_body"] = {"thinking": {"type": "enabled" if self.thinking else "disabled"}}
        t0 = time.monotonic()

        def fail(msg: str) -> LLMResponse:
            return LLMResponse(role, None, error=msg, latency_s=time.monotonic() - t0, prompt_hash=h, model=self.model)

        try:
            resp = self.client.chat.completions.create(**kwargs)
        except o.APITimeoutError:
            return fail("timeout")
        except o.RateLimitError as e:
            return fail(f"rate_limited: {e.message}")
        except o.BadRequestError as e:
            return fail(f"bad_request: {e.message}")
        except o.APIStatusError as e:
            return fail(f"api_status_{e.status_code}: {e.message}")
        except o.APIConnectionError as e:
            return fail(f"connection: {e}")
        except o.APIError as e:  # e.g. response validation errors: still a failed call, not a crash
            return fail(f"api_error: {type(e).__name__}: {e}")
        latency = time.monotonic() - t0
        usage = getattr(resp, "usage", None)
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        finish = getattr(choice, "finish_reason", None)
        base = dict(
            role=role,
            model=getattr(resp, "model", None) or self.model,
            latency_s=latency,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=finish,
            prompt_hash=h,
            request_id=getattr(resp, "id", None),
            system_fingerprint=getattr(resp, "system_fingerprint", None),
            cache_hit_tokens=int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0),
        )
        if choice is None:
            return LLMResponse(data=None, error="empty_response", **base)
        if finish == "content_filter":
            return LLMResponse(data=None, error="refusal", **base)
        if finish == "length":
            return LLMResponse(data=None, error="max_tokens", **base)
        if finish == "insufficient_system_resource":   # DeepSeek: generation interrupted server-side
            return LLMResponse(data=None, text=getattr(choice.message, "content", None) or "",
                               error="insufficient_system_resource", **base)
        text = (getattr(choice.message, "content", None) or "").strip()
        if not text:  # DeepSeek JSON mode can occasionally return empty content
            return LLMResponse(data=None, error="empty_response", **base)
        try:
            data = loads_strict(text)
        except ValueError as e:      # json.JSONDecodeError is a ValueError; so is NaN/Infinity
            return LLMResponse(data=None, text=text, error=f"invalid_json: {e}", **base)
        if self.enforce_schema:
            errs = schema_errors(data, schema)
            if errs:
                return LLMResponse(data=None, text=text, error="schema: " + "; ".join(errs[:3]), **base)
        return LLMResponse(data=data, text=text, **base)


class AnthropicJSONClient:
    """Claude via the official ``anthropic`` SDK with JSON-schema structured output."""

    def __init__(
        self,
        model: str | None = None,
        timeout_s: float = 20.0,
        max_tokens: int = 8000,
        effort: dict[str, str] | None = None,
        use_fallbacks: bool = True,
    ):
        import anthropic  # optional dependency

        self._anthropic = anthropic
        # max_retries=0: a retry would break the 20 s per-step latency budget; the planner falls back instead.
        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)
        self.model = model or (DEFAULT_MODEL if DEFAULT_PROVIDER == "anthropic" else PROVIDER_MODELS["anthropic"])
        self.max_tokens = max_tokens
        self.effort = {**DEFAULT_EFFORT, **(effort or {})}
        self.use_fallbacks = use_fallbacks

    def generation_settings(self, role: str, effort: str | None = None) -> dict[str, Any]:
        return {"provider": "anthropic", "max_tokens": self.max_tokens,
                "effort": effort or self.effort.get(role, "medium"), "fallbacks": self.use_fallbacks}

    def complete_json(self, *, role: str, system: str, user: str, schema: dict[str, Any],
                      effort: str | None = None) -> LLMResponse:
        a = self._anthropic
        h = prompt_hash(self.model, role, system, user, schema, self.generation_settings(role, effort))
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema},
                "effort": effort or self.effort.get(role, "medium"),
            },
        }
        if self.use_fallbacks:
            # Server-side refusal fallback, routed by refusal category.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        t0 = time.monotonic()
        try:
            resp = self.client.beta.messages.create(**kwargs)
        except a.APITimeoutError:
            return LLMResponse(role, None, error="timeout", latency_s=time.monotonic() - t0, prompt_hash=h, model=self.model)
        except a.RateLimitError as e:
            return LLMResponse(role, None, error=f"rate_limited: {e.message}", latency_s=time.monotonic() - t0,
                               prompt_hash=h, model=self.model)
        except a.BadRequestError as e:
            return LLMResponse(role, None, error=f"bad_request: {e.message}", latency_s=time.monotonic() - t0,
                               prompt_hash=h, model=self.model)
        except a.APIStatusError as e:
            return LLMResponse(role, None, error=f"api_status_{e.status_code}: {e.message}",
                               latency_s=time.monotonic() - t0, prompt_hash=h, model=self.model)
        except a.APIConnectionError as e:
            return LLMResponse(role, None, error=f"connection: {e}", latency_s=time.monotonic() - t0,
                               prompt_hash=h, model=self.model)
        except a.APIError as e:  # e.g. response validation errors: still a failed call, not a crash
            return LLMResponse(role, None, error=f"api_error: {type(e).__name__}: {e}",
                               latency_s=time.monotonic() - t0, prompt_hash=h, model=self.model)
        latency = time.monotonic() - t0
        usage = getattr(resp, "usage", None)
        base = dict(
            role=role,
            model=getattr(resp, "model", self.model),
            latency_s=latency,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            stop_reason=resp.stop_reason,
            prompt_hash=h,
            request_id=getattr(resp, "_request_id", None),
        )
        if resp.stop_reason == "refusal":
            return LLMResponse(data=None, error="refusal", **base)
        if resp.stop_reason == "max_tokens":
            return LLMResponse(data=None, error="max_tokens", **base)
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        try:
            data = loads_strict(text)
        except ValueError as e:
            return LLMResponse(data=None, text=text, error=f"invalid_json: {e}", **base)
        errs = schema_errors(data, schema)     # structured output is enforced server-side; check anyway
        if errs:
            return LLMResponse(data=None, text=text, error="schema: " + "; ".join(errs[:3]), **base)
        return LLMResponse(data=data, text=text, **base)


class CachedClient:
    """Prompt-hash cache. ``enabled=False`` for the three uncached repeats per held-out seed.

    The key includes the inner client's generation settings. Only successful replies are
    stored, together with the latency of the original call; a hit returns that latency (with
    ``cached=True``) so :class:`CallBudget` accepts or rejects it exactly as it did live.
    """

    def __init__(self, inner: LLMClient, cache_dir: Path | str = CACHE_DIR, enabled: bool = True):
        self.inner = inner
        self.model = inner.model
        self.dir = Path(cache_dir)
        self.enabled = enabled

    def generation_settings(self, role: str, effort: str | None = None) -> dict[str, Any]:
        return _settings_of(self.inner, role, effort)

    def _key(self, role: str, system: str, user: str, schema: dict[str, Any], effort: str | None) -> str:
        return prompt_hash(self.model, role, system, user, schema, _settings_of(self.inner, role, effort))

    def complete_json(self, *, role: str, system: str, user: str, schema: dict[str, Any],
                      effort: str | None = None) -> LLMResponse:
        h = self._key(role, system, user, schema, effort)
        p = self.dir / f"{h}.json"
        if self.enabled and p.exists():
            try:
                d = json.loads(p.read_text())
                d = {k: v for k, v in d.items() if k in _RESPONSE_FIELDS}
                d["cached"] = True
                d["latency_s"] = float(d.get("latency_s", 0.0))   # the original call's latency
                d["prompt_hash"] = h
                r = LLMResponse(**d)
                if r.ok:
                    return r
            except (ValueError, TypeError, OSError):
                pass                                 # unreadable entry: treat as a miss
        r = self.inner.complete_json(role=role, system=system, user=user, schema=schema, effort=effort)
        if self.enabled and r.ok:
            tmp = p.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(asdict(r), ensure_ascii=False))
                os.replace(tmp, p)                   # atomic: concurrent readers never see half a file
            except OSError:
                tmp.unlink(missing_ok=True)          # a full or read-only disk costs the cache, not the reply
        return r


class ScriptedClient:
    """Deterministic stand-in: returns fixture outputs (dict/str) or calls a responder.

    ``responder(role, system, user) -> Any``; a str is treated as raw model text (may be
    malformed JSON, which is exactly what E5 needs), an Exception instance simulates an
    API failure, a ``("sleep", s)`` tuple simulates a timed-out call that took ``s`` seconds and
    an ``LLMResponse`` is returned as is (e.g. a slow but valid reply).
    """

    def __init__(self, responder: Callable[[str, str, str], Any] | dict[str, Any] | list[Any], model: str = "scripted"):
        self.responder = responder
        self.model = model
        self.calls: list[dict[str, Any]] = []
        self._i = 0

    def _next(self, role: str, system: str, user: str) -> Any:
        if callable(self.responder):
            return self.responder(role, system, user)
        if isinstance(self.responder, dict):
            return self.responder.get(role)
        out = self.responder[self._i % len(self.responder)]
        self._i += 1
        return out

    def complete_json(self, *, role: str, system: str, user: str, schema: dict[str, Any],
                      effort: str | None = None) -> LLMResponse:
        h = prompt_hash(self.model, role, system, user, schema)
        self.calls.append({"role": role, "prompt_hash": h, "user": user})
        out = self._next(role, system, user)
        if isinstance(out, LLMResponse):
            return out
        if isinstance(out, BaseException):
            return LLMResponse(role, None, error=f"{type(out).__name__}: {out}", prompt_hash=h, model=self.model)
        if isinstance(out, tuple) and out and out[0] == "sleep":
            return LLMResponse(role, None, error="timeout", latency_s=float(out[1]), prompt_hash=h, model=self.model)
        if isinstance(out, (str, bytes)):
            try:
                data = loads_strict(out)
            except ValueError as e:
                return LLMResponse(role, None, text=str(out), error=f"invalid_json: {e}", prompt_hash=h, model=self.model)
            return LLMResponse(role, data, text=str(out), prompt_hash=h, model=self.model)
        return LLMResponse(role, out, text=json.dumps(out, default=str), prompt_hash=h, model=self.model)


@dataclass
class CallBudget:
    """At most ``max_calls`` LLM calls and ``latency_s`` seconds (cumulative) per decision step.

    The budget is shared by all LLM roles within one decision step. A reply is accepted only if
    the step's cumulative latency including it stays within ``latency_s``; cached replies carry
    the latency of the original call, so replays decide exactly as the live run did.
    """

    max_calls: int = 3
    latency_s: float = 20.0
    used_calls: int = 0
    used_s: float = 0.0
    step: int = -1
    log: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.used_calls, self.used_s = 0, 0.0

    def at_step(self, step: int) -> "CallBudget":
        """Shared by all LLM roles within one decision step; resets when the step changes."""
        if step != self.step:
            self.step = step
            self.reset()
        return self

    def remaining_s(self) -> float:
        return max(0.0, self.latency_s - self.used_s)

    def allow(self) -> bool:
        return self.used_calls < self.max_calls and self.used_s < self.latency_s

    def charge(self, r: LLMResponse) -> LLMResponse:
        self.used_calls += 1
        self.used_s += r.latency_s
        if r.error is None and self.used_s > self.latency_s + 1e-9:
            r.error = "latency_exceeded"
            r.data = None
        self.log.append({"role": r.role, "step": self.step, "latency_s": round(r.latency_s, 3), "error": r.error,
                         "in": r.input_tokens, "out": r.output_tokens, "cached": r.cached,
                         "used_s": round(self.used_s, 3)})
        return r


class AbandonedUsage:
    """Calls the caller stopped waiting for (step deadline). The request still completes and is
    billed; its tokens are added here when it does, so cost accounting does not lose them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = self.completed = self.input_tokens = self.output_tokens = self.cache_hit_tokens = 0

    def abandon(self) -> None:
        with self._lock:
            self.calls += 1

    def add(self, r: LLMResponse) -> None:
        with self._lock:
            self.completed += 1
            self.input_tokens += int(r.input_tokens or 0)
            self.output_tokens += int(r.output_tokens or 0)
            self.cache_hit_tokens += int(r.cache_hit_tokens or 0)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"calls": self.calls, "completed_late": self.completed, "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens, "cache_hit_tokens": self.cache_hit_tokens}

    @staticmethod
    def delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
        return {k: after[k] - before.get(k, 0) for k in after}


ABANDONED = AbandonedUsage()

# Wall-clock limit for calls outside a decision step (L1, L5, E4 explanations): the SDK timeout
# bounds each read, not the whole request.
UNBUDGETED_DEADLINE_S = 180.0


def _complete_with_deadline(client: LLMClient, deadline_s: float, *, role: str, system: str, user: str,
                            schema: dict[str, Any], effort: str | None) -> LLMResponse:
    """Run ``complete_json`` in a daemon thread and give up after ``deadline_s`` wall-clock seconds.

    An abandoned call keeps running in the background (the HTTP request cannot be cancelled from
    here); if it later succeeds, a ``CachedClient`` stores it with its real latency, so a replay
    rejects it on latency exactly as the live run did, and its billed tokens go to ``ABANDONED``.
    """
    box: dict[str, Any] = {}
    lock = threading.Lock()
    done = threading.Event()

    def run() -> None:
        try:
            r = client.complete_json(role=role, system=system, user=user, schema=schema, effort=effort)
        except Exception as e:  # noqa: BLE001 - becomes a failed call in the caller's thread
            r = LLMResponse(role, None, error=f"exception: {type(e).__name__}: {e}", model=getattr(client, "model", ""))
        with lock:
            box["r"] = r
            late = box.get("abandoned", False)
            done.set()
        if late:
            ABANDONED.add(r)

    t0 = time.monotonic()
    threading.Thread(target=run, name=f"llm-{role}", daemon=True).start()
    done.wait(deadline_s)
    with lock:
        if not done.is_set():
            box["abandoned"] = True
            ABANDONED.abandon()
            return LLMResponse(role, None, error="timeout", latency_s=max(deadline_s, time.monotonic() - t0),
                               model=getattr(client, "model", ""))
    return box["r"]


# HTTP 401 (bad key) / 402 (insufficient balance): every later call fails the same way, so a
# long run stops instead of finishing as a fallback-only run (E1 C/D).
ACCOUNT_ERRORS = ("api_status_401", "api_status_402")


def is_account_error(error: str | None) -> bool:
    return bool(error) and str(error).startswith(ACCOUNT_ERRORS)


def call(client: LLMClient, budget: CallBudget | None, *, role: str, system: str, user: str,
         schema: dict[str, Any], effort: str | None = None) -> LLMResponse:
    """Budgeted call: refuses (as a fallback) when the step's budget is spent, and stops waiting
    for the reply when the remaining latency budget of the step runs out (error ``timeout``)."""
    if budget is not None and not budget.allow():
        return LLMResponse(role, None, error="budget_exhausted", model=getattr(client, "model", ""))
    remaining = budget.remaining_s() if budget is not None else math.inf
    if not math.isfinite(remaining):           # no step budget (or an unbounded one): wall-clock limit
        remaining = UNBUDGETED_DEADLINE_S
    # Any exception (SDK, network stack, a bug in a stand-in client) is a failed call: the
    # caller falls back instead of losing the episode.
    r = _complete_with_deadline(client, remaining, role=role, system=system, user=user, schema=schema,
                                effort=effort)
    return budget.charge(r) if budget is not None else r


def make_client(kind: str | None = None, cache: bool = True, **kw: Any) -> LLMClient:
    """``deepseek`` (default, ``CAMPUSFLEX_PROVIDER``) or ``anthropic``. Scripted clients are constructed directly."""
    kind = (kind or DEFAULT_PROVIDER).lower()
    kw = {k: v for k, v in kw.items() if v is not None}
    if kind == "deepseek":
        return CachedClient(DeepSeekJSONClient(**kw), enabled=cache)
    if kind == "anthropic":
        return CachedClient(AnthropicJSONClient(**kw), enabled=cache)
    raise ValueError(f"unknown client kind {kind!r}")


def wrap_data(user_text: str, tag: str = "notice") -> str:
    """Place untrusted text in a delimited data field (prompt-injection guardrail).

    Any opening or closing ``tag`` inside the text (any case, any spacing) is neutralised, so
    the text cannot close its own field or open a fake one."""
    safe = re.sub(rf"<(\s*/?\s*{re.escape(tag)}\b)", r"&lt;\1", str(user_text), flags=re.I)
    return f"<{tag}>\n{safe}\n</{tag}>"


def preflight(client: LLMClient) -> LLMResponse:
    """One tiny structured call to prove the LLM is reachable before an LLM arm is run.

    Experiments must not run (and must report ``not_run``) when this fails; otherwise every
    call would fall back and a masks-only result could be mistaken for an LLM result.
    """
    return client.complete_json(
        role="preflight",
        system="Reply with the requested JSON only.",
        user='Return {"ok": true}.',
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                "additionalProperties": False},
        effort="low",
    )


def live_client_or_reason(cache: bool = True, kind: str | None = None,
                          **kw: Any) -> tuple[LLMClient | None, str | None]:
    """Return (client, None) if a live client of the configured provider works, else (None, reason)."""
    try:
        c = make_client(kind, cache=False, **kw)
    except Exception as e:  # SDK missing, credentials missing
        return None, f"{type(e).__name__}: {e}"
    r = preflight(c)
    if not r.ok:
        return None, f"preflight failed: {r.error}"
    return (CachedClient(c.inner, enabled=True) if cache else c), None
