"""Language-model client (Kimi K3 through an OpenAI-compatible endpoint).

The project owner chose Kimi K3, with credentials in `.env`:

    LLM_API_KEY=...
    LLM_ENDPOINT=https://.../v1        # API base, OpenAI-compatible
    LLM_MODEL=kimi-k3                  # optional; defaults to kimi-k3

Everything else in the system talks to this class and only to it: changing provider means
changing this file, not the agents. Every call also returns its cost in tokens and seconds,
which is what the gym's trace persists (§10.2).

Three things kimi-k3 imposes, discovered by probing (DECISIONS D-036):
  * the temperature is fixed (only 1 is accepted) — detected once, then omitted;
  * it is a reasoning model: `content` carries the answer and `reasoning_content` the chain of
    thought, and `max_tokens` covers BOTH — a short budget returns empty or truncated text;
  * it is slow: a planning call can take minutes, so read timeouts are not retried (retrying
    pays the same work again).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

ENV_PATH = Path(".env")


class CredentialsMissing(RuntimeError):
    """Raised when credentials are absent: the system never fabricates model answers."""


class EmptyCompletion(RuntimeError):
    """The model spent its budget reasoning and left no (or a truncated) answer."""


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Minimal .env reader, no dependencies. Ignores comments and strips quotes."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in ("LLM_API_KEY", "LLM_ENDPOINT", "LLM_MODEL"):
        if key in os.environ and os.environ[key]:
            values[key] = os.environ[key]
    return values


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    calls: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens
        self.seconds += other.seconds
        self.calls += other.calls

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage
    model: str
    finish_reason: str | None = None
    reasoning: str = ""          # kimi-k3 returns the chain of thought separately
    raw: dict = field(default_factory=dict)

    def json(self) -> Any:
        """Parse the answer as JSON, tolerating code fences."""
        text = self.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        return json.loads(text)


class LLMClient:
    """OpenAI-compatible chat client. Does nothing without credentials."""

    def __init__(self, model: str | None = None, timeout: float = 600.0,
                 env: dict[str, str] | None = None) -> None:
        """`timeout` is high on purpose: kimi-k3 reasons before answering and a planning
        request can take several minutes. A short timeout does not make it faster; it only
        triggers retries that pay the same work two and three times."""
        env = env if env is not None else load_env()
        self.api_key = env.get("LLM_API_KEY", "")
        self.endpoint = (env.get("LLM_ENDPOINT", "") or "").rstrip("/")
        self.model = model or env.get("LLM_MODEL") or "kimi-k3"
        self.timeout = timeout
        self.usage = LLMUsage()
        # Some models pin the temperature (kimi-k3 only accepts 1). Detected on the first call
        # and remembered, instead of keeping a list of models that goes stale.
        self.supports_temperature = True

    # ------------------------------------------------------------------ state
    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.endpoint)

    def require(self) -> None:
        if not self.configured:
            missing = [k for k, v in (("LLM_API_KEY", self.api_key),
                                      ("LLM_ENDPOINT", self.endpoint)) if not v]
            raise CredentialsMissing(
                f"missing {', '.join(missing)} in .env (or in the environment). Without "
                f"credentials no agent runs: the system does not simulate model answers. "
                f"Check the configuration with: python scripts/check_llm.py")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    # ------------------------------------------------------------------ calls
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30),
           # Server errors and rate limits are retried; a read timeout is NOT: if the model
           # took too long, retrying pays the same work again.
           retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
           reraise=True)
    def _post(self, path: str, payload: dict) -> dict:
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.endpoint}{path}", headers=self._headers(),
                                   json=payload)
        except httpx.ReadTimeout as exc:
            raise TimeoutError(
                f"{self.model} did not answer within {self.timeout:.0f} s. It is a reasoning "
                f"model: raise the timeout or lower max_tokens "
                f"(current: {payload.get('max_tokens')}).") from exc
        if resp.status_code >= 500 or resp.status_code == 429:
            resp.raise_for_status()
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code} on {path}: {resp.text[:400]}")
        return resp.json()

    def complete(self, messages: Iterable[dict], *, system: str | None = None,
                 temperature: float = 0.3, max_tokens: int = 8192,
                 response_format: dict | None = None,
                 tools: list[dict] | None = None, _retry: bool = True) -> LLMResponse:
        """`max_tokens` covers reasoning **and** answer: kimi-k3 spends hundreds of tokens
        thinking before writing, so a short budget returns empty text."""
        self.require()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ([{"role": "system", "content": system}] if system else []) + list(messages),
            "max_tokens": max_tokens,
        }
        if self.supports_temperature:
            payload["temperature"] = temperature
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        started = time.perf_counter()
        try:
            data = self._post("/chat/completions", payload)
        except RuntimeError as exc:
            if "temperature" in str(exc).lower() and self.supports_temperature:
                self.supports_temperature = False
                payload.pop("temperature", None)
                data = self._post("/chat/completions", payload)
            else:
                raise
        elapsed = time.perf_counter() - started

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage_raw = data.get("usage") or {}
        details = usage_raw.get("completion_tokens_details") or {}
        usage = LLMUsage(prompt_tokens=int(usage_raw.get("prompt_tokens", 0)),
                         completion_tokens=int(usage_raw.get("completion_tokens", 0)),
                         reasoning_tokens=int(details.get("reasoning_tokens", 0)),
                         cached_tokens=int((usage_raw.get("prompt_tokens_details") or {})
                                           .get("cached_tokens", 0)),
                         seconds=elapsed, calls=1)
        self.usage.add(usage)
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""

        truncated = choice.get("finish_reason") == "length"
        if not text.strip() or truncated:
            # Two faces of the same problem: the `max_tokens` budget covers reasoning and
            # answer, so if the model thinks a lot the answer comes out empty or cut mid-way.
            # Cut is worse than empty, because it looks valid until it is parsed.
            if _retry:
                return self.complete(messages, system=system, temperature=temperature,
                                     max_tokens=max_tokens * 2, response_format=response_format,
                                     tools=tools, _retry=False)
            reason = "was truncated by the token limit" if truncated else "returned no content"
            raise EmptyCompletion(
                f"{self.model} {reason}: {usage.reasoning_tokens} reasoning tokens out of "
                f"{max_tokens} available. Raise max_tokens or split the request.")
        return LLMResponse(text=text, usage=usage, model=data.get("model", self.model),
                           finish_reason=choice.get("finish_reason"), reasoning=reasoning,
                           raw=data)

    def list_models(self) -> list[str]:
        self.require()
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{self.endpoint}/models", headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        return [m.get("id", "?") for m in data.get("data", [])]
