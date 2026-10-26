"""DeepSeek client following the workspace calling conventions.

Conventions mirrored from project/era/implementation/llm.py:
- key read from a single-line file (UTF-8, BOM tolerated), never logged;
- SDK auto-retry disabled, bounded manual retries for transient failures only;
- error messages sanitized to exception type + HTTP status;
- per-call usage and returned model name recorded for provenance.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from openai import APIStatusError, APIConnectionError, APITimeoutError, OpenAIError

BASE_URL = "https://api.deepseek.com"


class DeepSeekError(RuntimeError):
    """Sanitized DeepSeek call failure (no server body, no credentials)."""


def read_deepseek_key(secret_path: Path | str) -> str:
    path = Path(secret_path)
    text = path.read_text(encoding="utf-8-sig")
    key = text.strip()
    if not key or "\n" in key:
        raise DeepSeekError(
            f"Key file {path} must contain exactly one non-empty line."
        )
    return key


@dataclass
class CallRecord:
    stage: str
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_seconds: float
    ok: bool
    error: str | None = None
    repair_attempts: int = 0


@dataclass
class UsageHistory:
    records: list[CallRecord] = field(default_factory=list)

    def totals(self) -> dict[str, int]:
        prompt = sum(r.prompt_tokens or 0 for r in self.records)
        completion = sum(r.completion_tokens or 0 for r in self.records)
        return {
            "calls": len(self.records),
            "failed_calls": sum(1 for r in self.records if not r.ok),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }


def extract_json(content: str) -> Any:
    """Extract a JSON value from model output, tolerating ```json fences
    and surrounding prose."""
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Model output contained no parsable JSON value.")


class DeepSeekClient:
    """Chat client for idea generation (longer outputs than the ERA code
    mutation calls: bigger max_tokens, thinking configurable)."""

    def __init__(
        self,
        model_name: str = "deepseek-flash",
        secret_path: Path | str = "../project/secret.txt",
        timeout_seconds: float = 180.0,
        transient_retries: int = 2,
        max_tokens: int = 8192,
        temperature: float = 1.0,
        thinking: str = "enabled",
    ):
        self.client = OpenAI(
            api_key=read_deepseek_key(secret_path),
            base_url=BASE_URL,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.model_name = model_name
        self.transient_retries = transient_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.history = UsageHistory()

    def chat(
        self,
        system: str,
        user: str,
        stage: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        started = time.monotonic()
        record = CallRecord(
            stage=stage, model=None, prompt_tokens=None,
            completion_tokens=None, latency_seconds=0.0, ok=False,
        )
        attempts = self.transient_retries + 1
        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=(
                        self.temperature if temperature is None else temperature
                    ),
                    extra_body={"thinking": {"type": self.thinking}},
                )
                content = (
                    response.choices[0].message.content
                    if response.choices else None
                )
                if not content or not content.strip():
                    raise DeepSeekError("Empty content returned.")
                finish_reason = (
                    response.choices[0].finish_reason
                    if response.choices else None
                )
                if finish_reason != "stop":
                    # Truncated output (e.g. thinking consumed the token
                    # budget) is not transient: record and fail the call.
                    raise DeepSeekError(
                        f"Incomplete response (finish_reason="
                        f"{finish_reason})."
                    )
                record.model = response.model
                if response.usage:
                    record.prompt_tokens = response.usage.prompt_tokens
                    record.completion_tokens = (
                        response.usage.completion_tokens
                    )
                record.ok = True
                record.latency_seconds = round(
                    time.monotonic() - started, 3
                )
                self.history.records.append(record)
                return content.strip()
            except OpenAIError as error:
                status = getattr(error, "status_code", None)
                transient = isinstance(
                    error, (APIConnectionError, APITimeoutError)
                ) or (isinstance(error, APIStatusError) and (
                    status is None or status >= 500 or status == 429
                ))
                if attempt < attempts - 1 and transient:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                record.error = f"{type(error).__name__}:{status}"
                record.latency_seconds = round(
                    time.monotonic() - started, 3
                )
                self.history.records.append(record)
                raise DeepSeekError(
                    f"DeepSeek call failed at stage '{stage}': "
                    f"{type(error).__name__} (HTTP {status})."
                ) from None
            except DeepSeekError as error:
                # Empty/truncated content: keep the failed call in the
                # usage history so manifests see it, then propagate.
                record.error = f"DeepSeekError:{str(error)[:80]}"
                record.latency_seconds = round(
                    time.monotonic() - started, 3
                )
                self.history.records.append(record)
                raise
        raise DeepSeekError(f"DeepSeek call failed at stage '{stage}'.")  # unreachable

    def chat_json(
        self,
        system: str,
        user: str,
        stage: str = "chat_json",
        temperature: float | None = None,
        max_repair_attempts: int = 1,
        on_repair: Callable[[int], None] | None = None,
    ) -> Any:
        """chat() with JSON parsing and at most max_repair_attempts bounded
        corrective retries. Repair counts are recorded on the call record."""
        content = self.chat(system, user, stage=stage, temperature=temperature)
        repairs = 0
        while True:
            try:
                return extract_json(content)
            except ValueError:
                if repairs >= max_repair_attempts:
                    raise
                repairs += 1
                if on_repair:
                    on_repair(repairs)
                content = self.chat(
                    system,
                    user + (
                        "\n\nYour previous reply was not valid JSON. "
                        "Reply again with ONLY the JSON value, no prose, "
                        "no markdown fences."
                    ),
                    stage=f"{stage}#repair{repairs}",
                    temperature=temperature,
                )
                if self.history.records:
                    self.history.records[-1].repair_attempts = repairs


class FakeLLM:
    """Deterministic stand-in for tests and dry runs (zero API cost)."""

    def __init__(self, responder: Callable[[str, str], str]):
        self.responder = responder
        self.model_name = "fake-model"
        self.temperature = 1.0
        self.history = UsageHistory()

    def chat(
        self,
        system: str,
        user: str,
        stage: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        content = self.responder(system, user)
        self.history.records.append(
            CallRecord(
                stage=stage, model=self.model_name, prompt_tokens=1,
                completion_tokens=1, latency_seconds=0.0, ok=True,
            )
        )
        return content

    def chat_json(self, system: str, user: str, stage: str = "chat_json",
                  temperature: float | None = None,
                  max_repair_attempts: int = 1,
                  on_repair: Callable[[int], None] | None = None) -> Any:
        return extract_json(self.chat(system, user, stage=stage))
