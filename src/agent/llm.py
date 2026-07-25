"""Shared OpenAI-compatible access for the agent's structured-output nodes.

The application has two semantic tiers: easy for bounded control calls and hard
for cited legal answers. Each tier can use the shared endpoint or override its
own URL and key, so the same code works with OmniRoute, a direct provider, or a
local OpenAI-compatible server. RAGAS has a separate pinned judge profile.
"""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Tier = Literal["easy", "hard"]

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MODELS: dict[Tier, str] = {"easy": "deepseek-v4-flash", "hard": "deepseek-v4-pro"}
_MAX_TOKENS: dict[Tier, int] = {"easy": 256, "hard": 1024}
_LEGACY_NAMES: dict[Tier, str] = {"easy": "FLASH", "hard": "PRO"}
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_TIMEOUT_SECONDS = 90.0
_SYNC_CLIENTS: dict[Tier, object] = {}


@dataclass(frozen=True)
class ModelProfile:
    model: str
    base_url: str
    api_key: str = field(repr=False)
    max_tokens: int
    timeout: float
    disable_thinking: bool
    extra_body: dict[str, object] | None = None


def _load_env() -> None:
    """Load .env into the environment (idempotent, no-op if absent)."""
    from dotenv import load_dotenv

    load_dotenv()


def _first_env(*names: str, default: str | None = None) -> str | None:
    _load_env()
    return next((value for name in names if (value := os.getenv(name))), default)


def _parse_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _resolve_key(tier: Tier = "easy") -> str:
    """Return a tier key, then the shared or legacy key, or fail at the boundary."""
    key = _first_env(f"LLM_{tier.upper()}_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            f"The {tier} LLM tier needs LLM_{tier.upper()}_API_KEY or LLM_API_KEY. "
            "The deterministic nodes run without a key."
        )
    return key


def has_api_key(tier: Tier = "easy") -> bool:
    """True when a tier-specific, shared, or legacy key is configured."""
    return bool(_first_env(f"LLM_{tier.upper()}_API_KEY", "LLM_API_KEY", "DEEPSEEK_API_KEY"))


def _model_for(tier: Tier) -> str:
    legacy = _LEGACY_NAMES[tier]
    return (
        _first_env(
            f"LLM_{tier.upper()}_MODEL",
            f"DEEPSEEK_MODEL_{legacy}",
            default=_MODELS[tier],
        )
        or _MODELS[tier]
    )


def _max_tokens_for(tier: Tier) -> int:
    """Return a positive, per-tier completion ceiling from the environment."""
    legacy = _LEGACY_NAMES[tier]
    raw = _first_env(
        f"LLM_{tier.upper()}_MAX_TOKENS",
        f"DEEPSEEK_MAX_TOKENS_{legacy}",
        default=str(_MAX_TOKENS[tier]),
    )
    value = int(raw or _MAX_TOKENS[tier])
    if value < 1:
        raise ValueError("LLM maximum token values must be positive")
    return value


def _base_url_for(tier: Tier) -> str:
    return (
        _first_env(
            f"LLM_{tier.upper()}_BASE_URL",
            "LLM_BASE_URL",
            "DEEPSEEK_BASE_URL",
            default=_DEFAULT_BASE_URL,
        )
        or _DEFAULT_BASE_URL
    )


def _timeout_seconds(tier: Tier = "easy") -> float:
    """Return a positive request limit instead of inheriting the SDK's 10 minutes."""
    raw = _first_env(
        f"LLM_{tier.upper()}_TIMEOUT_SECONDS",
        "LLM_TIMEOUT_SECONDS",
        "DEEPSEEK_TIMEOUT_SECONDS",
        default=str(_DEFAULT_TIMEOUT_SECONDS),
    )
    value = float(raw or _DEFAULT_TIMEOUT_SECONDS)
    if value <= 0:
        raise ValueError("LLM timeout values must be positive")
    return value


def _disable_thinking_for(tier: Tier) -> bool:
    name = f"LLM_{tier.upper()}_DISABLE_THINKING"
    raw = _first_env(name, "LLM_DISABLE_THINKING", default="true") or "true"
    return _parse_bool(raw, name)


def _extra_body_for(tier: Tier) -> dict[str, object] | None:
    raw = _first_env(f"LLM_{tier.upper()}_EXTRA_BODY", "LLM_EXTRA_BODY")
    if not raw:
        return None
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("LLM extra body must be a JSON object")
    return value


def _profile_for(tier: Tier) -> ModelProfile:
    return ModelProfile(
        model=_model_for(tier),
        base_url=_base_url_for(tier),
        api_key=_resolve_key(tier),
        max_tokens=_max_tokens_for(tier),
        timeout=_timeout_seconds(tier),
        disable_thinking=_disable_thinking_for(tier),
        extra_body=_extra_body_for(tier),
    )


def _judge_model_for() -> str:
    return (
        _first_env(
            "RAGAS_JUDGE_MODEL",
            "LLM_EASY_MODEL",
            "DEEPSEEK_MODEL_FLASH",
            default=_MODELS["easy"],
        )
        or _MODELS["easy"]
    )


def _judge_profile() -> ModelProfile:
    """Return the pinned RAGAS judge, falling back to the easy profile settings."""
    key = _first_env(
        "RAGAS_JUDGE_API_KEY",
        "LLM_EASY_API_KEY",
        "LLM_API_KEY",
        "DEEPSEEK_API_KEY",
    )
    if not key:
        raise RuntimeError("RAGAS needs RAGAS_JUDGE_API_KEY, LLM_EASY_API_KEY, or LLM_API_KEY")
    max_tokens = int(
        _first_env(
            "RAGAS_JUDGE_MAX_TOKENS",
            "LLM_EASY_MAX_TOKENS",
            "DEEPSEEK_MAX_TOKENS_FLASH",
            default=str(_MAX_TOKENS["easy"]),
        )
        or _MAX_TOKENS["easy"]
    )
    timeout = float(
        _first_env(
            "RAGAS_JUDGE_TIMEOUT_SECONDS",
            "LLM_EASY_TIMEOUT_SECONDS",
            "LLM_TIMEOUT_SECONDS",
            "DEEPSEEK_TIMEOUT_SECONDS",
            default=str(_DEFAULT_TIMEOUT_SECONDS),
        )
        or _DEFAULT_TIMEOUT_SECONDS
    )
    if max_tokens < 1 or timeout <= 0:
        raise ValueError("RAGAS judge token and timeout values must be positive")
    thinking_name = "RAGAS_JUDGE_DISABLE_THINKING"
    disable_thinking = _parse_bool(
        _first_env(
            thinking_name,
            "LLM_EASY_DISABLE_THINKING",
            "LLM_DISABLE_THINKING",
            default="true",
        )
        or "true",
        thinking_name,
    )
    return ModelProfile(
        model=_judge_model_for(),
        base_url=_first_env(
            "RAGAS_JUDGE_BASE_URL",
            "LLM_EASY_BASE_URL",
            "LLM_BASE_URL",
            "DEEPSEEK_BASE_URL",
            default=_DEFAULT_BASE_URL,
        )
        or _DEFAULT_BASE_URL,
        api_key=key,
        max_tokens=max_tokens,
        timeout=timeout,
        disable_thinking=disable_thinking,
    )


class _ClientWrapper:
    """Inject the selected model and bounded request controls once."""

    def __init__(
        self,
        client,
        model: str,
        max_tokens: int,
        is_async: bool,
        *,
        disable_thinking: bool = True,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._is_async = is_async
        self._disable_thinking = disable_thinking
        self._extra_body = extra_body

    def create(self, **kwargs):
        kwargs.setdefault("model", self._model)
        kwargs.setdefault("max_tokens", self._max_tokens)
        kwargs.setdefault("max_retries", 0)
        if self._extra_body is not None:
            kwargs.setdefault("extra_body", self._extra_body)
        elif self._disable_thinking:
            kwargs.setdefault("extra_body", {"thinking": {"type": "disabled"}})
        if self._is_async:
            return self._acreate(**kwargs)
        return self._client.create(**kwargs)

    async def _acreate(self, **kwargs):
        return await self._client.create(**kwargs)

    async def aclose(self) -> None:
        """Close an owned async instructor client before its event loop ends."""
        if not self._is_async:
            return
        result = self._client.close()
        if inspect.isawaitable(result):
            await result


def get_client(tier: Tier = "easy", *, async_client: bool = False):
    """Return an instructor-wrapped OpenAI-compatible client for ``tier``.

    Async clients are deliberately fresh because they belong to the caller's event
    loop. Sync clients are process-cached.
    """
    import instructor
    from openai import AsyncOpenAI, OpenAI

    profile = _profile_for(tier)
    if async_client:
        raw = instructor.from_openai(
            AsyncOpenAI(
                base_url=profile.base_url,
                api_key=profile.api_key,
                timeout=profile.timeout,
                max_retries=0,
            ),
            mode=instructor.Mode.TOOLS,
        )
        return _ClientWrapper(
            raw,
            profile.model,
            profile.max_tokens,
            is_async=True,
            disable_thinking=profile.disable_thinking,
            extra_body=profile.extra_body,
        )
    if tier not in _SYNC_CLIENTS:
        raw = instructor.from_openai(
            OpenAI(
                base_url=profile.base_url,
                api_key=profile.api_key,
                timeout=profile.timeout,
                max_retries=0,
            ),
            mode=instructor.Mode.TOOLS,
        )
        _SYNC_CLIENTS[tier] = _ClientWrapper(
            raw,
            profile.model,
            profile.max_tokens,
            is_async=False,
            disable_thinking=profile.disable_thinking,
            extra_body=profile.extra_body,
        )
    return _SYNC_CLIENTS[tier]


def load_prompt(name: str) -> str:
    """Load a prompt template from agent/prompts/<name>.txt (stripped)."""
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
