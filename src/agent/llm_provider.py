"""
LLM runtime / provider adapter — LiteLLM only.

Normalises every provider response into an ``LLMResponse`` object so the
rest of the agent system never touches provider-specific types. LiteLLM
handles OpenAI, Anthropic, and other providers under one unified API.
"""

import json
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import litellm
import openai

litellm.drop_params = True  # silently ignore params unsupported by a given provider


@dataclass
class LLMResponse:
    """Provider-agnostic representation of one LLM completion."""

    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @classmethod
    def from_openai(cls, response: Any) -> "LLMResponse":
        """Build from an OpenAI ``ChatCompletion`` or LiteLLM ``ModelResponse`` object."""
        if not response.choices:
            raise ValueError(
                f"LLM returned no choices (choices={response.choices!r}). "
                f"Model: {getattr(response, 'model', 'unknown')}. "
                f"Full response: {response}"
            )
        choice = response.choices[0]
        content = choice.message.content

        tool_calls = None
        if getattr(choice.message, "tool_calls", None):
            tool_calls = []
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append({
                    "tool_name": tc.function.name,
                    "tool_call_id": tc.id,
                    "arguments": args,
                })

        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0),
        }

        return cls(
            content=content,
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            usage=usage,
        )


class LLMProvider:
    """Unified LLM interface backed by LiteLLM."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2500,
        top_p: float = 1.0,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        return self._call_with_retry(kwargs)

    def call_text_only(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
        }
        return self._call_with_retry(kwargs)

    _RATE_LIMIT_BASE_DELAY: float = 10.0

    def _call_with_retry(self, kwargs: Dict[str, Any]) -> LLMResponse:
        delay = self.initial_retry_delay
        rl_delay = self._RATE_LIMIT_BASE_DELAY
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._call_litellm(kwargs)
                self._record_usage(result.usage)
                return result
            except Exception as e:
                last_exc = e
                print(f"  [LLMProvider] Attempt {attempt + 1}/{self.max_retries + 1} failed: {type(e).__name__}: {e}")
                if "HTTP 400" in str(e):
                    break
                if attempt < self.max_retries:
                    if isinstance(e, (openai.RateLimitError, litellm.RateLimitError)):
                        wait = rl_delay + random.uniform(0, rl_delay * 0.2)
                        print(f"  [LLMProvider] Rate limit hit — waiting {wait:.0f}s before retry...")
                        time.sleep(wait)
                        rl_delay *= 2
                    else:
                        time.sleep(delay + random.uniform(0, delay * 0.5))
                        delay *= 2
        raise RuntimeError(f"LLM call failed after all attempts. Last error: {last_exc}")

    _ANTHROPIC_MAX_TOKENS: int = 4096
    _REASONING_MIN_TOKENS: int = 16384

    def _call_litellm(self, kwargs: Dict[str, Any]) -> LLMResponse:
        model: str = kwargs["model"]
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        is_anthropic = model.startswith("anthropic/") or model.startswith("claude-")
        if is_anthropic:
            max_tokens = min(max_tokens, self._ANTHROPIC_MAX_TOKENS)

        is_reasoning = False
        try:
            is_reasoning = bool(litellm.utils.supports_reasoning(model))
        except Exception:
            pass

        if is_reasoning:
            max_tokens = max(max_tokens, self._REASONING_MIN_TOKENS)
            call_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": kwargs["messages"],
                "max_tokens": max_tokens,
            }
        else:
            call_kwargs = {
                "model": model,
                "messages": kwargs["messages"],
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": max_tokens,
                "top_p": kwargs.get("top_p", self.top_p),
            }
        if "tools" in kwargs:
            call_kwargs["tools"] = kwargs["tools"]
            call_kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")

        response = litellm.completion(**call_kwargs)
        return LLMResponse.from_openai(response)

    def _record_usage(self, usage: Dict[str, int]) -> None:
        self.total_input_tokens += usage.get("input_tokens", 0)
        self.total_output_tokens += usage.get("output_tokens", 0)
        self.total_requests += 1
