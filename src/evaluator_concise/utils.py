# utils.py
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple, Type

import openai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load .env from configs directory
_env_path = Path(__file__).resolve().parent.parent.parent / "configs" / ".env"
load_dotenv(_env_path)


def num_tokens_from_messages(messages, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            if value is None:
                continue
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens -= 1
    num_tokens += 2
    return num_tokens


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 10,
    errors: Tuple = (openai.RateLimitError,),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                delay *= exponential_base * (1 + (random.random() if jitter else 0))
                print(f"  [retry {num_retries}/{max_retries}] {type(e).__name__}: {e} — retrying in {delay:.1f}s...")
                time.sleep(delay)
            except Exception as e:
                raise e

    return wrapper


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Get an OpenAI client using OPENAI_API_KEY from environment or the provided key."""
    return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))


get_api_client = get_openai_client


@retry_with_exponential_backoff
def chat_completions_parsed_with_backoff(client: OpenAI, **kwargs):
    return client.beta.chat.completions.parse(**kwargs)


def send_to_openai_parsed(
    *,
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    response_format: Type[BaseModel],
    model: str = "gpt-4o-mini",
    token_limit: Optional[int] = None,
    safety_margin: int = 200,
    temperature: float = 0.8,
    top_p: float = 1.0,
    return_usage: bool = False,
):
    """
    Send a request to OpenAI with structured output parsing.

    Args:
        return_usage: If True, returns (parsed_response, usage_dict) tuple.
                      usage_dict contains 'prompt_tokens' and 'completion_tokens'.
    """
    if token_limit is None:
        token_limit = 32000 if ("gpt-4" in model or "gpt-5" in model) else 16000

    # GPT-4 and similar models have a max completion token limit of 16384
    max_completion_limit = 16384

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_tokens = num_tokens_from_messages(conversation)
    max_possible_response = min(token_limit - prompt_tokens - safety_margin, max_completion_limit)
    if max_possible_response <= 0:
        raise ValueError(
            f"Prompt too long for context. prompt_tokens={prompt_tokens}, token_limit={token_limit}"
        )

    completion = chat_completions_parsed_with_backoff(
        client=client,
        model=model,
        messages=conversation,
        response_format=response_format,
        max_completion_tokens=max_possible_response,
        temperature=temperature,
        top_p=top_p,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        # Parsing failed - raise with details about the refusal or raw content
        msg = completion.choices[0].message
        if msg.refusal:
            raise ValueError(f"Model refused to respond: {msg.refusal}")
        raise ValueError(f"Failed to parse response. Raw content: {msg.content[:200] if msg.content else 'None'}...")

    # Ensure we got a Pydantic model, not a raw string
    if isinstance(parsed, str):
        raise ValueError(f"Expected Pydantic model but got string. Content: {parsed[:200]}...")

    if return_usage:
        usage = {
            "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
            "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
        }
        return parsed, usage

    return parsed