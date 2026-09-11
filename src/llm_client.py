"""
src/llm_client.py

Thin wrapper around the Gemini API (current google-genai SDK).

Setup:
    1. Get an API key from https://aistudio.google.com/apikey
    2. Set it as an environment variable:
         Windows (PowerShell):  $env:GEMINI_API_KEY = "your-key-here"
         Mac/Linux:             export GEMINI_API_KEY="your-key-here"
"""
import os
import time
import traceback
from google import genai
from google.genai import types

_MODEL_NAME = "gemini-3.1-flash-lite"
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a key at https://aistudio.google.com/apikey and set it before running."
        )
    _client = genai.Client(api_key=api_key)
    return _client


def _describe_error(e: Exception) -> str:
    parts = [f"{type(e).__name__}: {e}"]
    for attr in ("code", "message", "status", "response"):
        if hasattr(e, attr):
            parts.append(f"{attr}={getattr(e, attr)!r}")
    cause = e.__cause__ or e.__context__
    if cause is not None and cause is not e:
        parts.append(f"caused by -> {_describe_error(cause)}")
    if hasattr(e, "exceptions"):
        for sub in e.exceptions:
            parts.append(f"sub-exception -> {_describe_error(sub)}")
    return " | ".join(parts)


def call_llm(prompt: str, model: str = _MODEL_NAME, retries: int = 3, temperature: float = 0.2) -> str:
    client = _get_client()
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return resp.text.strip()
        except Exception as e:
            last_err = e
            detail = _describe_error(e)
            print(f"  [llm_client] error: {detail}")
            wait = 2 ** attempt
            print(f"  [llm_client] retrying in {wait}s...")
            time.sleep(wait)
    print("  [llm_client] full traceback of final failure:")
    traceback.print_exception(type(last_err), last_err, last_err.__traceback__)
    raise RuntimeError(f"Gemini call failed after {retries} retries: {_describe_error(last_err)}")