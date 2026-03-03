"""
Ollama & Claude AI backend for PlaylistGen.

When OLLAMA_BASE_URL is set → uses local Ollama at {base_url}/v1/chat/completions
Otherwise → falls back to Anthropic API if ANTHROPIC_API_KEY is configured
"""

import json
import logging
import os
from typing import Tuple, Optional
try:
    import requests
except ImportError:
    raise RuntimeError("requests library required for Ollama HTTP calls")

def _call_ollama(
    prompt: str,
    model: str,
    base_url: str
) -> Tuple[str, int]:
    endpoint = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        resp = requests.post(endpoint, json=payload)
        if not resp.ok:
            logging.error("Ollama API HTTP %d: %s", resp.status_code, resp.text[:500])
            return prompt, 0
    except Exception as e:
        logging.warning("Ollama connection failed: %s", str(e)[:80])
        return prompt, 4

    try:
        data = resp.json()
        choices_list = data.get("choices") or []
        if not choices_list:
            logging.warning("Ollama returned empty response")
            return prompt, 4
        raw = (choices_list[0].get("message") or {}).get("content", "")
    except Exception as e:
        logging.error("Failed to parse Ollama response: %s", str(e)[:60])
        return prompt, 4

    raw = raw.strip()
    if raw.startswith("`"+chr(96)+"json"):
        start = len("`"+chr(96)+"json")
    elif raw.startswith("`"+chr(96)):
        start = len("```")
    else:
        start = 0
    if start:
        try:
            data_out = json.loads(raw[start:])
        except Exception:
            return prompt, 4
    else:
        try:
            data_out = json.loads(raw)
        except Exception:
            logging.debug("Not valid JSON from Ollama")
            return prompt, 4

    name = (data_out.get("name") or str(data_out)).strip() or ""
    try:
        cohesion = int(data_out.get("cohesion", 5))
    except Exception:
        cohesion = 4 if not data_out else 5

    logging.info("Ollama named it: '%s' (cohesion %d)", name, cohesion)
    return name, cohesion


def _call_llm(
    prompt,
    model="qwen35-tuned",
    base_url=None,
    api_key=None
) -> Tuple[str, int]:
    """
    AI dispatcher:
      - If OLLAMA_BASE_URL is set → use local Ollama backend at {base_url}/v1/chat/completions
      - Else if ANTHROPIC_API_KEY in config/env → use Claude Anthropic API  
      - Else raise error with setup instructions for both options
    """
    from .config import load_config
    
    ollama_base_url = base_url or os.getenv("OLLAMA_BASE_URL")
    if ollama_base_url:
        return _call_ollama(prompt, model, ollama_base_url)
    
    cfg: dict = load_config()
    anthropic_api_key: str = api_key or os.getenv("ANTHROPIC_API_KEY") or cfg.get("ANTHROPIC_API_KEY", "")

    if not anthropic_api_key:
        raise RuntimeError(
            "No AI backend configured. Either:\n"
            "  - Set OLLAMA_BASE_URL=http://localhost:11434 to use a local model, or\n"
            "  - Set ANTHROPIC_API_KEY to use the Claude API."
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        system = (
            "You are an expert music playlist curator. "
            "Given a summary of tracks in a playlist cluster, produce:\n"
            "1. A creative, evocative playlist name (max 5 words, no quotes).\n"
            "2. A cohesion score from 1 (incoherent mix) to 10 (perfectly themed).\n\n"
            'Reply with ONLY a JSON object: {"name": "...", "cohesion": N}'
        )
        client_msg = client.messages.create(
            model=model,
            max_tokens=64,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = client_msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        name = str(data.get("name", "")).strip()
        cohesion = int(data.get("cohesion", 5))
        return name, cohesion
    except Exception as exc:
        logging.debug("Claude API call failed: %s", exc)
        return prompt, 0
