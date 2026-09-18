"""Direct option scoring through a local llama-server GGUF model."""

from __future__ import annotations

import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .core import LETTERS, digest, direct_messages, softmax

PROMPT_VERSION = "llama-server-direct-options-v1"


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"llama-server request to {url} failed: {error}") from error


class LlamaServerClient:
    """Minimal client for the llama.cpp endpoints needed by direct scoring."""

    def __init__(self, server: str, model: str, post=_post_json):
        self.server = server.rstrip("/")
        if not self.server.startswith(("http://", "https://")):
            raise ValueError("llama-server URL must start with http:// or https://")
        self.model = model
        self._post = post

    def _request(self, endpoint: str, payload: dict) -> dict:
        return self._post(f"{self.server}{endpoint}", {"model": self.model, **payload})

    def apply_template(self, messages: list[dict]) -> str:
        response = self._request("/apply-template", {"messages": messages})
        prompt = response.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError("llama-server did not return a rendered prompt")
        return prompt

    def tokenize(self, content: str) -> list[int]:
        response = self._request("/tokenize", {"content": content})
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise RuntimeError("llama-server did not return integer token IDs")
        return tokens

    def completion(self, prompt: str, slots: list[int]) -> dict:
        # Equal bias makes every declared slot survive top-N filtering.  The
        # server returns probabilities after that bias; normalizing over just
        # these equally shifted slots preserves their relative model scores.
        return self._request(
            "/completion",
            {
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": max(10, len(slots)),
                "temperature": 1,
                "samplers": ["temperature"],
                "post_sampling_probs": True,
                "logit_bias": {str(slot): 100 for slot in slots},
                "cache_prompt": False,
            },
        )


def _encode_prompt(client: LlamaServerClient, row: dict, max_tokens: int) -> tuple[str, list[int], str, int]:
    prompt = client.apply_template(direct_messages(row))
    ids = client.tokenize(prompt)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = []
    for letter in LETTERS[: len(row["options"])]:
        extended = client.tokenize(prompt + letter)
        suffix = extended[len(ids) :]
        if extended[: len(ids)] != ids or len(suffix) != 1:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
        slots.append(suffix[0])
    if len(slots) != len(set(slots)):
        raise ValueError("Answer-slot tokens collide")
    return prompt, slots, digest(prompt), len(ids)


def _option_probabilities(response: dict, slots: list[int]) -> list[float]:
    positions = response.get("completion_probabilities")
    if not isinstance(positions, list) or len(positions) != 1:
        raise RuntimeError("llama-server did not return one completion probability position")
    top = positions[0].get("top_probs") if isinstance(positions[0], dict) else None
    if not isinstance(top, list):
        raise RuntimeError("llama-server did not return post-sampling top probabilities")
    values = {entry.get("id"): entry.get("prob") for entry in top if isinstance(entry, dict)}
    try:
        selected = [float(values[slot]) for slot in slots]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("llama-server did not return every declared option slot") from error
    if any(value <= 0 or not math.isfinite(value) for value in selected):
        raise RuntimeError("llama-server returned non-positive or non-finite option probabilities")
    return softmax([math.log(value) for value in selected])


def score(client: LlamaServerClient, row: dict, max_tokens: int = 4096) -> dict:
    """Score one decision using a quantized llama-server model."""
    started = time.perf_counter()
    prompt, slots, prompt_hash, input_tokens = _encode_prompt(client, row, max_tokens)
    forward_start = time.perf_counter()
    response = client.completion(prompt, slots)
    forward_seconds = time.perf_counter() - forward_start
    probabilities = _option_probabilities(response, slots)
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": probabilities,
        "option_token_ids": slots,
        "input_tokens": input_tokens,
        "forward_seconds": forward_seconds,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": {
            "source": client.model,
            "backend": "llama-server",
            "server": client.server,
            "format": "GGUF (server-selected quantization)",
        },
        "readout": "llama-server post-sampling option probabilities after equal logit bias",
        "probability_status": "conditional option score; uncalibrated; GGUF/backend-dependent",
    }
