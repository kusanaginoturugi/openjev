import math

import pytest

from openjev_phase1.llama_server import LlamaServerClient, _option_probabilities, score


ROW = {
    "id": "mail-1",
    "state": "An invoice mailing address must be changed.",
    "question": "Which team should handle it?",
    "options": [
        {"id": "billing", "description": "Billing."},
        {"id": "sales", "description": "Sales."},
        {"id": "support", "description": "Technical support."},
    ],
}


def test_llama_server_direct_score_uses_equal_slot_bias():
    prompt = "rendered prompt"
    prompt_ids = [10, 11]
    slot_ids = {"A": 41, "B": 42, "C": 43}
    completion_payload = None

    def post(url, payload):
        nonlocal completion_payload
        if url.endswith("/apply-template"):
            return {"prompt": prompt}
        if url.endswith("/tokenize"):
            content = payload["content"]
            if content == prompt:
                return {"tokens": prompt_ids}
            return {"tokens": prompt_ids + [slot_ids[content[-1]]]}
        if url.endswith("/completion"):
            completion_payload = payload
            return {
                "completion_probabilities": [
                    {"top_probs": [{"id": 41, "prob": 0.5}, {"id": 42, "prob": 0.3}, {"id": 43, "prob": 0.2}]}
                ]
            }
        raise AssertionError(url)

    result = score(LlamaServerClient("http://server.test", "model", post), ROW)
    assert result["probabilities"] == pytest.approx([0.5, 0.3, 0.2])
    assert result["option_token_ids"] == [41, 42, 43]
    assert completion_payload["logit_bias"] == {"41": 100, "42": 100, "43": 100}
    assert completion_payload["post_sampling_probs"] is True
    assert completion_payload["samplers"] == ["temperature"]


def test_llama_server_rejects_missing_option_probability():
    with pytest.raises(RuntimeError, match="every declared option slot"):
        _option_probabilities({"completion_probabilities": [{"top_probs": [{"id": 41, "prob": 1.0}]}]}, [41, 42])


def test_llama_server_probabilities_are_normalized():
    values = _option_probabilities(
        {"completion_probabilities": [{"top_probs": [{"id": 41, "prob": 0.25}, {"id": 42, "prob": 0.5}]}]},
        [41, 42],
    )
    assert values == pytest.approx([1 / 3, 2 / 3])
    assert math.isclose(sum(values), 1.0)
