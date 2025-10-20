import json
import types
from pathlib import Path

from reasoning_bank import ReasoningBank


class _FakeResponse:
    def __init__(self, output, output_text):
        self.output = output
        self.output_text = output_text


class _FakeClient:
    def __init__(self, response):
        self._response = response

        def _create_embedding(model, input):
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
            )

        self.embeddings = types.SimpleNamespace(create=_create_embedding)
        self.responses = types.SimpleNamespace(create=self._return_response)

    def _return_response(self, *args, **kwargs):
        return self._response


def _make_structured_response():
    payload = {
        "confidence": 0.9,
        "outcome": "success",
        "memories": [
            {
                "title": "Title",
                "description": "Desc",
                "content": "Useful steps",
                "outcome": "success",
                "tags": ["tag1"],
                "source": "unit-test",
            }
        ],
    }
    output = [{"content": [{"type": "json", "json": payload}]}]
    return _FakeResponse(output, json.dumps(payload))


def _make_fallback_response():
    payload = {
        "confidence": 0.8,
        "outcome": "failure",
        "memories": [
            {
                "title": "Failure lesson",
                "description": "Why it failed",
                "content": "Avoid this pattern",
                "tags": ["guardrail"],
            }
        ],
    }
    return _FakeResponse([], json.dumps(payload))


def test_process_interaction_uses_structured_json(tmp_path):
    response = _make_structured_response()
    client = _FakeClient(response)
    bank_path = tmp_path / "bank.json"
    bank = ReasoningBank(
        client=client,
        storage_path=bank_path,
        embedding_model="embedding-model",
        reasoning_model="reasoning-model",
    )

    bank.process_interaction("Test", "Answer", "Transcript")
    stats = bank.stats()

    assert stats["count"] == 1
    stored = json.loads(bank_path.read_text())["memories"]
    assert stored[0]["title"] == "Title"
    assert stored[0]["outcome"] == "success"
    assert stored[0]["metadata"]["task"] == "Test"


def test_process_interaction_fallback_to_output_text(tmp_path):
    response = _make_fallback_response()
    client = _FakeClient(response)
    bank_path = tmp_path / "bank.json"
    bank = ReasoningBank(
        client=client,
        storage_path=bank_path,
        embedding_model="embedding-model",
        reasoning_model="reasoning-model",
    )

    bank.process_interaction("Test", "Answer", "Transcript")
    stored = json.loads(bank_path.read_text())["memories"]
    assert stored[0]["outcome"] == "failure"
    assert stored[0]["tags"] == ["guardrail"]
