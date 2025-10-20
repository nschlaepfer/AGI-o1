import builtins
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def agi_module(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("REASONING_BANK_ENABLED", "false")
    spec = importlib.util.spec_from_file_location("agi_o1", project_root / "AGI-o1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_task_session_requires_confirm(monkeypatch, agi_module):
    inputs = iter(["hello", ":success", ":confirm"])

    def fake_input(prompt):
        return next(inputs)

    def fake_chat(message, is_initial_response=False, chat=None):
        if chat:
            chat.add_message("assistant", f"response to {message}")
        return f"response to {message}"

    evaluation_payload = {
        "outcome": "success",
        "confidence": 0.95,
        "rationale": "all good",
        "improvements": [],
    }

    class FakeReasoningBank:
        def __init__(self):
            self.calls = []

        def process_interaction(self, **kwargs):
            self.calls.append(kwargs)

    fake_rb = FakeReasoningBank()

    monkeypatch.setattr(builtins, "input", fake_input)
    monkeypatch.setattr(agi_module, "gpt4o_chat", fake_chat)
    monkeypatch.setattr(agi_module, "evaluate_interaction", lambda *a, **k: evaluation_payload)
    original_rb = agi_module.reasoning_bank
    agi_module.reasoning_bank = fake_rb
    try:
        agi_module.run_task_session("Diagnose issue", max_turns=5)
    finally:
        agi_module.reasoning_bank = original_rb

    assert len(fake_rb.calls) == 1
    call = fake_rb.calls[0]
    assert call["outcome"] == "success"
    assert call["metadata"]["evaluation"]["outcome"] == "success"
