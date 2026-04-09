from types import SimpleNamespace

import inference


def test_probe_calls_openai_when_enabled(monkeypatch):
    calls = []

    class DummyResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="OK")

    class DummyClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.responses = DummyResponses()

    monkeypatch.setenv("OPENENV_LLM_PROXY_PROBE", "1")
    monkeypatch.setattr(inference, "OpenAI", DummyClient)

    settings = SimpleNamespace(
        api_base_url="https://router.huggingface.co/v1",
        model_name="Qwen/Qwen2.5-72B-Instruct",
        api_key="test-key",
    )
    inference._probe_litellm_proxy(settings)

    assert len(calls) == 1
    assert calls[0]["model"] == "Qwen/Qwen2.5-72B-Instruct"


def test_probe_skips_when_disabled(monkeypatch):
    calls = []

    class DummyResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text="OK")

    class DummyClient:
        def __init__(self, **kwargs):
            self.responses = DummyResponses()

    monkeypatch.setenv("OPENENV_LLM_PROXY_PROBE", "0")
    monkeypatch.setattr(inference, "OpenAI", DummyClient)

    settings = SimpleNamespace(
        api_base_url="https://router.huggingface.co/v1",
        model_name="Qwen/Qwen2.5-72B-Instruct",
        api_key="test-key",
    )
    inference._probe_litellm_proxy(settings)

    assert calls == []
