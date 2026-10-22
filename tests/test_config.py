from __future__ import annotations

from autoclip.config import Settings


def test_openai_compat_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compat")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.llm_provider == "openai_compat"
    assert s.llm_base_url == "http://127.0.0.1:9999/v1"
    assert s.llm_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(s)
    assert s.llm_model == "some-model"
    assert s.vision_model is None


def test_llm_defaults_off(monkeypatch):
    for key in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "VISION_MODEL"):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider == "none"
    assert s.llm_api_key is None
