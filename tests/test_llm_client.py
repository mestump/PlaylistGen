"""Tests for playlistgen.llm_client — AI dispatcher (Ollama / Claude)."""

import json
from unittest.mock import MagicMock, patch

import pytest


def _ollama_resp(content):
    """Helper: build a mock Ollama HTTP response with given content string."""
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


# ---------------------------------------------------------------------------
# _call_ollama tests
# ---------------------------------------------------------------------------


class TestCallOllama:
    def test_successful_json_response(self):
        from playlistgen.llm_client import _call_ollama

        resp = _ollama_resp(json.dumps({"name": "Midnight Groove", "cohesion": 8}))
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("test prompt", "llama3", "http://localhost:11434")
        assert name == "Midnight Groove"
        assert cohesion == 8

    def test_markdown_fenced_json(self):
        from playlistgen.llm_client import _call_ollama

        raw = '```json\n{"name": "Sunset Vibes", "cohesion": 7}\n```'
        resp = _ollama_resp(raw)
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("test", "model", "http://localhost:11434")
        assert name == "Sunset Vibes"
        assert cohesion == 7

    def test_plain_backtick_fence(self):
        from playlistgen.llm_client import _call_ollama

        raw = '```\n{"name": "Chill Wave", "cohesion": 6}\n```'
        resp = _ollama_resp(raw)
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("test", "model", "http://localhost:11434")
        assert name == "Chill Wave"
        assert cohesion == 6

    def test_missing_cohesion_defaults_to_5(self):
        from playlistgen.llm_client import _call_ollama

        resp = _ollama_resp(json.dumps({"name": "No Score"}))
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("test", "model", "http://localhost:11434")
        assert name == "No Score"
        assert cohesion == 5

    def test_missing_name_falls_back_to_str(self):
        from playlistgen.llm_client import _call_ollama

        resp = _ollama_resp(json.dumps({"cohesion": 3}))
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("test", "model", "http://localhost:11434")
        # name should be str representation of the dict (no empty string)
        assert name != ""
        assert cohesion == 3

    def test_http_error_returns_prompt(self):
        from playlistgen.llm_client import _call_ollama

        resp = MagicMock()
        resp.ok = False
        resp.status_code = 500
        resp.text = "Internal Server Error"
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("my prompt", "model", "http://localhost:11434")
        assert name == "my prompt"
        assert cohesion == 0

    def test_connection_error_returns_prompt(self):
        from playlistgen.llm_client import _call_ollama

        with patch(
            "playlistgen.llm_client.requests.post",
            side_effect=ConnectionError("refused"),
        ):
            name, cohesion = _call_ollama("my prompt", "model", "http://localhost:11434")
        assert name == "my prompt"
        assert cohesion == 4

    def test_timeout_error_returns_prompt(self):
        from playlistgen.llm_client import _call_ollama
        import requests as req

        with patch(
            "playlistgen.llm_client.requests.post",
            side_effect=req.exceptions.ReadTimeout("timed out"),
        ):
            name, cohesion = _call_ollama("my prompt", "model", "http://localhost:11434")
        assert name == "my prompt"
        assert cohesion == 4

    def test_empty_choices(self):
        from playlistgen.llm_client import _call_ollama

        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {"choices": []}
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("prompt", "model", "http://localhost:11434")
        assert name == "prompt"
        assert cohesion == 4

    def test_none_choices_key(self):
        from playlistgen.llm_client import _call_ollama

        resp = MagicMock()
        resp.ok = True
        resp.json.return_value = {"choices": None}
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("prompt", "model", "http://localhost:11434")
        assert name == "prompt"
        assert cohesion == 4

    def test_invalid_json_content(self):
        from playlistgen.llm_client import _call_ollama

        resp = _ollama_resp("not json at all")
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("prompt", "model", "http://localhost:11434")
        assert name == "prompt"
        assert cohesion == 4

    def test_invalid_json_inside_fence(self):
        from playlistgen.llm_client import _call_ollama

        resp = _ollama_resp('```json\n{broken json\n```')
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("prompt", "model", "http://localhost:11434")
        assert name == "prompt"
        assert cohesion == 4

    def test_empty_content_string(self):
        from playlistgen.llm_client import _call_ollama

        resp = _ollama_resp("")
        with patch("playlistgen.llm_client.requests.post", return_value=resp):
            name, cohesion = _call_ollama("prompt", "model", "http://localhost:11434")
        assert name == "prompt"
        assert cohesion == 4

    def test_posts_correct_payload(self):
        from playlistgen.llm_client import _call_ollama

        resp = _ollama_resp(json.dumps({"name": "X", "cohesion": 1}))
        with patch("playlistgen.llm_client.requests.post", return_value=resp) as mock_post:
            _call_ollama("my prompt", "llama3", "http://myhost:1234")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://myhost:1234/v1/chat/completions"
        assert kwargs["json"]["model"] == "llama3"
        assert kwargs["json"]["messages"][0]["content"] == "my prompt"


# ---------------------------------------------------------------------------
# _call_llm dispatcher tests
# ---------------------------------------------------------------------------


class TestCallLlm:
    def test_dispatches_to_ollama_when_base_url_set(self):
        from playlistgen.llm_client import _call_llm

        resp = _ollama_resp(json.dumps({"name": "Test", "cohesion": 5}))
        with patch("playlistgen.llm_client.requests.post", return_value=resp) as mock_post:
            name, cohesion = _call_llm("prompt", base_url="http://localhost:11434")
        assert mock_post.called
        assert name == "Test"

    def test_ollama_env_var_used_when_no_base_url_arg(self):
        from playlistgen.llm_client import _call_llm

        resp = _ollama_resp(json.dumps({"name": "EnvTest", "cohesion": 6}))
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434"}):
            with patch("playlistgen.llm_client.requests.post", return_value=resp):
                name, cohesion = _call_llm("prompt")
        assert name == "EnvTest"
        assert cohesion == 6

    def test_invalid_ollama_url_returns_gracefully(self):
        from playlistgen.llm_client import _call_llm

        name, cohesion = _call_llm("prompt", base_url="ftp://invalid")
        assert name == "prompt"
        assert cohesion == 0

    def test_raises_when_no_backend_configured(self):
        from playlistgen.llm_client import _call_llm

        with patch.dict("os.environ", {}, clear=True):
            with patch("playlistgen.config.load_config", return_value={}):
                with pytest.raises(RuntimeError, match="No AI backend configured"):
                    _call_llm("prompt", base_url=None, api_key=None)

    def test_dispatches_to_claude_when_api_key_set(self):
        from playlistgen.llm_client import _call_llm

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"name": "Claude Mix", "cohesion": 9}')]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with patch.dict("os.environ", {"OLLAMA_BASE_URL": ""}, clear=False):
            with patch("playlistgen.config.load_config", return_value={}):
                with patch("anthropic.Anthropic", return_value=mock_client):
                    name, cohesion = _call_llm("prompt", api_key="sk-test-123")
        assert name == "Claude Mix"
        assert cohesion == 9

    def test_claude_markdown_fenced_response(self):
        from playlistgen.llm_client import _call_llm

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='```json\n{"name": "Fenced", "cohesion": 4}\n```')]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with patch.dict("os.environ", {"OLLAMA_BASE_URL": ""}, clear=False):
            with patch("playlistgen.config.load_config", return_value={}):
                with patch("anthropic.Anthropic", return_value=mock_client):
                    name, cohesion = _call_llm("prompt", api_key="sk-test-123")
        assert name == "Fenced"
        assert cohesion == 4

    def test_claude_api_error_returns_prompt(self):
        from playlistgen.llm_client import _call_llm

        with patch.dict("os.environ", {"OLLAMA_BASE_URL": ""}, clear=False):
            with patch("playlistgen.config.load_config", return_value={}):
                with patch("anthropic.Anthropic", side_effect=Exception("API error")):
                    name, cohesion = _call_llm("prompt", api_key="sk-test-123")
        assert name == "prompt"
        assert cohesion == 0

    def test_api_key_from_config_file(self):
        from playlistgen.llm_client import _call_llm

        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text='{"name": "Config Key", "cohesion": 7}')]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg

        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "playlistgen.config.load_config",
                return_value={"ANTHROPIC_API_KEY": "sk-from-config"},
            ):
                with patch("anthropic.Anthropic", return_value=mock_client) as mock_cls:
                    name, cohesion = _call_llm("prompt", base_url=None, api_key=None)
        mock_cls.assert_called_once_with(api_key="sk-from-config")
        assert name == "Config Key"
