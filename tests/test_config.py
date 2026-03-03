"""
Tests for playlistgen/config.py
Covers: load_config, save_config, caching, env var overrides
"""

import importlib
import os
import pytest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _reset_cache():
    """Reset the module-level config cache between tests."""
    import playlistgen.config as cfg_mod
    cfg_mod._config_cache = None
    cfg_mod._config_path = None


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Automatically reset config cache before and after every test."""
    _reset_cache()
    yield
    _reset_cache()


# ---------------------------------------------------------------------------
# load_config — defaults
# ---------------------------------------------------------------------------


class TestLoadConfigDefaults:
    def test_returns_dict(self, tmp_path):
        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert isinstance(cfg, dict)

    def test_contains_required_keys(self, tmp_path):
        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        for key in (
            "ITUNES_JSON",
            "CLUSTER_COUNT",
            "MAX_PER_ARTIST",
            "TRACKS_PER_MIX",
            "OUTPUT_DIR",
        ):
            assert key in cfg, f"Missing key: {key}"

    def test_default_cluster_count(self, tmp_path):
        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["CLUSTER_COUNT"] == 6

    def test_default_tracks_per_mix(self, tmp_path):
        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["TRACKS_PER_MIX"] == 50

    def test_default_max_per_artist(self, tmp_path):
        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["MAX_PER_ARTIST"] == 4

    def test_default_ai_enhance_false(self, tmp_path):
        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["AI_ENHANCE"] is False


# ---------------------------------------------------------------------------
# load_config — user config file merging
# ---------------------------------------------------------------------------


class TestLoadConfigMerge:
    def test_user_overrides_default(self, tmp_path):
        import yaml
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.safe_dump({"CLUSTER_COUNT": 12}))

        from playlistgen.config import load_config
        cfg = load_config(path=str(config_file))
        assert cfg["CLUSTER_COUNT"] == 12

    def test_user_config_merges_not_replaces(self, tmp_path):
        import yaml
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.safe_dump({"CLUSTER_COUNT": 10}))

        from playlistgen.config import load_config
        cfg = load_config(path=str(config_file))
        # Keys not in user config should still have defaults
        assert "MAX_PER_ARTIST" in cfg
        assert "OUTPUT_DIR" in cfg

    def test_user_can_add_extra_keys(self, tmp_path):
        import yaml
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.safe_dump({"MY_CUSTOM_KEY": "hello"}))

        from playlistgen.config import load_config
        cfg = load_config(path=str(config_file))
        assert cfg.get("MY_CUSTOM_KEY") == "hello"

    def test_empty_yaml_uses_defaults(self, tmp_path):
        config_file = tmp_path / "config.yml"
        config_file.write_text("")  # empty file

        from playlistgen.config import load_config
        cfg = load_config(path=str(config_file))
        assert cfg["CLUSTER_COUNT"] == 6


# ---------------------------------------------------------------------------
# load_config — environment variable overrides
# ---------------------------------------------------------------------------


class TestLoadConfigEnvVars:
    def test_env_var_overrides_string(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ITUNES_JSON", "/env/override/itunes.json")

        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["ITUNES_JSON"] == "/env/override/itunes.json"

    def test_env_var_overrides_int(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLUSTER_COUNT", "20")

        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["CLUSTER_COUNT"] == 20
        assert isinstance(cfg["CLUSTER_COUNT"], int)

    def test_env_var_overrides_bool_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_ENHANCE", "true")

        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["AI_ENHANCE"] is True

    def test_env_var_overrides_bool_false(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_ENHANCE", "0")

        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["AI_ENHANCE"] is False

    def test_env_var_bool_yes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YEAR_MIX_ENABLED", "yes")

        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["YEAR_MIX_ENABLED"] is True

    def test_env_var_bool_1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUTAGEN_ENABLED", "1")

        from playlistgen.config import load_config
        cfg = load_config(path=str(tmp_path / "nonexistent.yml"))
        assert cfg["MUTAGEN_ENABLED"] is True

    def test_env_var_takes_priority_over_file(self, tmp_path, monkeypatch):
        import yaml
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.safe_dump({"CLUSTER_COUNT": 8}))
        monkeypatch.setenv("CLUSTER_COUNT", "25")

        from playlistgen.config import load_config
        cfg = load_config(path=str(config_file))
        assert cfg["CLUSTER_COUNT"] == 25


# ---------------------------------------------------------------------------
# Config caching behaviour
# ---------------------------------------------------------------------------


class TestLoadConfigCaching:
    def test_same_object_returned_when_cached(self, tmp_path):
        from playlistgen.config import load_config
        no_file = str(tmp_path / "nonexistent.yml")
        cfg1 = load_config(path=no_file)
        # Second call with path=None should return cache
        cfg2 = load_config(path=None)
        assert cfg1 is cfg2

    def test_explicit_path_bypasses_cache(self, tmp_path):
        import yaml
        from playlistgen.config import load_config

        # Prime the cache
        cfg1 = load_config(path=str(tmp_path / "nonexistent.yml"))

        # Write a different file
        config_file = tmp_path / "other.yml"
        config_file.write_text(yaml.safe_dump({"CLUSTER_COUNT": 99}))

        # Explicit path should reload
        cfg2 = load_config(path=str(config_file))
        assert cfg2["CLUSTER_COUNT"] == 99


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_saves_yaml_file(self, tmp_path):
        import yaml
        from playlistgen.config import save_config

        out = tmp_path / "saved.yml"
        save_config({"CLUSTER_COUNT": 7, "OUTPUT_DIR": "./mixes"}, path=str(out))
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        assert data["CLUSTER_COUNT"] == 7

    def test_creates_parent_directories(self, tmp_path):
        from playlistgen.config import save_config

        nested = tmp_path / "a" / "b" / "config.yml"
        save_config({"CLUSTER_COUNT": 3}, path=str(nested))
        assert nested.exists()

    def test_sensitive_key_anthropic_api_key_nullified(self, tmp_path):
        import yaml
        from playlistgen.config import save_config

        out = tmp_path / "config.yml"
        save_config(
            {"ANTHROPIC_API_KEY": "sk-secret-123", "CLUSTER_COUNT": 5},
            path=str(out),
        )
        data = yaml.safe_load(out.read_text())
        # Should be written as None, not the real key
        assert data.get("ANTHROPIC_API_KEY") is None

    def test_sensitive_key_lastfm_api_key_nullified(self, tmp_path):
        import yaml
        from playlistgen.config import save_config

        out = tmp_path / "config.yml"
        save_config(
            {"LASTFM_API_KEY": "my_lastfm_key", "CLUSTER_COUNT": 5},
            path=str(out),
        )
        data = yaml.safe_load(out.read_text())
        assert data.get("LASTFM_API_KEY") is None

    def test_none_sensitive_key_stays_none(self, tmp_path):
        import yaml
        from playlistgen.config import save_config

        out = tmp_path / "config.yml"
        save_config(
            {"ANTHROPIC_API_KEY": None, "CLUSTER_COUNT": 5},
            path=str(out),
        )
        data = yaml.safe_load(out.read_text())
        assert data.get("ANTHROPIC_API_KEY") is None

    def test_does_not_mutate_original_dict(self, tmp_path):
        from playlistgen.config import save_config

        original = {"ANTHROPIC_API_KEY": "real-key", "CLUSTER_COUNT": 6}
        save_config(original, path=str(tmp_path / "config.yml"))
        # The original dict should still have the real key
        assert original["ANTHROPIC_API_KEY"] == "real-key"

    def test_non_sensitive_keys_preserved(self, tmp_path):
        import yaml
        from playlistgen.config import save_config

        out = tmp_path / "config.yml"
        save_config({"OUTPUT_DIR": "./my_mixes", "CLUSTER_COUNT": 9}, path=str(out))
        data = yaml.safe_load(out.read_text())
        assert data["OUTPUT_DIR"] == "./my_mixes"
        assert data["CLUSTER_COUNT"] == 9
