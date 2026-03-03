"""
Tests for playlistgen/utils.py
Covers: sanitize_label, validate_path, validate_url, validate_config
"""

import os
import pytest
from pathlib import Path

from playlistgen.utils import (
    sanitize_label,
    validate_path,
    validate_url,
    validate_config,
)


# ---------------------------------------------------------------------------
# sanitize_label
# ---------------------------------------------------------------------------


class TestSanitizeLabel:
    def test_plain_string_unchanged(self):
        assert sanitize_label("My Playlist") == "My Playlist"

    def test_forward_slash_replaced(self):
        result = sanitize_label("AC/DC")
        assert "/" not in result
        assert "AC" in result
        assert "DC" in result

    def test_backslash_replaced(self):
        result = sanitize_label("AC\\DC")
        assert "\\" not in result

    def test_slash_replaced_with_dash(self):
        # The source replaces / and \ with ' - '
        assert sanitize_label("A/B") == "A - B"

    def test_illegal_chars_removed(self):
        # '<>:"|?*' should all be stripped
        for ch in '<>:"|?*':
            result = sanitize_label(f"label{ch}name")
            assert ch not in result

    def test_leading_trailing_whitespace_stripped(self):
        assert sanitize_label("  hello world  ") == "hello world"

    def test_internal_whitespace_collapsed(self):
        assert sanitize_label("hello   world") == "hello world"

    def test_trailing_ampersand_stripped(self):
        result = sanitize_label("Rock &")
        assert not result.endswith("&")

    def test_empty_string(self):
        # sanitize_label of empty string should return an empty string
        result = sanitize_label("")
        assert result == ""

    def test_only_illegal_chars(self):
        result = sanitize_label('<>:"|?*')
        assert result == ""

    def test_combined_slash_and_illegal(self):
        result = sanitize_label('A/B<C')
        assert "/" not in result
        assert "<" not in result

    def test_unicode_letters_preserved(self):
        label = "Café del Mar"
        result = sanitize_label(label)
        assert "Café" in result
        assert "Mar" in result

    def test_multiple_slashes(self):
        result = sanitize_label("A/B/C")
        assert "/" not in result

    def test_tab_and_newline_collapsed(self):
        result = sanitize_label("hello\tworld\nnewline")
        # split() handles tabs and newlines
        assert "\t" not in result
        assert "\n" not in result


# ---------------------------------------------------------------------------
# validate_path
# ---------------------------------------------------------------------------


class TestValidatePath:
    def test_returns_absolute_path(self, tmp_path):
        p = str(tmp_path / "subdir")
        result = validate_path(p)
        assert os.path.isabs(result)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_path("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_path("   ")

    def test_existing_path_no_must_exist(self, tmp_path):
        # must_exist=False (default): path does not need to exist
        nonexistent = str(tmp_path / "no_such_file.txt")
        result = validate_path(nonexistent)
        assert nonexistent in result or result.endswith("no_such_file.txt")

    def test_must_exist_true_existing(self, tmp_path):
        f = tmp_path / "real.txt"
        f.write_text("hello")
        result = validate_path(str(f), must_exist=True)
        assert result == str(f.resolve())

    def test_must_exist_true_missing_raises(self, tmp_path):
        nonexistent = str(tmp_path / "ghost.txt")
        with pytest.raises(ValueError, match="does not exist"):
            validate_path(nonexistent, must_exist=True)

    def test_base_dir_accepts_child(self, tmp_path):
        child = tmp_path / "child" / "file.txt"
        result = validate_path(str(child), base_dir=str(tmp_path))
        assert str(tmp_path) in result

    def test_base_dir_rejects_traversal(self, tmp_path):
        # Try to escape base_dir via ..
        traversal = str(tmp_path / ".." / "outside.txt")
        with pytest.raises(ValueError, match="traversal"):
            validate_path(traversal, base_dir=str(tmp_path))

    def test_base_dir_rejects_absolute_escape(self, tmp_path):
        # Provide an absolute path that is outside base_dir
        import tempfile
        other = tempfile.gettempdir()
        if os.path.realpath(other) == os.path.realpath(str(tmp_path)):
            pytest.skip("tmp_path is system tmpdir; cannot test escape")
        with pytest.raises(ValueError, match="traversal"):
            validate_path(other, base_dir=str(tmp_path))

    def test_tilde_expansion(self):
        result = validate_path("~")
        assert os.path.isabs(result)

    def test_resolves_dotdot(self, tmp_path):
        # /tmp/x/.. should resolve to /tmp
        child = tmp_path / "a"
        child.mkdir()
        result = validate_path(str(child / ".."))
        assert result == str(tmp_path.resolve())


# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------


class TestValidateUrl:
    def test_valid_http_url(self):
        url = "http://example.com"
        assert validate_url(url) == url

    def test_valid_https_url(self):
        url = "https://example.com/path?q=1"
        assert validate_url(url) == url

    def test_ftp_scheme_raises_by_default(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("ftp://example.com")

    def test_custom_allowed_scheme(self):
        url = "ftp://example.com"
        result = validate_url(url, allowed_schemes=("ftp",))
        assert result == url

    def test_no_hostname_raises(self):
        with pytest.raises(ValueError, match="hostname"):
            validate_url("http://")

    def test_missing_scheme_raises(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("example.com")

    def test_localhost_url_valid(self):
        url = "http://localhost:11434"
        result = validate_url(url)
        assert result == url

    def test_url_with_path_and_port(self):
        url = "https://api.example.com:8443/v1/endpoint"
        assert validate_url(url) == url

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            validate_url("")

    def test_scheme_not_in_allowed_list(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("https://example.com", allowed_schemes=("http",))

    def test_multiple_allowed_schemes(self):
        url = "ftp://files.example.com"
        result = validate_url(url, allowed_schemes=("http", "https", "ftp"))
        assert result == url


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config_no_warnings(self):
        cfg = {
            "CLUSTER_COUNT": 6,
            "MAX_PER_ARTIST": 4,
            "TRACKS_PER_MIX": 50,
        }
        warnings = validate_config(cfg)
        assert warnings == []

    def test_out_of_range_low_clamped(self):
        cfg = {"CLUSTER_COUNT": 0}  # min is 1
        warnings = validate_config(cfg)
        assert cfg["CLUSTER_COUNT"] == 1
        assert any("CLUSTER_COUNT" in w for w in warnings)

    def test_out_of_range_high_clamped(self):
        cfg = {"CLUSTER_COUNT": 9999}  # max is 100
        warnings = validate_config(cfg)
        assert cfg["CLUSTER_COUNT"] == 100
        assert any("CLUSTER_COUNT" in w for w in warnings)

    def test_invalid_type_string_warns(self):
        cfg = {"MAX_PER_ARTIST": "not_a_number"}
        warnings = validate_config(cfg)
        assert any("MAX_PER_ARTIST" in w for w in warnings)

    def test_valid_ollama_url_no_warning(self):
        cfg = {"OLLAMA_BASE_URL": "http://localhost:11434"}
        warnings = validate_config(cfg)
        assert not any("OLLAMA_BASE_URL" in w for w in warnings)
        assert cfg["OLLAMA_BASE_URL"] == "http://localhost:11434"

    def test_invalid_ollama_url_warns_and_clears(self):
        cfg = {"OLLAMA_BASE_URL": "not-a-url"}
        warnings = validate_config(cfg)
        assert any("OLLAMA_BASE_URL" in w for w in warnings)
        assert cfg["OLLAMA_BASE_URL"] is None

    def test_none_values_are_ignored(self):
        cfg = {"CLUSTER_COUNT": None}
        warnings = validate_config(cfg)
        assert warnings == []

    def test_missing_keys_not_warned(self):
        cfg = {}
        warnings = validate_config(cfg)
        assert warnings == []

    def test_multiple_out_of_range_values(self):
        cfg = {
            "CLUSTER_COUNT": 200,
            "MAX_PER_ARTIST": 0,
            "TRACKS_PER_MIX": 50,
        }
        warnings = validate_config(cfg)
        assert len(warnings) == 2
        assert cfg["CLUSTER_COUNT"] == 100
        assert cfg["MAX_PER_ARTIST"] == 1

    def test_audio_analysis_workers_clamped(self):
        cfg = {"AUDIO_ANALYSIS_WORKERS": -1}
        warnings = validate_config(cfg)
        assert cfg["AUDIO_ANALYSIS_WORKERS"] == 0
        assert any("AUDIO_ANALYSIS_WORKERS" in w for w in warnings)

    def test_string_numeric_accepted(self):
        # int() of a numeric string should work fine
        cfg = {"CLUSTER_COUNT": "5"}
        warnings = validate_config(cfg)
        # "5" is in range [1,100], no warning expected
        assert warnings == []

    def test_session_gap_minutes_range(self):
        cfg = {"SESSION_GAP_MINUTES": 2000}  # max is 1440
        warnings = validate_config(cfg)
        assert cfg["SESSION_GAP_MINUTES"] == 1440
        assert any("SESSION_GAP_MINUTES" in w for w in warnings)

    def test_lastfm_rate_limit_ms_zero_allowed(self):
        cfg = {"LASTFM_RATE_LIMIT_MS": 0}  # min is 0
        warnings = validate_config(cfg)
        assert warnings == []
        assert cfg["LASTFM_RATE_LIMIT_MS"] == 0
