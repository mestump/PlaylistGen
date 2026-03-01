"""Tests for lastfm_client.py — SQLite cache, rate limiting, migration."""

import json
import time
import sqlite3
import tempfile
from pathlib import Path

import pytest
from playlistgen.lastfm_client import (
    init_cache_db,
    _get_cached,
    _set_cached,
    load_tag_db_from_sqlite,
    migrate_json_to_sqlite,
)


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test_tags.sqlite")
    conn = init_cache_db(db_path)
    yield db_path, conn
    conn.close()


def test_init_creates_table(tmp_db):
    db_path, conn = tmp_db
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='tag_cache'"
    ).fetchall()
    assert rows, "tag_cache table should be created"


def test_set_and_get_cached(tmp_db):
    _, conn = tmp_db
    _set_cached(conn, "artist a - track 1", ["rock", "indie"])
    result = _get_cached(conn, "artist a - track 1")
    assert result == ["rock", "indie"]


def test_get_cached_miss_returns_none(tmp_db):
    _, conn = tmp_db
    result = _get_cached(conn, "does not exist")
    assert result is None


def test_set_cached_upserts(tmp_db):
    _, conn = tmp_db
    _set_cached(conn, "artist a - track 1", ["rock"])
    _set_cached(conn, "artist a - track 1", ["rock", "indie"])
    result = _get_cached(conn, "artist a - track 1")
    assert result == ["rock", "indie"]


def test_load_tag_db_from_sqlite(tmp_path):
    db_path = str(tmp_path / "test.sqlite")
    conn = init_cache_db(db_path)
    _set_cached(conn, "artist a - track 1", ["rock"])
    _set_cached(conn, "artist b - track 2", ["pop", "happy"])
    conn.close()

    db = load_tag_db_from_sqlite(db_path)
    assert db["artist a - track 1"] == ["rock"]
    assert db["artist b - track 2"] == ["pop", "happy"]


def test_load_tag_db_missing_file(tmp_path):
    result = load_tag_db_from_sqlite(str(tmp_path / "missing.sqlite"))
    assert result == {}


def test_migrate_json_to_sqlite_list_format(tmp_path):
    # Old JSON format: {"artist - track": ["tag1", "tag2"]}
    json_path = tmp_path / "old_cache.json"
    old_data = {
        "artist a - track 1": ["rock", "indie"],
        "artist b - track 2": ["pop"],
    }
    json_path.write_text(json.dumps(old_data))

    db_path = str(tmp_path / "new.sqlite")
    conn = init_cache_db(db_path)
    count = migrate_json_to_sqlite(str(json_path), conn)
    conn.close()

    assert count == 2
    db = load_tag_db_from_sqlite(db_path)
    assert db["artist a - track 1"] == ["rock", "indie"]


def test_migrate_json_to_sqlite_dict_format(tmp_path):
    # Legacy dict format: {"artist - track": {"tags": [...], "mood": "..."}}
    json_path = tmp_path / "old_cache.json"
    old_data = {
        "artist a - track 1": {"tags": ["rock", "indie"], "mood": "Happy"},
    }
    json_path.write_text(json.dumps(old_data))

    db_path = str(tmp_path / "new.sqlite")
    conn = init_cache_db(db_path)
    migrate_json_to_sqlite(str(json_path), conn)
    conn.close()

    db = load_tag_db_from_sqlite(db_path)
    assert db["artist a - track 1"] == ["rock", "indie"]


def test_migrate_skips_existing(tmp_path):
    json_path = tmp_path / "old.json"
    json_path.write_text(json.dumps({"a - b": ["rock"]}))

    db_path = str(tmp_path / "new.sqlite")
    conn = init_cache_db(db_path)
    _set_cached(conn, "a - b", ["pop"])  # pre-existing entry
    count = migrate_json_to_sqlite(str(json_path), conn)
    conn.close()

    assert count == 0  # nothing migrated (already cached)
    db = load_tag_db_from_sqlite(db_path)
    assert db["a - b"] == ["pop"]  # original value preserved


def test_migrate_missing_file(tmp_path):
    db_path = str(tmp_path / "new.sqlite")
    conn = init_cache_db(db_path)
    count = migrate_json_to_sqlite(str(tmp_path / "nonexistent.json"), conn)
    assert count == 0
    conn.close()
