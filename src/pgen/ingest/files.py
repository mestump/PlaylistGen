"""Directory-scanner source. Tags via mutagen when readable; filename 'Artist - Title' fallback."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..resolve import upsert_track

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".wav", ".alac"}
DEFAULT_LIBRARY = str(Path.home() / "Music")


def _tags(path: Path) -> tuple[str | None, str | None, str | None, int | None]:
    try:
        from mutagen import File as MutagenFile  # type: ignore[import-not-found]
    except ImportError:
        return _from_filename(path)
    try:
        mf = MutagenFile(str(path), easy=True)
    except Exception:
        return _from_filename(path)
    if mf is None:
        return _from_filename(path)

    def first(tag: str) -> str | None:
        v = mf.tags.get(tag) if mf.tags else None
        return str(v[0]) if v else None

    artist, title, album = first("artist"), first("title"), first("album")
    year = None
    for k in ("date", "year", "originalyear"):
        v = mf.tags.get(k) if mf.tags else None
        if v:
            digits = "".join(ch for ch in str(v[0]) if ch.isdigit())[:4]
            year = int(digits) if len(digits) == 4 else None
            if year:
                break
    if not (artist and title):
        fb_a, fb_t, _, _ = _from_filename(path)
        artist = artist or fb_a
        title = title or fb_t
    return artist, title, album, year


def _from_filename(path: Path) -> tuple[str | None, str | None, None, None]:
    stem = path.stem
    if " - " in stem:
        artist, _, title = stem.partition(" - ")
        return artist.strip() or None, title.strip() or None, None, None
    return None, stem.strip() or None, None, None


def ingest_directory(
    con: sqlite3.Connection, root: str | Path, source: str = "files", limit: int | None = None
) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"library root not found: {root}")
    seen = 0
    tagged = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        if "$RECYCLE.BIN" in path.parts or path.name.startswith("._"):
            continue
        artist, title, album, year = _tags(path)
        try:
            st = path.stat()
            mtime, size = st.st_mtime, st.st_size
        except OSError:
            mtime = size = None
        upsert_track(
            con,
            artist,
            title,
            album=album,
            year=year,
            path=str(path),
            path_source=source,
            mtime=mtime,
            size=size,
        )
        if artist:
            tagged += 1
        seen += 1
        if limit and seen >= limit:
            break
    con.commit()
    return {"source": source, "files": seen, "with_artist_tag": tagged, "root": str(root)}
