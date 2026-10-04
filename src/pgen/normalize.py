"""Track identity normalization. The canonical key is (artist_canon, title_canon)."""

from __future__ import annotations

import re
import unicodedata

_FEAT = re.compile(r"\s*[\(\[]?\s*(feat|ft|featuring|with)\.?\s+[^)\]]*[\)\]]?\s*$", re.I)
_VERSION = re.compile(
    r"\s+[\(\-]?\s*(album version|single version|radio edit|remastered( \d{4})?|"
    r"bonus track|live|acoustic|demo|instrumental|edit|remix|extended|"
    r"deluxe( edition)?|mono|stereo|version)\b[^)\]]*[\)\]]?\s*$",
    re.I,
)
_PUNCT = re.compile(r"[^\w\s&]")
_SPACE = re.compile(r"\s+")
_LEADING_THE = re.compile(r"^the\s+", re.I)


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _FEAT.sub(" ", s)
    s = _VERSION.sub(" ", s)
    s = s.replace("&", " and ")
    s = _PUNCT.sub(" ", s)
    s = _SPACE.sub(" ", s).strip().lower()
    s = _LEADING_THE.sub("", s)
    return s


def canon_artist(artist: str | None) -> str:
    if not artist:
        return ""
    return _clean(str(artist))


def canon_title(title: str | None) -> str:
    if not title:
        return ""
    return _clean(str(title))


def track_key(artist: str | None, title: str | None) -> tuple[str, str]:
    return canon_artist(artist), canon_title(title)
