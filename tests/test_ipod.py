import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgen.ingest.ipod import parse_itdb  # noqa: E402


def _mhod(typ: int, text: str | None) -> bytes:
    if text is None:
        return struct.pack("<4sIIIII", b"mhod", 0x18, 0x18, typ, 0, 0)
    raw = text.encode("utf-16-le")
    body = struct.pack("<IIII", 1, len(raw) + 2, 1, 0) + raw + b"\x00\x00"
    return struct.pack("<4sIIIII", b"mhod", 0x18, 0x18 + len(body), typ, 0, 0) + body


def _mhit(plays: int, skips: int, children: bytes) -> bytes:
    total = 0x40 + len(children)
    head = bytearray(0x40)
    struct.pack_into("<4sII", head, 0, b"mhit", 0x40, total)
    struct.pack_into("<I", head, 0x2C, skips)
    struct.pack_into("<I", head, 0x30, plays)
    return bytes(head) + children


def build_itdb(tracks: list[tuple[str, str, int, int]]) -> tuple[bytes, int]:
    mhods = b""
    for title, artist, plays, skips in tracks:
        mhods += _mhit(plays, skips, _mhod(1, title) + _mhod(2, ":iPod:x.y") + _mhod(4, artist))
    mhit_hdrlen = 0x40

    mhlt = bytearray(0x5C)
    struct.pack_into("<4sII", mhlt, 0, b"mhlt", 0x5C, 0x5C + len(mhods))

    track_body = bytes(mhlt) + mhods
    mhsd = bytearray(0x60)
    struct.pack_into("<4sII", mhsd, 0, b"mhsd", 0x60, 0x60 + len(track_body))
    struct.pack_into("<I", mhsd, 0x0C, 1)  # track mhsd

    mhbd = bytearray(0x100)
    struct.pack_into("<4sII", mhbd, 0, b"mhbd", 0x100, 0x100 + len(mhsd))
    return bytes(mhbd) + bytes(mhsd) + track_body, mhit_hdrlen


def test_parse_minimal_itdb(tmp_path: Path):
    blob, _ = build_itdb(
        [("Peacebone", "Animal Collective", 12, 3), ("No Cars Go", "Arcade Fire", 0, 0)]
    )
    p = tmp_path / "iTunesDB"
    p.write_bytes(blob)
    got = parse_itdb(p)
    assert len(got) == 2
    assert got[0] == {
        "title": "Peacebone",
        "artist": "Animal Collective",
        "device_path": ":iPod:x.y",
    }
