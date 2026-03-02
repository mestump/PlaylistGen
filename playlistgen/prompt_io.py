"""
Prompt I/O — API-free AI enrichment and curation via copy-paste workflow.

Generates self-contained prompts the user can paste into any AI assistant
(Claude.ai, ChatGPT, Gemini, etc.) and import the response back — no API key
needed beyond what the user already has.

Two modes
---------
  enrich  — classify mood / energy / valence for tracks missing metadata
  curate  — group scored tracks into N themed playlists

Typical workflow
----------------
  # 1. Generate the prompt file
  playlistgen export-ai-prompt --mode enrich

  # 2. Paste the prompt section into any AI.  Copy the JSON response.

  # 3. Paste the response into the RESPONSE section at the bottom of the
  #    generated file (or save it as a separate .json file), then run:
  playlistgen import-ai-result playlistgen_enrich_prompt.txt
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ENRICH_PROMPT = """\
You are a music metadata expert. Classify each numbered track below with its \
mood, energy level, and valence (emotional positivity/brightness).

OUTPUT FORMAT — respond with ONLY a raw JSON array. No explanation, no code \
fences, no markdown. Start your response with [ and end with ].

Example shape (do NOT copy these values):
[
  {{"idx": 1, "mood": "Happy",  "energy": 8, "valence": 7, "tags": ["upbeat", "danceable", "pop"]}},
  {{"idx": 2, "mood": "Sad",    "energy": 3, "valence": 2, "tags": ["melancholic", "slow", "ballad"]}}
]

RULES
• mood    — pick exactly ONE from: Happy, Sad, Angry, Chill, Energetic, \
Romantic, Epic, Dreamy, Groovy, Nostalgic
• energy  — integer 1–10  (1 = ambient/very quiet, 10 = loud/intense)
• valence — integer 1–10  (1 = very dark/negative, 10 = very bright/positive)
• tags    — list of 3–5 short descriptive music tags (e.g. "driving beat", \
"indie guitar", "lush strings")
• Include ALL {n} tracks — make your best guess if uncertain
• Return exactly {n} objects in the same numbered order

TRACKS TO CLASSIFY
{track_list}"""

_CURATE_PROMPT = """\
You are an expert music playlist curator. Group the {n_tracks} numbered tracks \
below into exactly {n_playlists} cohesive, themed playlists.

OUTPUT FORMAT — respond with ONLY a raw JSON object. No explanation, no code \
fences, no markdown. Start your response with {{ and end with }}.

Example shape (do NOT copy these values):
{{
  "playlists": [
    {{
      "name": "Golden Hour Drive",
      "track_ids": [1, 7, 23, 45],
      "theme": "Warm, nostalgic indie-pop for late afternoon"
    }}
  ]
}}

RULES
• Create exactly {n_playlists} playlists
• Each track may appear in AT MOST one playlist (some tracks may be unassigned)
• Each playlist should have 15–50 tracks
• Consider: mood, genre, energy level, era, and overall sonic coherence
• Playlist names — creative and evocative, max 5 words, no generic names \
like "Mix 1" or "Playlist A"
• track_ids use the 1-based numbers from the list below

TRACKS ({n_tracks} total)
{track_list}"""

# ---------------------------------------------------------------------------
# File wrapper (the .txt file the user edits)
# ---------------------------------------------------------------------------

_SEP = "─" * 72

_FILE_TEMPLATE = """\
{sep}
PLAYLISTGEN  ·  AI {mode_upper} PROMPT  ·  {date}
Mode: {mode}  ·  Tracks in prompt: {n_tracks}{batch_info}
{sep}

HOW TO USE
══════════
  1. Copy everything between ── PROMPT START ── and ── PROMPT END ── below.

  2. Paste it into Claude.ai, ChatGPT, Gemini, or any AI assistant.
     Recommended output token capacity by AI:
       Claude.ai (free/Pro/Max) — 32K out  → up to 500 tracks
       ChatGPT Plus (GPT-4o)   — 16K out  → up to 250 tracks
       Gemini Advanced          — 8K out   → up to 100 tracks
     Override with: playlistgen export-ai-prompt --batch-size N

  3. Copy the AI's entire JSON response.

  4. Either:
       a) Paste the JSON into the RESPONSE section at the bottom of this file
          (replace the placeholder text between the markers), then run:
            playlistgen import-ai-result "{filename}"
       b) Save the JSON as a separate file and run:
            playlistgen import-ai-result path/to/response.json

  5. For enrich mode: enrichment is cached — re-run playlistgen normally.
     For curate mode: playlists are written immediately to your output dir.

{sep}
── PROMPT START ──────────────────────────────────────────────────────────────

{prompt}

── PROMPT END ────────────────────────────────────────────────────────────────
{sep}

── RESPONSE START ────────────────────────────────────────────────────────────

<<< PASTE THE AI'S JSON RESPONSE HERE — delete this line >>>

── RESPONSE END ──────────────────────────────────────────────────────────────
"""

# ---------------------------------------------------------------------------
# Track-list formatters
# ---------------------------------------------------------------------------


def _format_enrich_line(i: int, row: pd.Series) -> str:
    """Build a compact, token-efficient track description line."""
    artist = str(row.get("Artist") or "Unknown Artist").strip()
    name = str(row.get("Name") or "Unknown Track").strip()
    parts = [f"{artist} — {name}"]
    genre = str(row.get("Genre") or "").strip()
    if genre:
        parts.append(f"Genre:{genre}")
    bpm = row.get("BPM")
    if bpm is not None and pd.notnull(bpm) and float(bpm) > 0:
        parts.append(f"BPM:{int(bpm)}")
    year = row.get("Year")
    if year is not None and pd.notnull(year):
        try:
            parts.append(f"Year:{int(year)}")
        except (ValueError, TypeError):
            pass
    return f"{i}. {' | '.join(parts)}"


def _format_curate_line(i: int, row: pd.Series) -> str:
    """Build a richer track description for curation (includes mood/energy)."""
    artist = str(row.get("Artist") or "Unknown Artist").strip()
    name = str(row.get("Name") or "Unknown Track").strip()
    parts = [f"{artist} — {name}"]
    mood = str(row.get("Mood") or "").strip()
    if mood and mood not in ("Unknown", ""):
        parts.append(f"Mood:{mood}")
    genre = str(row.get("Genre") or "").strip()
    if genre:
        parts.append(f"Genre:{genre}")
    energy = row.get("Energy")
    if energy is not None and pd.notnull(energy):
        try:
            parts.append(f"Energy:{int(round(float(energy)))}")
        except (ValueError, TypeError):
            pass
    bpm = row.get("BPM")
    if bpm is not None and pd.notnull(bpm) and float(bpm) > 0:
        parts.append(f"BPM:{int(bpm)}")
    year = row.get("Year")
    if year is not None and pd.notnull(year):
        try:
            parts.append(f"Era:{int(year)}")
        except (ValueError, TypeError):
            pass
    return f"{i}. {' | '.join(parts)}"


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_enrichment_prompt(
    df: pd.DataFrame,
    out_path: Optional[str] = None,
    batch_size: int = 300,
    batch_index: int = 0,
    cache_db: Optional[str] = None,
) -> Path:
    """
    Generate a self-contained AI enrichment prompt for tracks missing Mood data.

    Only includes tracks that have no Mood (or Mood == "Unknown").  Checks the
    enrichment SQLite cache to skip already-processed tracks.

    Args:
        df:          Library DataFrame (Artist, Name, Genre, BPM, Year, Mood).
        out_path:    Output .txt file path.  Defaults to
                     ./playlistgen_enrich_prompt[_N].txt
        batch_size:  Max tracks per prompt (default 150; most AIs handle this fine).
        batch_index: Which batch to export when the library is large (0-based).
        cache_db:    Enrichment SQLite cache path (to skip already-enriched tracks).

    Returns:
        Path to the generated .txt file.
    """
    if cache_db is None:
        cache_db = str(Path.home() / ".playlistgen" / "claude_enrichment.sqlite")

    # Identify tracks needing enrichment
    needs_enrich = []
    cached_keys: set = set()

    if Path(cache_db).exists():
        try:
            import sqlite3
            conn = sqlite3.connect(cache_db)
            rows = conn.execute("SELECT key FROM claude_enrichment").fetchall()
            cached_keys = {r[0] for r in rows}
            conn.close()
        except Exception as exc:
            logger.debug("Could not read enrichment cache: %s", exc)

    for idx, row in df.iterrows():
        artist = str(row.get("Artist") or "").strip()
        name = str(row.get("Name") or "").strip()
        if not artist or not name:
            continue
        mood = str(row.get("Mood") or "").strip()
        if mood and mood not in ("Unknown", ""):
            continue  # already has a mood
        key = f"{artist} - {name}".lower()
        if key in cached_keys:
            continue
        needs_enrich.append((idx, row))

    total_needing = len(needs_enrich)
    n_batches = max(1, (total_needing + batch_size - 1) // batch_size)

    if total_needing == 0:
        logger.info("All tracks already have mood data — nothing to enrich.")
        print("\nAll tracks already have mood data. Nothing to export.")
        return Path(out_path or "playlistgen_enrich_prompt.txt")

    # Slice the requested batch
    start = batch_index * batch_size
    batch = needs_enrich[start : start + batch_size]
    if not batch:
        raise ValueError(
            f"batch_index {batch_index} is out of range "
            f"(total batches: {n_batches} for {total_needing} tracks)"
        )

    # Build the numbered track list
    track_lines = [_format_enrich_line(i + 1, row) for i, (_, row) in enumerate(batch)]
    track_list_str = "\n".join(track_lines)
    n = len(batch)

    prompt_text = _ENRICH_PROMPT.format(n=n, track_list=track_list_str)

    # Determine output path
    if out_path is None:
        suffix = f"_{batch_index + 1}" if n_batches > 1 else ""
        out_path = f"playlistgen_enrich_prompt{suffix}.txt"
    out = Path(out_path)

    batch_info = ""
    if n_batches > 1:
        batch_info = (
            f"  ·  Batch {batch_index + 1} of {n_batches} "
            f"({total_needing} tracks total need enrichment)"
        )

    file_content = _FILE_TEMPLATE.format(
        sep=_SEP,
        mode_upper="ENRICHMENT",
        mode="enrich",
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_tracks=n,
        batch_info=batch_info,
        filename=out.name,
        prompt=prompt_text,
    )

    out.write_text(file_content, encoding="utf-8")

    remaining_msg = (
        f"  ({total_needing - start - n} more tracks in subsequent batches)"
        if n_batches > 1 and batch_index < n_batches - 1
        else ""
    )

    print(f"""
{_SEP}
  PLAYLISTGEN — AI Enrichment Prompt Generated
{_SEP}

  File  : {out.resolve()}
  Tracks: {n} (batch {batch_index + 1}/{n_batches}){remaining_msg}

  NEXT STEPS
  ──────────
  1. Open the file and copy the prompt between the PROMPT START/END markers.

  2. Paste into Claude.ai, ChatGPT, Gemini, or any AI with a large output
     window (16K+ tokens recommended).

  3. Copy the AI's JSON response, paste it into the RESPONSE section at the
     bottom of the file (replace the placeholder), then run:

       playlistgen import-ai-result "{out.name}"
{f"  4. Repeat for the next batch: playlistgen export-ai-prompt --batch {batch_index + 2}" if n_batches > 1 and batch_index < n_batches - 1 else ""}
{_SEP}
""")

    return out


def export_curation_prompt(
    scored_df: pd.DataFrame,
    n_playlists: int = 6,
    out_path: Optional[str] = None,
    max_tracks: int = 500,
) -> Path:
    """
    Generate a self-contained AI playlist curation prompt.

    Sends the top `max_tracks` scored tracks to the prompt.  The AI groups
    them into n_playlists themed playlists.

    Args:
        scored_df:   DataFrame with Score, Mood, Genre, Artist, Name, etc.
        n_playlists: Number of playlists to request (default 6).
        out_path:    Output .txt file path.  Defaults to
                     ./playlistgen_curate_prompt.txt
        max_tracks:  Max tracks to include (default 300).

    Returns:
        Path to the generated .txt file.
    """
    top_df = (
        scored_df.sort_values("Score", ascending=False)
        .head(max_tracks)
        .reset_index(drop=True)
    )
    if top_df.empty:
        raise ValueError("scored_df is empty — run the pipeline first.")

    track_lines = [
        _format_curate_line(i + 1, row) for i, (_, row) in enumerate(top_df.iterrows())
    ]
    track_list_str = "\n".join(track_lines)
    n_tracks = len(top_df)

    prompt_text = _CURATE_PROMPT.format(
        n_tracks=n_tracks,
        n_playlists=n_playlists,
        track_list=track_list_str,
    )

    if out_path is None:
        out_path = "playlistgen_curate_prompt.txt"
    out = Path(out_path)

    file_content = _FILE_TEMPLATE.format(
        sep=_SEP,
        mode_upper="CURATION",
        mode="curate",
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_tracks=n_tracks,
        batch_info=f"  ·  Requesting {n_playlists} playlists",
        filename=out.name,
        prompt=prompt_text,
    )

    # Embed a snapshot of the top_df index map so import can reconstruct
    # which original rows map to 1-based prompt indices.
    index_map = {
        str(i + 1): {
            "Artist": str(row.get("Artist") or ""),
            "Name": str(row.get("Name") or ""),
            "orig_idx": int(idx),
        }
        for i, (idx, row) in enumerate(top_df.iterrows())
    }
    snapshot_json = json.dumps(index_map, ensure_ascii=False)
    file_content += (
        f"\n{_SEP}\n"
        "── TRACK INDEX (do not edit — used by import) ─────────────────────────────────\n"
        f"{snapshot_json}\n"
        "── END TRACK INDEX ──────────────────────────────────────────────────────────────\n"
    )

    out.write_text(file_content, encoding="utf-8")

    print(f"""
{_SEP}
  PLAYLISTGEN — AI Curation Prompt Generated
{_SEP}

  File     : {out.resolve()}
  Tracks   : {n_tracks}
  Playlists: {n_playlists}

  NEXT STEPS
  ──────────
  1. Open the file and copy the prompt between the PROMPT START/END markers.

  2. Paste into Claude.ai, ChatGPT, Gemini, or any AI.
     (Claude.ai handles 300-track prompts with ease.)

  3. Copy the AI's JSON response, paste it into the RESPONSE section at the
     bottom of the file, then run:

       playlistgen import-ai-result "{out.name}"

  Playlists will be written to your configured output directory.
{_SEP}
""")

    return out


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def _extract_json_from_text(text: str) -> str:
    """
    Pull the first JSON array or object out of a blob of text.

    Handles:
    - Raw JSON (starts with [ or {)
    - Markdown code fences (```json ... ``` or ``` ... ```)
    - JSON embedded after explanatory text
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        inner = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
        inner = re.sub(r"\n?```$", "", inner.strip())
        text = inner.strip()

    if text.startswith(("[", "{")):
        return text

    # Try to find the first [ or { and match to its closing counterpart
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start=start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text  # fall through and let json.loads fail with a clear error


def _extract_response_from_file(text: str) -> str:
    """
    Extract the content between ── RESPONSE START ── and ── RESPONSE END ──
    markers in a playlistgen prompt file.  Falls back to the whole text if
    no markers are found (supports plain .json files too).
    """
    match = re.search(
        r"──\s*RESPONSE START\s*─+\s*\n(.*?)──\s*RESPONSE END\s*─+",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    # Check if it looks like a raw JSON file
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        return stripped

    return text  # let downstream parsing handle it


def _extract_track_index_from_file(text: str) -> Optional[dict]:
    """Extract the embedded track index JSON from a curate prompt file."""
    match = re.search(
        r"──\s*TRACK INDEX.*?─+\s*\n(\{.*?\})\s*\n──\s*END TRACK INDEX",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _detect_mode(text: str) -> str:
    """Detect 'enrich' or 'curate' from a prompt file's header line."""
    if re.search(r"Mode:\s*enrich", text, re.IGNORECASE):
        return "enrich"
    if re.search(r"Mode:\s*curate", text, re.IGNORECASE):
        return "curate"
    # Guess from JSON shape
    stripped = _extract_response_from_file(text)
    json_str = _extract_json_from_text(stripped)
    if json_str.lstrip().startswith("["):
        return "enrich"
    return "curate"


# ---------------------------------------------------------------------------
# Import functions
# ---------------------------------------------------------------------------


def import_enrichment_result(
    source: str,
    df: pd.DataFrame,
    cache_db: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse an AI enrichment response and apply it to the DataFrame + cache.

    Args:
        source:   Path to a .txt prompt file (with RESPONSE section) or a
                  plain .json file containing the AI's array response.
        df:       Library DataFrame to update in-place (copy is returned).
        cache_db: Enrichment SQLite cache path.

    Returns:
        Updated DataFrame with Mood, Energy, Valence filled.
    """
    if cache_db is None:
        cache_db = str(Path.home() / ".playlistgen" / "claude_enrichment.sqlite")

    raw = Path(source).read_text(encoding="utf-8")
    response_text = _extract_response_from_file(raw)
    if not response_text or response_text.startswith("<<< PASTE"):
        raise ValueError(
            "No AI response found in the file.  "
            "Paste the JSON into the RESPONSE section and try again."
        )

    json_str = _extract_json_from_text(response_text)
    try:
        results = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse the AI's response as JSON: {exc}") from exc

    if isinstance(results, dict):
        # Some AIs wrap the array: {"results": [...]}
        results = results.get("results", results.get("tracks", []))

    if not isinstance(results, list):
        raise ValueError(
            "Expected a JSON array of enrichment objects but got: "
            + type(results).__name__
        )

    # Open cache
    conn = None
    try:
        import sqlite3
        Path(cache_db).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(cache_db)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS claude_enrichment (
                key TEXT PRIMARY KEY,
                mood TEXT,
                energy INTEGER,
                valence INTEGER,
                tags TEXT,
                enriched_at INTEGER
            )"""
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Could not open enrichment cache: %s — continuing without cache.", exc)

    df = df.copy()
    for col in ("Mood", "Energy", "Valence"):
        if col not in df.columns:
            df[col] = None

    # Build a lookup: 1-based idx → (df_row_index, cache_key)
    # We need to reconstruct which prompt positions map to which df rows.
    # We replay the same "needs enrichment" logic used at export time.
    needs_enrich = []
    cached_keys: set = set()
    if conn:
        try:
            rows = conn.execute("SELECT key FROM claude_enrichment").fetchall()
            cached_keys = {r[0] for r in rows}
        except Exception:
            pass

    for idx, row in df.iterrows():
        artist = str(row.get("Artist") or "").strip()
        name = str(row.get("Name") or "").strip()
        if not artist or not name:
            continue
        mood = str(row.get("Mood") or "").strip()
        if mood and mood not in ("Unknown", ""):
            continue
        key = f"{artist} - {name}".lower()
        if key in cached_keys:
            continue
        needs_enrich.append((idx, key))

    applied = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        idx_1based = item.get("idx")
        if idx_1based is None:
            continue
        try:
            idx_1based = int(idx_1based)
        except (ValueError, TypeError):
            continue

        if not (1 <= idx_1based <= len(needs_enrich)):
            logger.debug("idx %d out of range (%d tracks) — skipping", idx_1based, len(needs_enrich))
            continue

        df_idx, key = needs_enrich[idx_1based - 1]
        mood = str(item.get("mood") or "").strip()
        energy = item.get("energy")
        valence = item.get("valence")
        tags = item.get("tags", [])

        if mood and mood in (
            "Happy", "Sad", "Angry", "Chill", "Energetic",
            "Romantic", "Epic", "Dreamy", "Groovy", "Nostalgic",
        ):
            df.at[df_idx, "Mood"] = mood
        elif mood:
            df.at[df_idx, "Mood"] = mood  # accept non-canonical moods too

        if energy is not None:
            try:
                df.at[df_idx, "Energy"] = int(energy)
            except (ValueError, TypeError):
                pass
        if valence is not None:
            try:
                df.at[df_idx, "Valence"] = int(valence)
            except (ValueError, TypeError):
                pass

        if conn:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO claude_enrichment
                       (key, mood, energy, valence, tags, enriched_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        key,
                        mood or None,
                        int(energy) if energy is not None else None,
                        int(valence) if valence is not None else None,
                        json.dumps(tags) if tags else None,
                        int(time.time()),
                    ),
                )
            except Exception as exc:
                logger.debug("Cache write failed for '%s': %s", key, exc)

        applied += 1

    if conn:
        try:
            conn.commit()
            conn.close()
        except Exception:
            pass

    mood_total = (df["Mood"].notna() & ~df["Mood"].isin(["Unknown", ""])).sum()
    print(f"""
{_SEP}
  PLAYLISTGEN — Enrichment Import Complete
{_SEP}

  Tracks enriched : {applied}
  Mood coverage   : {mood_total} / {len(df)} tracks now have mood data
  Cache updated   : {cache_db}

  Run  playlistgen  normally — the enriched data will be used automatically.
{_SEP}
""")
    logger.info("Import: %d tracks enriched; %d total with mood data.", applied, mood_total)
    return df


def import_curation_result(
    source: str,
    scored_df: pd.DataFrame,
) -> List[Tuple[str, pd.DataFrame]]:
    """
    Parse an AI curation response and return (label, DataFrame) playlist tuples.

    Uses the embedded track index in the prompt file to reconstruct which
    original df rows belong to each playlist.

    Args:
        source:    Path to the .txt curate prompt file (with RESPONSE section).
        scored_df: The scored DataFrame used when the prompt was generated.

    Returns:
        List of (playlist_name, DataFrame) tuples, ready for M3U export.
    """
    raw = Path(source).read_text(encoding="utf-8")

    # Extract the response
    response_text = _extract_response_from_file(raw)
    if not response_text or response_text.startswith("<<< PASTE"):
        raise ValueError(
            "No AI response found in the file.  "
            "Paste the JSON into the RESPONSE section and try again."
        )

    json_str = _extract_json_from_text(response_text)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse the AI's response as JSON: {exc}") from exc

    if isinstance(data, list):
        playlists_data = data
    elif isinstance(data, dict):
        playlists_data = data.get("playlists", [])
    else:
        raise ValueError("Unexpected JSON shape — expected a playlists object or array.")

    # Try to use the embedded track index for robust mapping
    track_index = _extract_track_index_from_file(raw)

    # Fallback: reconstruct the top-300 order the same way export_curation_prompt does
    if track_index is None:
        logger.warning(
            "No TRACK INDEX section found in file — reconstructing from scored_df order."
        )
        top_df = (
            scored_df.sort_values("Score", ascending=False)
            .head(300)
            .reset_index(drop=True)
        )
        track_index = {
            str(i + 1): {"orig_idx": int(idx)}
            for i, idx in enumerate(top_df.index)
        }

    result: List[Tuple[str, pd.DataFrame]] = []
    assigned: set = set()

    for pl in playlists_data:
        if not isinstance(pl, dict):
            continue
        name = str(pl.get("name") or "").strip() or "AI Curated Mix"
        track_ids = pl.get("track_ids", [])
        if not track_ids:
            continue

        df_rows = []
        for tid in track_ids:
            entry = track_index.get(str(tid))
            if entry is None:
                continue
            orig_idx = entry.get("orig_idx")
            if orig_idx is None or orig_idx in assigned:
                continue
            if orig_idx in scored_df.index:
                df_rows.append(scored_df.loc[orig_idx])
                assigned.add(orig_idx)

        if not df_rows:
            continue

        playlist_df = pd.DataFrame(df_rows).reset_index(drop=True)
        result.append((name, playlist_df))
        logger.info("Imported playlist '%s': %d tracks", name, len(playlist_df))

    print(f"""
{_SEP}
  PLAYLISTGEN — Curation Import Complete
{_SEP}

  Playlists imported : {len(result)}
  Tracks assigned    : {len(assigned)}
{_SEP}
""")
    return result


# ---------------------------------------------------------------------------
# Auto-dispatch: import either mode from a single file
# ---------------------------------------------------------------------------


def import_ai_result(
    source: str,
    df: Optional[pd.DataFrame] = None,
    scored_df: Optional[pd.DataFrame] = None,
    cache_db: Optional[str] = None,
    mode: Optional[str] = None,
) -> object:
    """
    Auto-detect mode and dispatch to import_enrichment_result or
    import_curation_result.

    Returns:
        Enriched DataFrame (enrich mode) or List[(label, df)] (curate mode).
    """
    raw = Path(source).read_text(encoding="utf-8")
    detected = mode or _detect_mode(raw)

    if detected == "enrich":
        if df is None:
            raise ValueError("df is required for enrich mode import")
        return import_enrichment_result(source, df, cache_db=cache_db)
    else:
        if scored_df is None:
            raise ValueError("scored_df is required for curate mode import")
        return import_curation_result(source, scored_df)
