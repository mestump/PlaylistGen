"""
Ollama-based batch metadata enrichment for PlaylistGen.

Drop-in alternative to Claude batch enrichment in ai_enhancer.py.
Uses a local Ollama instance for mood/energy/valence classification,
keeping all data on-device with zero API cost.

Usage:
    from playlistgen.enrichers.ollama_enricher import batch_enrich_ollama
    df = batch_enrich_ollama(df, base_url="http://localhost:11434", model="llama3")
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

_ENRICH_SCHEMA = """
CREATE TABLE IF NOT EXISTS ollama_enrichment (
    key         TEXT PRIMARY KEY,
    mood        TEXT,
    energy      INTEGER,
    valence     INTEGER,
    tags        TEXT,
    enriched_at INTEGER
);
"""

_SYSTEM_PROMPT = (
    "You are a music metadata expert. Given a numbered list of tracks, classify "
    "each one's mood, energy, and valence.\n\n"
    "Respond with ONLY a JSON array of objects, one per track:\n"
    '[{"idx": N, "mood": "Mood", "energy": E, "valence": V, '
    '"tags": ["tag1", "tag2"]}]\n\n'
    "Mood must be exactly one of: Happy, Sad, Angry, Chill, Energetic, "
    "Romantic, Epic, Dreamy, Groovy, Nostalgic\n"
    "Energy: integer 1-10 (1=very quiet/calm, 10=very loud/intense)\n"
    "Valence: integer 1-10 (1=very dark/negative, 10=very bright/positive)\n"
    "Tags: list of 3-5 short descriptive music tags\n"
    "Return ONLY valid JSON. No markdown, no explanation."
)


def _init_cache_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_ENRICH_SCHEMA)
    conn.commit()
    return conn


def _parse_json_response(raw: str) -> Optional[list]:
    """Extract JSON array from Ollama response, handling markdown fences."""
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first and last fence lines
        inner = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```") and not in_fence:
                in_fence = True
                continue
            if line.strip() == "```" and in_fence:
                break
            if in_fence:
                inner.append(line)
        raw = "\n".join(inner)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array in the response
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def batch_enrich_ollama(
    df: pd.DataFrame,
    base_url: str = "http://localhost:11434",
    model: str = "llama3",
    batch_size: int = 50,
    cache_db: Optional[str] = None,
    rate_limit_ms: int = 0,
) -> pd.DataFrame:
    """
    Batch-enrich track metadata using a local Ollama instance.

    Sends tracks in batches for mood/energy/valence classification.
    Results are cached in SQLite; subsequent runs skip enriched tracks.

    Args:
        df:             Library DataFrame with Artist, Name, Genre, BPM columns.
        base_url:       Ollama server URL (default http://localhost:11434).
        model:          Ollama model name.
        batch_size:     Tracks per API call (default 50, lower than Claude due to
                        smaller context windows on most local models).
        cache_db:       SQLite cache path.
        rate_limit_ms:  Minimum ms between API calls.

    Returns:
        DataFrame with Mood, Energy, Valence columns filled where previously missing.
    """
    if not REQUESTS_AVAILABLE:
        logging.warning("requests not installed — Ollama enrichment skipped.")
        return df

    if cache_db is None:
        cache_db = str(Path.home() / ".playlistgen" / "ollama_enrichment.sqlite")
    cache_db = str(Path(cache_db).expanduser())

    try:
        conn = _init_cache_db(cache_db)
    except Exception as exc:
        logging.warning("Ollama enrichment cache DB init failed: %s", exc)
        conn = None

    df = df.copy()
    for col in ("Mood", "Energy", "Valence"):
        if col not in df.columns:
            df[col] = None

    # Build list of tracks needing enrichment
    to_enrich = []
    for idx, row in df.iterrows():
        artist = str(row.get("Artist") or "")
        name = str(row.get("Name") or "")
        if not artist or not name:
            continue
        key = f"{artist} - {name}".lower().strip()

        # Skip if already enriched
        if pd.notnull(row.get("Mood")) and row.get("Mood") not in ("Unknown", "", None):
            continue

        # Check cache
        if conn:
            cached = conn.execute(
                "SELECT mood, energy, valence FROM ollama_enrichment WHERE key=?",
                (key,),
            ).fetchone()
            if cached:
                if cached[0]:
                    df.at[idx, "Mood"] = cached[0]
                if cached[1] is not None:
                    df.at[idx, "Energy"] = int(cached[1])
                if cached[2] is not None:
                    df.at[idx, "Valence"] = int(cached[2])
                continue

        genre = str(row.get("Genre") or "")
        bpm = row.get("BPM")
        bpm_str = f" | BPM:{int(bpm)}" if bpm is not None and pd.notnull(bpm) and bpm > 0 else ""
        genre_str = f" | Genre:{genre}" if genre else ""
        label = f"{artist} - {name}{genre_str}{bpm_str}"
        to_enrich.append((idx, key, label))

    if not to_enrich:
        logging.info("Ollama enrichment: all tracks already enriched (cached).")
        if conn:
            conn.close()
        return df

    num_batches = (len(to_enrich) + batch_size - 1) // batch_size
    logging.info(
        "Ollama enrichment: %d tracks in %d batches of up to %d…",
        len(to_enrich), num_batches, batch_size,
    )

    enriched_count = 0
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    last_batch_time = 0.0

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch = to_enrich[batch_start : batch_start + batch_size]

        # Rate limiting
        if rate_limit_ms > 0 and last_batch_time > 0:
            elapsed = time.time() - last_batch_time
            wait = (rate_limit_ms / 1000.0) - elapsed
            if wait > 0:
                time.sleep(wait)

        user_msg = "Classify these tracks:\n" + "\n".join(
            f"{i + 1}. {item[2]}" for i, item in enumerate(batch)
        )

        # Call Ollama with retry
        results = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    endpoint,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": _SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        "stream": False,
                    },
                    timeout=120,
                )
                last_batch_time = time.time()

                if not resp.ok:
                    logging.warning(
                        "Ollama HTTP %d (attempt %d/3): %s",
                        resp.status_code, attempt + 1, resp.text[:200],
                    )
                    time.sleep(2 ** attempt)
                    continue

                data = resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                results = _parse_json_response(content)
                if results:
                    break
                logging.warning("Ollama returned unparseable response (attempt %d/3)", attempt + 1)
                time.sleep(2 ** attempt)

            except Exception as exc:
                logging.warning("Ollama call failed (attempt %d/3): %s", attempt + 1, exc)
                time.sleep(2 ** attempt)

        if not results:
            logging.warning("Batch %d/%d failed after 3 attempts — skipping.", batch_num + 1, num_batches)
            continue

        # Apply results
        now = int(time.time())
        cache_rows = []
        for item in results:
            try:
                batch_idx = int(item.get("idx", 0)) - 1
                if batch_idx < 0 or batch_idx >= len(batch):
                    continue
                df_idx, key, _ = batch[batch_idx]

                mood = str(item.get("mood", "")).strip()
                energy = item.get("energy")
                valence = item.get("valence")
                tags = item.get("tags", [])

                if mood:
                    df.at[df_idx, "Mood"] = mood
                if energy is not None:
                    df.at[df_idx, "Energy"] = int(energy)
                if valence is not None:
                    df.at[df_idx, "Valence"] = int(valence)

                cache_rows.append((
                    key, mood,
                    int(energy) if energy is not None else None,
                    int(valence) if valence is not None else None,
                    json.dumps(tags) if tags else None,
                    now,
                ))
                enriched_count += 1
            except (KeyError, ValueError, TypeError) as exc:
                logging.debug("Skipping malformed enrichment item: %s", exc)

        # Batch write to cache
        if conn and cache_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO ollama_enrichment "
                "(key, mood, energy, valence, tags, enriched_at) VALUES (?, ?, ?, ?, ?, ?)",
                cache_rows,
            )
            conn.commit()

        logging.info(
            "Batch %d/%d: enriched %d tracks", batch_num + 1, num_batches, len(cache_rows)
        )

    if conn:
        conn.close()

    logging.info(
        "Ollama enrichment complete: %d/%d tracks enriched.",
        enriched_count, len(to_enrich),
    )
    return df
