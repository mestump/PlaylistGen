"""Interactive CLI-based text user interface using ``questionary``."""

from __future__ import annotations

import logging
from pathlib import Path

import questionary
from questionary import Separator, Choice

from .config import load_config, save_config
from .pipeline import run_pipeline
from .seed_playlist import build_seed_playlist
from .utils import validate_path, validate_url


# ---------------------------------------------------------------------------
# Status / helpers
# ---------------------------------------------------------------------------

def _enrichment_cache_count() -> int:
    """Return the number of tracks already enriched in the SQLite cache."""
    cache_db = Path.home() / ".playlistgen" / "claude_enrichment.sqlite"
    if not cache_db.exists():
        return 0
    try:
        import sqlite3
        conn = sqlite3.connect(str(cache_db))
        n = conn.execute("SELECT COUNT(*) FROM claude_enrichment").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


def _status_line(cfg: dict) -> str:
    """Return a one-line summary of what features are currently active."""
    parts = []

    ai_key = cfg.get("ANTHROPIC_API_KEY")
    if ai_key:
        parts.append("AI: enabled (API)")
    else:
        cached = _enrichment_cache_count()
        if cached:
            parts.append(f"AI: {cached} tracks enriched (paste-in cache)")
        else:
            parts.append("AI: no API key (paste-in available)")

    history = cfg.get("SPOTIFY_HISTORY_PATH")
    if history and Path(str(history)).exists():
        parts.append("session model: ready")
    elif history:
        parts.append("session model: path not found")
    else:
        parts.append("session model: not configured")

    lastfm = cfg.get("LASTFM_API_KEY")
    if lastfm:
        parts.append("Last.fm: enabled")

    return "  " + " | ".join(parts)


def _test_spotify_path(path_str: str) -> tuple[bool, str]:
    """
    Try loading the Spotify history at path_str.
    Returns (ok, summary_message).
    """
    from pathlib import Path as _Path
    p = _Path(path_str)
    if not p.exists():
        return False, f"Path not found: {p}"
    try:
        from .session_model import load_streaming_history
        df = load_streaming_history(str(p))
        if df.empty:
            return False, "No streaming history records found in that path."
        n_plays = len(df)
        n_tracks = df["track_id"].nunique() if "track_id" in df.columns else "?"
        ts_min = df["timestamp"].min() if "timestamp" in df.columns else None
        ts_max = df["timestamp"].max() if "timestamp" in df.columns else None
        date_min = ts_min.strftime("%Y-%m-%d") if ts_min is not None and pd.notna(ts_min) else "?"
        date_max = ts_max.strftime("%Y-%m-%d") if ts_max is not None and pd.notna(ts_max) else "?"
        return True, (
            f"{n_plays:,} plays  |  {n_tracks:,} unique tracks  |  "
            f"{date_min} – {date_max}"
        )
    except Exception as exc:
        return False, f"Error loading history: {exc}"


def _welcome_first_run(cfg: dict) -> bool:
    """
    Show a setup wizard if this looks like a first run (no library configured).
    Returns True if setup was completed, False if user skipped.
    """
    itunes_json = Path(cfg.get("ITUNES_JSON", "itunes_slimmed.json"))
    itunes_xml_str = cfg.get("ITUNES_XML", "iTunes Music Library.xml")
    itunes_xml = Path(itunes_xml_str) if itunes_xml_str else None
    has_library = itunes_json.exists() or (itunes_xml is not None and itunes_xml.exists())

    if has_library:
        return False  # not first run

    print()
    print("Welcome to PlaylistGen!")
    print("It looks like this is your first run. Let's get you set up.")
    print()

    run_setup = questionary.confirm(
        "Run the quick setup wizard?", default=True
    ).ask()
    if not run_setup:
        return False

    # Library source
    source = questionary.select(
        "Where is your music library?",
        choices=[
            Choice("iTunes / Music.app (export XML file)", value="itunes"),
            Choice("Local folder of music files", value="dir"),
        ],
    ).ask()

    if source == "itunes":
        print()
        print("To export your iTunes library:")
        print("  1. Open iTunes or Music.app")
        print("  2. Go to File → Library → Export Library")
        print("  3. Save the .xml file in this directory")
        print()
        xml_path = questionary.text(
            "Path to iTunes Library XML",
            default=str(itunes_xml),
        ).ask()
        if xml_path:
            try:
                cfg["ITUNES_XML"] = validate_path(xml_path, must_exist=True)
            except ValueError as exc:
                print(f"  Warning: {exc} — path stored but may not work.")
                cfg["ITUNES_XML"] = xml_path
    else:
        lib_dir = questionary.text(
            "Path to your music folder",
            default=str(Path.home() / "Music"),
        ).ask()
        if lib_dir:
            try:
                cfg["LIBRARY_DIR"] = validate_path(lib_dir, must_exist=True)
            except ValueError as exc:
                print(f"  Warning: {exc} — path stored but may not work.")
                cfg["LIBRARY_DIR"] = lib_dir

    # Spotify history
    print()
    use_history = questionary.confirm(
        "Do you have a Spotify streaming history export? (spotify.com → Privacy → Download data)",
        default=False,
    ).ask()
    if use_history:
        while True:
            history_path = questionary.text(
                "Path to folder containing your Spotify JSON export files (or a single .json file):",
            ).ask()
            if not history_path:
                break
            ok, summary = _test_spotify_path(history_path)
            if ok:
                print(f"  ✓  Loaded: {summary}")
                cfg["SPOTIFY_HISTORY_PATH"] = history_path
                break
            else:
                print(f"  ✗  {summary}")
                retry = questionary.confirm("Try a different path?", default=True).ask()
                if not retry:
                    print("  Skipping Spotify history for now.")
                    print("  You can add it later via Setup → Re-configure Spotify history.")
                    break

    # AI key
    print()
    use_ai = questionary.confirm(
        "Do you have an Anthropic API key? (enables AI mood tagging and smart playlist curation)",
        default=False,
    ).ask()
    if use_ai:
        ai_key = questionary.text("Anthropic API key (sk-ant-...)").ask()
        if ai_key:
            cfg["ANTHROPIC_API_KEY"] = ai_key
            cfg["AI_ENHANCE"] = True
    else:
        print()
        print("No problem — you can still use AI features without an API key.")
        print("Use  'Generate AI prompt (paste-in)'  from the main menu to get a")
        print("prompt you paste into Claude.ai, ChatGPT, or Gemini for free.")

    save_config(cfg)
    print()
    print("Setup saved. Generating your first mix now...")
    print()
    return True


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------

def edit_tokens(cfg: dict) -> None:
    """Walk through API keys and data paths, saving only non-empty values."""
    print()
    print("Configure API keys and data paths")
    print("(Press Enter to keep the current value; enter a space to clear a field)")
    print()

    fields = [
        ("ANTHROPIC_API_KEY",
         "Anthropic API key (enables AI mood tagging and playlist curation)",
         "sk-ant-..."),
        ("LASTFM_API_KEY",
         "Last.fm API key (optional fallback for mood tags — visit last.fm/api/account/create)",
         ""),
        ("SPOTIFY_HISTORY_PATH",
         "Spotify streaming history path (folder of JSON files from spotify.com privacy export)",
         ""),
        ("SPOTIFY_CLIENT_ID",
         "Spotify Client ID (only needed for Discover mode — see developer.spotify.com)",
         ""),
        ("SPOTIFY_CLIENT_SECRET",
         "Spotify Client Secret (only needed for Discover mode)",
         ""),
        ("OLLAMA_BASE_URL",
         "Ollama backend URL for AI inference", 
         "http://localhost:11434" ),
        ("OLLAMA_MODEL",
         "Ollama model name to use (default: hf.co/unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS)",
         "hf.co/unsloth/Qwen3.5-35B-A3B-GGUF:UD-IQ2_XXS"),
    ]

    for key, prompt, placeholder in fields:
        current = str(cfg.get(key) or "")
        display_default = current if current else (f"[{placeholder}]" if placeholder else "[not set]")
        answer = questionary.text(
            prompt,
            default=current,
        ).ask()
        if answer is None:
            continue
        answer = answer.strip()
        if answer == "":
            # Entered a space or just pressed enter on empty — clear it
            cfg.pop(key, None)
        else:
            # Validate URLs before storing
            if "URL" in key and answer:
                try:
                    validate_url(answer)
                except ValueError as exc:
                    print(f"  Warning: {exc}")
                    continue
            # Validate file paths before storing
            if "PATH" in key and answer and "URL" not in key:
                try:
                    answer = validate_path(answer, must_exist=True)
                except ValueError as exc:
                    print(f"  Warning: {exc} — stored anyway.")
            cfg[key] = answer

    # Auto-enable AI features if key was just set
    if cfg.get("ANTHROPIC_API_KEY"):
        cfg["AI_ENHANCE"] = True

    save_config(cfg)
    print()
    print("Settings saved.")


def edit_config(cfg: dict) -> None:
    """Generic key/value config editor for advanced users."""
    print()
    print("Edit a raw config value (advanced). Type the key name:")

    # Group config keys by category for easier navigation
    categorized = {}
    prefixes = {
        "ITUNES": "Library",
        "SPOTIFY": "Spotify",
        "OUTPUT": "Output",
        "CLUSTER": "Clustering",
        "AI": "AI Features",
        "LASTFM": "Last.fm",
        "AUDIO": "Audio Analysis",
        "LIBROSA": "Audio Analysis",
        "SESSION": "Session Model",
        "RECENCY": "Session Model",
        "TRACKS": "Playlist",
        "MAX": "Playlist",
        "NUM": "Playlist",
        "YEAR": "Clustering",
        "FEEDBACK": "Advanced",
        "TAG": "Advanced",
        "PROFILE": "Advanced",
    }
    for k in sorted(cfg.keys()):
        category = "Other"
        for prefix, cat in prefixes.items():
            if k.startswith(prefix):
                category = cat
                break
        categorized.setdefault(category, []).append(k)

    choices = []
    for cat in sorted(categorized):
        choices.append(Separator(f"── {cat} ──"))
        for k in categorized[cat]:
            val = cfg.get(k, "")
            # Mask API keys
            display_val = ("***" if ("KEY" in k or "SECRET" in k or "TOKEN" in k) and val
                           else str(val))
            choices.append(Choice(f"{k}  [{display_val}]", value=k))
    choices.append(Separator())
    choices.append(Choice("Cancel", value=None))

    key = questionary.select("Select a setting to edit:", choices=choices).ask()
    if not key:
        return

    val = questionary.text(f"New value for {key}:", default=str(cfg.get(key, ""))).ask()
    if val is not None:
        # Validate URL-type keys
        if "URL" in key and val.strip():
            try:
                validate_url(val.strip())
            except ValueError as exc:
                print(f"  Invalid URL: {exc}")
                return
        # Validate path-type keys
        if "PATH" in key and "URL" not in key and val.strip():
            try:
                val = validate_path(val.strip(), must_exist=False)
            except ValueError as exc:
                print(f"  Invalid path: {exc}")
                return
        cfg[key] = val
        save_config(cfg)
        print(f"  {key} updated.")


# ---------------------------------------------------------------------------
# Spotify export helper
# ---------------------------------------------------------------------------

def _handle_export_spotify(cfg: dict) -> None:
    """Let the user pick an M3U file and export it to their Spotify account."""
    from .spotify_export import export_playlist_to_spotify

    out_dir = cfg.get("OUTPUT_DIR", "./mixes")
    m3u_dir = Path(out_dir).expanduser()

    if not m3u_dir.is_dir():
        print(f"  No playlist output directory found at {m3u_dir}")
        print("  Generate a mix first, then export it.")
        return

    m3u_files = sorted(m3u_dir.glob("*.m3u"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not m3u_files:
        print(f"  No .m3u files found in {m3u_dir}")
        print("  Generate a mix first, then export it to Spotify.")
        return

    choices = [Choice(f.stem, value=str(f)) for f in m3u_files[:20]]
    choices.append(Choice("Cancel", value=None))

    selected = questionary.select(
        "Select a playlist to export to Spotify:",
        choices=choices,
    ).ask()
    if not selected:
        return

    m3u_path = Path(selected)
    playlist_name = questionary.text(
        "Spotify playlist name:",
        default=m3u_path.stem,
    ).ask()
    if not playlist_name:
        return

    private = questionary.confirm("Create as private playlist?", default=False).ask()

    # Parse M3U to get Artist - Name pairs
    tracks = []
    for line in m3u_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            info = line.split(",", 1)
            if len(info) == 2:
                parts = info[1].split(" - ", 1)
                if len(parts) == 2:
                    tracks.append({"Artist": parts[0].strip(), "Name": parts[1].strip()})

    if not tracks:
        print("  No tracks found in the M3U file.")
        return

    print(f"\n  Exporting {len(tracks)} tracks to Spotify…")

    import pandas as pd
    playlist_df = pd.DataFrame(tracks)
    url = export_playlist_to_spotify(
        playlist_df,
        playlist_name,
        cfg=cfg,
        public=not private,
    )

    if url:
        print(f"\n  Spotify playlist created: {url}")
    else:
        print("\n  Export failed. Check that SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are set.")
        print("  Configure them via 'Configure API keys & data paths' in the menu.")


# ---------------------------------------------------------------------------
# Spotify history setup helper
# ---------------------------------------------------------------------------

def _handle_spotify_setup(cfg: dict) -> None:
    """Let the user set (or correct) the Spotify streaming history path with live validation."""
    current = cfg.get("SPOTIFY_HISTORY_PATH", "")
    print()
    if current:
        ok, summary = _test_spotify_path(current)
        if ok:
            print(f"Current path: {current}")
            print(f"  Status  : ✓  {summary}")
        else:
            print(f"Current path: {current}")
            print(f"  Status  : ✗  {summary}")
    else:
        print("Spotify streaming history is not configured.")
    print()
    print("Download your data at spotify.com → Account → Privacy settings → Download your data")
    print("Then unzip and point to the folder containing the JSON files.")
    print()

    while True:
        new_path = questionary.text(
            "New path (or press Enter to keep current / skip):",
            default=current,
        ).ask()
        if new_path is None or new_path.strip() == current:
            return
        new_path = new_path.strip()
        if not new_path:
            return
        ok, summary = _test_spotify_path(new_path)
        if ok:
            print(f"  ✓  {summary}")
            cfg["SPOTIFY_HISTORY_PATH"] = new_path
            from .config import save_config
            save_config(cfg)
            print("  Saved.")
            return
        else:
            print(f"  ✗  {summary}")
            retry = questionary.confirm("Try a different path?", default=True).ask()
            if not retry:
                return


# ---------------------------------------------------------------------------
# Paste-in AI workflow helpers
# ---------------------------------------------------------------------------

def _load_library(cfg: dict):
    """Load the library DataFrame from iTunes JSON or directory scan."""
    from .pipeline import ensure_itunes_json
    from .itunes import load_itunes_json

    lib_dir = cfg.get("LIBRARY_DIR")
    if lib_dir and Path(lib_dir).exists():
        from .itunes import build_library_from_dir
        return build_library_from_dir(lib_dir)

    itunes_json = ensure_itunes_json(cfg)
    return load_itunes_json(str(itunes_json))


def _pending_batch_count(library_df, batch_size: int, cache_db: str | None = None) -> int:
    """Return how many batches of unenriched tracks exist."""
    import sqlite3
    if cache_db is None:
        cache_db = str(Path.home() / ".playlistgen" / "claude_enrichment.sqlite")
    cached_keys: set = set()
    if Path(cache_db).exists():
        try:
            conn = sqlite3.connect(cache_db)
            rows = conn.execute("SELECT key FROM claude_enrichment").fetchall()
            cached_keys = {r[0] for r in rows}
            conn.close()
        except Exception:
            pass
    import pandas as pd
    n = 0
    for _, row in library_df.iterrows():
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
        n += 1
    return max(1, (n + batch_size - 1) // batch_size)


def _handle_export_session(cfg: dict) -> None:
    """Generate a full-library Claude session .md file (all batches in one upload)."""
    from .prompt_io import export_enrichment_session

    print()
    print("Building full-library Claude enrichment session file…")

    try:
        library_df = _load_library(cfg)
    except Exception as exc:
        print(f"Could not load library: {exc}")
        return

    batch_str = questionary.text(
        "Tracks per batch (300 is safe for Claude.ai — it handles large output windows well):",
        default="300",
    ).ask()
    try:
        batch_size = int(batch_str or "300")
    except ValueError:
        batch_size = 300

    try:
        export_enrichment_session(library_df, batch_size=batch_size)
    except Exception as exc:
        logging.exception("Session export failed")
        print(f"Export failed: {exc}")


def _handle_paste_enrich(cfg: dict) -> None:
    """Export a single-batch enrichment prompt, guide the user, then import."""
    from .prompt_io import export_enrichment_prompt, import_enrichment_result

    print()
    print("Generating AI enrichment prompt…")

    try:
        library_df = _load_library(cfg)
    except Exception as exc:
        print(f"Could not load library: {exc}")
        return

    batch_str = questionary.text(
        "Max tracks per prompt (300 for Claude.ai/ChatGPT Plus; 100 for Gemini free):",
        default="300",
    ).ask()
    try:
        batch_size = int(batch_str or "300")
    except ValueError:
        batch_size = 300

    n_batches = _pending_batch_count(library_df, batch_size)
    if n_batches > 1:
        print()
        print(f"  Your library needs {n_batches} batches to fully enrich.")
        print("  Tip: use 'Generate Claude session file' to create a single upload")
        print("  that tells Claude to process all batches automatically.")
        print()

    try:
        out = export_enrichment_prompt(library_df, batch_size=batch_size)
    except Exception as exc:
        print(f"Export failed: {exc}")
        return

    print()
    print("─" * 60)
    print("  STEPS")
    print("─" * 60)
    print(f"  1. Open: {out.resolve()}")
    print("  2. Copy the section between PROMPT START and PROMPT END")
    print("  3. Paste into Claude.ai, ChatGPT, Gemini, or any AI")
    print("  4. Copy the AI's entire JSON response")
    print("  5. Paste it into the RESPONSE section at the bottom of the file")
    print("─" * 60)
    print()

    ready = questionary.confirm(
        "Ready to import the AI response from the file?",
        default=False,
    ).ask()
    if not ready:
        print()
        print(f"No problem — come back and run 'Import AI response from file'")
        print(f"when you're ready, or run:")
        print(f"  playlistgen import-ai-result \"{out.name}\"")
        return

    try:
        import_enrichment_result(str(out), library_df)
    except ValueError as exc:
        print()
        print(f"Import failed: {exc}")
        print("Make sure you pasted the AI's JSON into the RESPONSE section of the file.")
    except Exception as exc:
        logging.exception("Enrichment import failed")
        print(f"Import failed: {exc}")


def _handle_paste_curate(cfg: dict) -> None:
    """Export a curation prompt, guide the user through the AI step, then import."""
    from .tag_mood_service import load_tag_mood_db
    from .scoring import score_tracks
    from .prompt_io import export_curation_prompt, import_curation_result
    from .playlist_builder import save_m3u

    print()
    print("Scoring your library for curation prompt…")

    try:
        library_df = _load_library(cfg)
    except Exception as exc:
        print(f"Could not load library: {exc}")
        return

    tag_db = load_tag_mood_db()
    scored_df = score_tracks(library_df, tag_mood_db=tag_db)

    n_str = questionary.text(
        "How many playlists should the AI create?",
        default="6",
    ).ask()
    try:
        n_playlists = int(n_str or "6")
    except ValueError:
        n_playlists = 6

    try:
        out = export_curation_prompt(scored_df, n_playlists=n_playlists)
    except Exception as exc:
        print(f"Export failed: {exc}")
        return

    print()
    print("─" * 60)
    print("  STEPS")
    print("─" * 60)
    print(f"  1. Open: {out.resolve()}")
    print("  2. Copy the section between PROMPT START and PROMPT END")
    print("  3. Paste into Claude.ai, ChatGPT, Gemini, or any AI")
    print("  4. Copy the AI's entire JSON response")
    print("  5. Paste it into the RESPONSE section at the bottom of the file")
    print("─" * 60)
    print()

    ready = questionary.confirm(
        "Ready to import the AI response and write playlists?",
        default=False,
    ).ask()
    if not ready:
        print()
        print(f"No problem — come back and run 'Import AI response from file'")
        print(f"when you're ready, or run:")
        print(f"  playlistgen import-ai-result \"{out.name}\"")
        return

    try:
        playlists = import_curation_result(str(out), scored_df)
    except ValueError as exc:
        print()
        print(f"Import failed: {exc}")
        print("Make sure you pasted the AI's JSON into the RESPONSE section of the file.")
        return
    except Exception as exc:
        logging.exception("Curation import failed")
        print(f"Import failed: {exc}")
        return

    out_dir = cfg.get("OUTPUT_DIR", "./mixes")
    for label, playlist_df in playlists:
        save_m3u(playlist_df, label, out_dir=out_dir)
    print(f"  {len(playlists)} playlists written to {out_dir}/")


def _handle_import_ai(cfg: dict) -> None:
    """Import a previously generated prompt file with AI response pasted in."""
    from .prompt_io import _detect_mode, import_enrichment_result, import_curation_result
    from .tag_mood_service import load_tag_mood_db
    from .scoring import score_tracks
    from .playlist_builder import save_m3u

    print()
    file_path = questionary.text(
        "Path to the prompt .txt file (or a plain .json response file):",
        placeholder="playlistgen_enrich_prompt.txt",
    ).ask()
    if not file_path:
        return

    file_path = file_path.strip()
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    raw = Path(file_path).read_text(encoding="utf-8")
    mode = _detect_mode(raw)
    print(f"  Detected mode: {mode}")

    try:
        library_df = _load_library(cfg)
    except Exception as exc:
        print(f"Could not load library: {exc}")
        return

    if mode == "enrich":
        try:
            import_enrichment_result(file_path, library_df)
        except ValueError as exc:
            print(f"Import failed: {exc}")
        except Exception as exc:
            logging.exception("Import failed")
            print(f"Import failed: {exc}")
    else:
        tag_db = load_tag_mood_db()
        scored_df = score_tracks(library_df, tag_mood_db=tag_db)
        try:
            playlists = import_curation_result(file_path, scored_df)
        except ValueError as exc:
            print(f"Import failed: {exc}")
            return
        except Exception as exc:
            logging.exception("Import failed")
            print(f"Import failed: {exc}")
            return

        out_dir = cfg.get("OUTPUT_DIR", "./mixes")
        for label, playlist_df in playlists:
            save_m3u(playlist_df, label, out_dir=out_dir)
        print(f"  {len(playlists)} playlists written to {out_dir}/")


# ---------------------------------------------------------------------------
# Main GUI loop
# ---------------------------------------------------------------------------

def run_gui() -> str | None:
    """Launch an interactive text user interface and execute the chosen action."""
    logging.basicConfig(level=logging.WARNING)  # keep GUI clean; INFO goes to log file

    cfg = load_config()

    # First-run wizard
    if _welcome_first_run(cfg):
        cfg = load_config()  # reload after wizard saves
        library_dir = cfg.get("LIBRARY_DIR")
        run_pipeline(cfg, library_dir=library_dir)
        return "generate_mix"

    while True:
        print()
        print(_status_line(cfg))
        print()

        choices = [
            Separator("── Generate Playlists ──"),
            Choice(
                "Smart mix  (auto-selects best clustering strategy)",
                value="generate_mix",
            ),
            Choice(
                "Mix from a seed song  (builds a playlist around one track)",
                value="seed",
            ),
            Choice(
                "Filter by mood  (e.g. Chill, Energetic, Sad, Happy…)",
                value="mood",
            ),
            Choice(
                "Filter by genre  (e.g. Rock, Jazz, Hip-Hop…)",
                value="genre",
            ),
            Separator("── AI Features (API key) ──"),
            Choice(
                "Claude: Smart playlist curation  (Claude groups tracks into themed playlists)",
                value="ai_curate",
            ),
            Choice(
                "Claude: Enrich library metadata  (Claude adds mood/energy tags to your whole library)",
                value="ai_enrich",
            ),
            Separator("── AI Features (no API key needed) ──"),
            Choice(
                "Generate Claude session file   (one upload → all batches; artifacts per batch)",
                value="export_session",
            ),
            Choice(
                "Generate AI enrichment prompt  (single batch — paste into ChatGPT / Gemini / etc.)",
                value="paste_enrich",
            ),
            Choice(
                "Generate AI curation prompt    (paste into Claude.ai / ChatGPT / Gemini — free)",
                value="paste_curate",
            ),
            Choice(
                "Import AI response from file   (after pasting the AI's JSON response back)",
                value="import_ai",
            ),
            Separator("── Export ──"),
            Choice(
                "Export last playlist to Spotify  (push an M3U to your Spotify account)",
                value="export_spotify",
            ),
            Separator("── Setup & Maintenance ──"),
            Choice(
                "Configure API keys & data paths  (Anthropic, Last.fm, Spotify history)",
                value="settings",
            ),
            Choice(
                "Re-configure Spotify history     (test & update your streaming history path)",
                value="spotify_setup",
            ),
            Choice(
                "Refresh metadata cache           (re-fetch Last.fm tags for new or updated tracks)",
                value="recache",
            ),
            Separator("── Advanced ──"),
            Choice(
                "Edit a config value  (advanced settings)",
                value="config",
            ),
            Choice("Exit", value="exit"),
        ]

        action = questionary.select(
            "What would you like to do?",
            choices=choices,
        ).ask()

        if action is None or action == "exit":
            return action

        _handle_action(action, cfg)

        # Reload config after each action (settings may have changed)
        cfg = load_config()

        print()
        again = questionary.confirm("Return to main menu?", default=True).ask()
        if not again:
            return action


def _handle_action(action: str, cfg: dict) -> None:
    """Dispatch a menu action."""

    if action == "generate_mix":
        print()
        print("Generating smart mix…")
        print("(Tip: use 'Filter by mood' or 'Filter by genre' for a focused playlist)")
        print()
        run_pipeline(cfg)

    elif action == "seed":
        print()
        song = questionary.text(
            "Enter a seed song in the format  'Artist - Title':",
            placeholder="e.g. Radiohead - Karma Police",
        ).ask()
        if not song:
            return
        num_str = questionary.text(
            "How many tracks in the playlist?",
            default="20",
        ).ask()
        try:
            num = int(num_str or "20")
        except ValueError:
            num = 20
        print()
        print(f"Building playlist around '{song}'…")
        build_seed_playlist(song, cfg=cfg, limit=num)

    elif action == "mood":
        print()
        mood = questionary.select(
            "Select a mood:",
            choices=[
                Choice("Happy  — upbeat, feel-good tracks", value="Happy"),
                Choice("Sad  — melancholy, bittersweet", value="Sad"),
                Choice("Angry  — aggressive, high-energy", value="Angry"),
                Choice("Chill  — relaxed, laid-back", value="Chill"),
                Choice("Energetic  — driving, pumping", value="Energetic"),
                Choice("Romantic  — love songs, ballads", value="Romantic"),
                Choice("Epic  — anthemic, cinematic", value="Epic"),
                Choice("Dreamy  — ethereal, ambient", value="Dreamy"),
                Choice("Groovy  — funky, soulful", value="Groovy"),
                Choice("Nostalgic  — retro, throwback", value="Nostalgic"),
                Choice("Enter manually…", value="_custom"),
            ],
        ).ask()
        if not mood:
            return
        if mood == "_custom":
            mood = questionary.text("Mood name:").ask()
            if not mood:
                return
        print()
        print(f"Generating {mood} playlist…")
        run_pipeline(cfg, mood=mood)

    elif action == "genre":
        print()
        genre = questionary.text(
            "Enter a genre (e.g. Rock, Jazz, Hip-Hop, Classical):",
            placeholder="Rock",
        ).ask()
        if not genre:
            return
        print()
        print(f"Generating {genre} playlist…")
        run_pipeline(cfg, genre=genre)

    elif action == "ai_curate":
        if not cfg.get("ANTHROPIC_API_KEY"):
            print()
            print("No Anthropic API key configured.")
            print("Use 'Generate AI curation prompt' below to curate playlists")
            print("by pasting a prompt into Claude.ai, ChatGPT, or Gemini — no key needed.")
            return
        print()
        print("Claude is curating your playlists…")
        print("(This sends track metadata to Claude — no audio files leave your computer)")
        print()
        cfg_copy = {**cfg, "AI_CURATE": True}
        run_pipeline(cfg_copy)

    elif action == "ai_enrich":
        if not cfg.get("ANTHROPIC_API_KEY"):
            print()
            print("No Anthropic API key configured.")
            print("Use 'Generate AI enrichment prompt' below to enrich your library")
            print("by pasting a prompt into Claude.ai, ChatGPT, or Gemini — no key needed.")
            return
        print()
        print("Claude is tagging your library with mood and energy data…")
        print("Results are cached — only new or changed tracks are re-analysed on future runs.")
        print()
        try:
            from .pipeline import ensure_itunes_json
            from .itunes import load_itunes_json
            from .ai_enhancer import batch_enrich_metadata

            itunes_json = ensure_itunes_json(cfg)
            library_df = load_itunes_json(str(itunes_json))
            enrich_cache = str(
                Path(
                    cfg.get(
                        "AI_ENRICH_CACHE_DB",
                        Path.home() / ".playlistgen" / "claude_enrichment.sqlite",
                    )
                ).expanduser()
            )
            enriched_df = batch_enrich_metadata(
                library_df,
                api_key=cfg["ANTHROPIC_API_KEY"],
                model=cfg.get("AI_MODEL", "claude-haiku-4-5-20251001"),
                cache_db=enrich_cache,
            )
            mood_count = (
                enriched_df["Mood"].notna() & (enriched_df["Mood"] != "Unknown")
            ).sum()
            print()
            print(f"Done. Mood tagged for {mood_count} / {len(enriched_df)} tracks.")
        except Exception as exc:
            logging.exception("Enrichment failed")
            print(f"Enrichment failed: {exc}")

    elif action == "settings":
        edit_tokens(cfg)
        # Show what's now enabled
        cfg_fresh = load_config()
        print()
        print("Current status:")
        print(_status_line(cfg_fresh))

    elif action == "recache":
        print()
        print("Re-fetching metadata from Last.fm for new or updated tracks…")
        print("(Already-cached tracks are skipped — only new tracks are fetched)")
        print()
        try:
            from .pipeline import ensure_itunes_json, ensure_tag_cache
            itunes_json = ensure_itunes_json(cfg)
            ensure_tag_cache(cfg, itunes_json)
            print("Metadata cache refreshed.")
        except Exception as exc:
            logging.exception("Recache failed")
            print(f"Recache failed: {exc}")

    elif action == "export_spotify":
        _handle_export_spotify(cfg)

    elif action == "spotify_setup":
        _handle_spotify_setup(cfg)

    elif action == "export_session":
        _handle_export_session(cfg)

    elif action == "paste_enrich":
        _handle_paste_enrich(cfg)

    elif action == "paste_curate":
        _handle_paste_curate(cfg)

    elif action == "import_ai":
        _handle_import_ai(cfg)

    elif action == "config":
        edit_config(cfg)
