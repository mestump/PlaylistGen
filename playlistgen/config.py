from pathlib import Path
import os

try:
    import yaml
except ImportError:
    try:
        from ruamel import yaml
    except ImportError:
        yaml = None

_config_cache = None


def load_config(path: str = None) -> dict:
    global _config_cache
    if _config_cache is not None and path is None:
        return _config_cache

    defaults = {
        # ... your existing defaults ...
        "ITUNES_JSON": "./itunes_slimmed.json",
        "SPOTIFY_DIR": "./spotify_history",
        "PROFILE_PATH": "./taste_profile.json",
        "OUTPUT_DIR": "./mixes",
        "LASTFM_API_KEY": None,
        "CLUSTER_COUNT": 6,
        "MAX_PER_ARTIST": 4,
        "TRACKS_PER_MIX": 50,
        "YEAR_MIX_ENABLED": True,
        "YEAR_MIX_RANGE": 1,
        "SPOTIFY_MOOD_ENABLED": True,
        "ITUNES_MOOD_ENABLED": True,
        "MOOD_CONCURRENCY": 10,
        # Add this line to set a default mood cache path
        "TAG_MOOD_CACHE": str(Path.home() / ".playlistgen" / "lastfm_tags_cache.json"),
        "CACHE_DB": str(Path.home() / ".playlistgen" / "mood_cache.sqlite"),
    }

    # Determine where to load the user config: explicit path, project-root config.yml, or home config
    if path:
        config_path = Path(path)
    else:
        cwd_cfg = Path("config.yml")
        if cwd_cfg.exists():
            config_path = cwd_cfg
        else:
            config_path = Path.home() / ".playlistgen" / "config.yml"
    user_cfg = {}
    if yaml and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                # Use PyYAML or ruamel.yaml safe_load if available
                user_cfg = yaml.safe_load(f) or {}
            except AttributeError:
                # Fallback for ruamel.yaml without safe_load
                try:
                    from ruamel.yaml import YAML

                    user_cfg = YAML(typ="safe", pure=True).load(f) or {}
                except Exception:
                    user_cfg = {}

    merged = {**defaults, **user_cfg}

    # Override with environment variables if present
    for key, default_val in defaults.items():
        env_val = os.getenv(key)
        if env_val is not None:
            if isinstance(default_val, bool):
                merged[key] = env_val.lower() in ("1", "true", "yes")
            elif isinstance(default_val, int):
                merged[key] = int(env_val)
            else:
                merged[key] = env_val

    _config_cache = merged
    return _config_cache
