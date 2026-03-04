import os
import re
from pathlib import Path
from urllib.parse import urlparse


def sanitize_label(label: str) -> str:
    """Sanitize label for safe filesystem use."""
    # Replace / and \ with ' - '
    lbl = label.replace("/", " - ").replace("\\", " - ")
    # Remove illegal filesystem characters
    for ch in '<>:"|?*':
        lbl = lbl.replace(ch, "")
    # Collapse whitespace and trim
    return " ".join(lbl.split()).rstrip("& ").strip()


def validate_path(path: str, must_exist: bool = False, base_dir: str | None = None) -> str:
    """
    Validate and resolve a filesystem path, guarding against path traversal.

    Args:
        path:       The path to validate.
        must_exist: If True, raise ValueError when the resolved path doesn't exist.
        base_dir:   If set, the resolved path must be under this directory.

    Returns:
        The resolved absolute path as a string.

    Raises:
        ValueError: On empty path, path traversal attempt, or non-existent path.
    """
    if not path or not path.strip():
        raise ValueError("Path must not be empty")

    resolved = Path(path).expanduser().resolve()

    if base_dir is not None:
        base = Path(base_dir).expanduser().resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Path traversal blocked: {path!r} resolves outside {base_dir!r}"
            )

    if must_exist and not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")

    return str(resolved)


def validate_url(url: str, allowed_schemes: tuple[str, ...] = ("http", "https")) -> str:
    """
    Validate a URL has an allowed scheme and a non-empty host.

    Returns the URL unchanged if valid, raises ValueError otherwise.
    """
    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise ValueError(
            f"URL scheme {parsed.scheme!r} not allowed (expected {allowed_schemes})"
        )
    if not parsed.hostname:
        raise ValueError(f"URL has no hostname: {url!r}")
    return url


def validate_config(cfg: dict) -> list[str]:
    """
    Validate configuration values. Returns a list of warning messages
    for any invalid values that were corrected to safe defaults.
    """
    warnings = []

    # Numeric ranges
    int_ranges = {
        "CLUSTER_COUNT": (1, 100),
        "MAX_PER_ARTIST": (1, 100),
        "TRACKS_PER_MIX": (1, 10000),
        "AUDIO_ANALYSIS_WORKERS": (0, 64),
        "AUDIO_ANALYSIS_DURATION": (1, 600),
        "SESSION_GAP_MINUTES": (1, 1440),
        "RECENCY_HALF_LIFE_DAYS": (1, 3650),
        "AI_ENRICH_BATCH_SIZE": (1, 1000),
        "LASTFM_RATE_LIMIT_MS": (0, 10000),
    }
    for key, (lo, hi) in int_ranges.items():
        val = cfg.get(key)
        if val is not None:
            try:
                num = int(val)
                if num < lo or num > hi:
                    cfg[key] = max(lo, min(hi, num))
                    warnings.append(
                        f"{key}={val} out of range [{lo}, {hi}], clamped to {cfg[key]}"
                    )
            except (ValueError, TypeError):
                warnings.append(f"{key}={val!r} is not a valid integer")

    # Validate URL if present
    ollama_url = cfg.get("OLLAMA_BASE_URL")
    if ollama_url:
        try:
            validate_url(ollama_url)
        except ValueError as e:
            warnings.append(f"OLLAMA_BASE_URL invalid: {e}")
            cfg["OLLAMA_BASE_URL"] = None

    return warnings


try:
    from tqdm import tqdm

    def progress_bar(iterable, desc: str = "", total: int = None):
        """Unified progress bar with nicer formatting."""
        kwargs = {
            "desc": desc,
            "total": total,
            "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        }
        try:
            kwargs["colour"] = "cyan"
            return tqdm(iterable, **kwargs)
        except TypeError:  # older tqdm without colour support
            kwargs.pop("colour", None)
            return tqdm(iterable, **kwargs)

except Exception:  # pragma: no cover - tqdm missing

    def progress_bar(iterable, desc: str = "", total: int = None):
        return iterable
