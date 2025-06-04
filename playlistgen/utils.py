import re


def sanitize_label(label: str) -> str:
    """Sanitize label for safe filesystem use."""
    # Replace / and \ with ' - '
    lbl = label.replace("/", " - ").replace("\\", " - ")
    # Remove illegal filesystem characters
    for ch in '<>:"|?*':
        lbl = lbl.replace(ch, "")
    # Collapse whitespace and trim
    return " ".join(lbl.split()).rstrip("& ").strip()


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
