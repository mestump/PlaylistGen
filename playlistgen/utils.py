import re

def sanitize_label(label: str) -> str:
    """Sanitize label for safe filesystem use."""
    # Replace / and \ with ' - '
    lbl = label.replace('/', ' - ').replace('\\', ' - ')
    # Remove illegal filesystem characters
    for ch in '<>:"|?*':
        lbl = lbl.replace(ch, '')
    # Collapse whitespace and trim
    return ' '.join(lbl.split()).rstrip('& ').strip()
