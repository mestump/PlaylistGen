import json
from pathlib import Path
from typing import Dict


def load_feedback(path: str) -> Dict:
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_feedback(path: str, data: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_feedback(path: str, playlist_name: str, action: str) -> None:
    data = load_feedback(path)
    actions = data.setdefault(playlist_name, [])
    actions.append(action)
    save_feedback(path, data)
