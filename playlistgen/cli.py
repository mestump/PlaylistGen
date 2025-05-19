"""CLI"""

import argparse
import logging
import os
from pathlib import Path


from .config import load_config
from .pipeline import run_pipeline


def file_newer(a, b):
    """Return True if file `a` exists and is newer than file `b` (or if `b` is missing)."""
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    return os.path.getmtime(a) > os.path.getmtime(b)

def main():
    cfg = load_config()
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
