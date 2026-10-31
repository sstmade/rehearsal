"""Rehearsal MVP."""
import os
from pathlib import Path


def load_env():
    """Read KEY=VALUE lines from rehearsal-mvp/.env into os.environ (existing variables win)."""
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
