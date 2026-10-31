"""Content-based versions so an experiment can state exactly what code produced it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from agents.base_agent import BaseAgent

ROOT = Path(__file__).resolve().parent.parent
_CODE_DIRS = ("agents", "data", "engine")


def _hash_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()[:12]


def code_version() -> str:
    """Hash of every source file that influences results (agents, data, engine)."""
    files = [p for d in _CODE_DIRS for p in (ROOT / d).rglob("*.py")]
    return _hash_files(files)


def agent_version(agent: BaseAgent) -> str:
    """Hash of the agent's configuration plus the shared scoring code it runs on."""
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "name": agent.name,
        "weights": dict(agent.weights),
        "risk_penalty": agent.risk_penalty,
    }, sort_keys=True).encode())
    digest.update(_hash_files([ROOT / "engine" / "scoring.py",
                               ROOT / "agents" / "base_agent.py"]).encode())
    return digest.hexdigest()[:8]
