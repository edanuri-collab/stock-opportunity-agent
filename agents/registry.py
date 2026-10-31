"""Finds every agent: the core Python agents plus spec files.

* ``agents/library/*.toml``: strategies shipped with the project.
* ``agents/custom/*.toml``: strategies created by the user (e.g. with the dashboard builder).

A broken spec file never stops the others from loading; it is reported in ``errors``.
"""
from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from agents import default_agents
from agents.base_agent import BaseAgent
from agents.configurable import build_agent
from agents.spec import AgentSpec, SpecError, slug, to_toml

AGENTS_DIR = Path(__file__).resolve().parent
LIBRARY_DIR = AGENTS_DIR / "library"
CUSTOM_DIR = AGENTS_DIR / "custom"
SPEC_DIRS = (LIBRARY_DIR, CUSTOM_DIR)


@dataclass
class Registry:
    agents: List[BaseAgent] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)      # file -> problem
    sources: Dict[str, str] = field(default_factory=dict)     # agent name -> core|library|custom


def spec_files(dirs: Sequence[Path] = SPEC_DIRS) -> List[Path]:
    return sorted(p for d in dirs if Path(d).is_dir() for p in Path(d).glob("*.toml"))


def read_spec(path: Path) -> AgentSpec:
    try:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SpecError(f"cannot read file: {exc}") from exc
    return AgentSpec.from_dict(raw)


def load_registry(dirs: Sequence[Path] = SPEC_DIRS) -> Registry:
    reg = Registry()
    for agent in default_agents():
        reg.agents.append(agent)
        reg.sources[agent.name] = "core"
    for path in spec_files(dirs):
        try:
            spec = read_spec(path)
        except SpecError as exc:
            reg.errors[path.name] = str(exc)
            continue
        if spec.name in reg.sources:
            reg.errors[path.name] = f"duplicate agent name {spec.name!r}"
            continue
        reg.agents.append(build_agent(spec))
        reg.sources[spec.name] = path.parent.name
    return reg


def fingerprint(dirs: Sequence[Path] = SPEC_DIRS) -> str:
    """Changes whenever a spec file is added, removed or edited (used as a cache key)."""
    digest = hashlib.sha256()
    for path in spec_files(dirs):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def save_spec(spec: AgentSpec, directory: Path = CUSTOM_DIR, overwrite: bool = False) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{slug(spec.name)}.toml"
    if target.exists() and not overwrite:
        raise SpecError(f"an agent file named {target.name} already exists")
    target.write_text(to_toml(spec), encoding="utf-8")
    return target


def spec_of(agent: BaseAgent) -> Tuple[AgentSpec, bool]:
    """Spec of any agent; core agents are expressed as an equivalent weights spec.
    Returns (spec, exact) where exact is False for core agents with custom code."""
    spec = getattr(agent, "spec", None)
    if spec is not None:  # duck-typed: survives module reloads (e.g. Streamlit hot reload)
        return spec, True
    raw = {"name": agent.name, "family": agent.family, "description": agent.description,
           "origin": "original", "risk_penalty": agent.risk_penalty,
           "weights": {k: v for k, v in agent.weights.items() if k != "noise"} or {"value": 1}}
    return AgentSpec.from_dict(raw), False
