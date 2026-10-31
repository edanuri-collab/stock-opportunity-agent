"""Runtime configuration, read from environment variables (no secrets, no hard-coding)."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    n_stocks: int = 120
    n_periods: int = 24
    top_n: int = 10
    random_trials: int = 200
    experiment_dir: str = "experiments"

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            seed=_int_from_env("SOA_SEED", d.seed),
            n_stocks=_int_from_env("SOA_N_STOCKS", d.n_stocks),
            n_periods=_int_from_env("SOA_N_PERIODS", d.n_periods),
            top_n=_int_from_env("SOA_TOP_N", d.top_n),
            random_trials=_int_from_env("SOA_RANDOM_TRIALS", d.random_trials),
            experiment_dir=os.environ.get("SOA_EXPERIMENT_DIR") or d.experiment_dir,
        )
