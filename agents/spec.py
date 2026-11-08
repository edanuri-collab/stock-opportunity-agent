"""Declarative agent specifications: an agent described as data instead of code.

A spec has optional *filters* (conditions a stock must pass at time T) and exactly one
scoring method:

* ``weights``: weighted sum of the shared factor scores (like the core agents), or
* ``rank``: cross-sectional percentile rank of data fields among the stocks that passed.

An optional ``switch`` condition (e.g. ``market_trend > 0``) selects the scoring method per
stock: the main one while it holds, the alternative (``alt_weights`` / ``alt_rank``) otherwise.

Every field is computed from the ``StockSnapshot`` alone, so a spec agent can only see what
was knowable at T. Specs are stored as TOML files (see ``agents/registry.py``).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from data.models import StockSnapshot
from engine.scoring import FACTORS, compute_components, fcf_yield

FAMILIES = ("value", "quality", "momentum", "growth", "low_risk", "event", "contrarian",
            "size", "multi_factor", "control")
ORIGINS = ("known", "original", "core")
OPS: Dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
}
PREFER = ("high", "low")
WEIGHT_FACTORS = tuple(f for f in FACTORS if f != "noise")  # noise is reserved for control


def _earnings_yield(s: StockSnapshot) -> float:
    return 1.0 / s.pe_ratio if s.pe_ratio != 0 else 0.0


def f_score(s: StockSnapshot) -> float:
    """Simplified Piotroski F-Score, 0-7. The share-issuance test and the balance-sheet items
    the data model does not carry are left out, so this is not the published 0-9 score."""
    return float(sum((
        s.roe > 0, s.free_cash_flow > 0, s.roe_change > 0,
        fcf_yield(s) > _earnings_yield(s),  # cash earnings above accounting earnings
        s.debt_change < 0, s.gross_margin_change > 0, s.revenue_growth > 0,
    )))


def _factor(name: str) -> Callable[[StockSnapshot], float]:
    return lambda s: getattr(compute_components(s), name)


# field -> (unit kind, getter). Kinds drive display/entry: pct = fraction (0.15 = 15%),
# money = USD, ratio = plain number, score = 0-100.
FIELDS: Dict[str, Tuple[str, Callable[[StockSnapshot], float]]] = {
    "market_cap": ("money", lambda s: s.market_cap),
    "pe_ratio": ("ratio", lambda s: s.pe_ratio),
    "earnings_yield": ("pct", _earnings_yield),
    "price_to_sales": ("ratio", lambda s: s.price_to_sales),
    "fcf_yield": ("pct", fcf_yield),
    "revenue_growth": ("pct", lambda s: s.revenue_growth),
    "earnings_growth": ("pct", lambda s: s.earnings_growth),
    "debt_to_equity": ("ratio", lambda s: s.debt_to_equity),
    "momentum": ("pct", lambda s: s.momentum),
    "relative_strength": ("score", lambda s: s.relative_strength),
    "distance_from_high": ("pct", lambda s: s.distance_from_high),
    "volume_ratio": ("ratio", lambda s: s.volume_ratio),
    "volatility": ("pct", lambda s: s.volatility),
    "catalyst_score": ("score", lambda s: s.catalyst_score),
    "sentiment_score": ("score", lambda s: s.sentiment_score),
    "short_interest": ("pct", lambda s: s.short_interest),
    "days_to_cover": ("ratio", lambda s: s.days_to_cover),
    "price_to_book": ("ratio", lambda s: s.price_to_book),
    "roe": ("pct", lambda s: s.roe),
    "roic": ("pct", lambda s: s.roic),
    "gross_margin": ("pct", lambda s: s.gross_margin),
    "dividend_yield": ("pct", lambda s: s.dividend_yield),
    "earnings_surprise": ("pct", lambda s: s.earnings_surprise),
    "momentum_12_1": ("pct", lambda s: s.momentum_12_1),
    "return_1m": ("pct", lambda s: s.return_1m),
    "beta": ("ratio", lambda s: s.beta),
    "market_trend": ("pct", lambda s: s.market_trend),
    "roe_change": ("pct", lambda s: s.roe_change),
    "gross_margin_change": ("pct", lambda s: s.gross_margin_change),
    "debt_change": ("ratio", lambda s: s.debt_change),
    "f_score": ("ratio", f_score),
    **{f"factor_{f}": ("score", _factor(f)) for f in WEIGHT_FACTORS + ("risk",)},
}


def field_value(stock: StockSnapshot, name: str) -> float:
    return FIELDS[name][1](stock)


class SpecError(ValueError):
    """A spec is invalid; the message says what to fix."""


@dataclass(frozen=True)
class Filter:
    field: str
    op: str
    value: float

    def passes(self, stock: StockSnapshot) -> bool:
        return OPS[self.op](field_value(stock, self.field), self.value)


@dataclass(frozen=True)
class RankField:
    field: str
    prefer: str = "high"
    weight: float = 1.0


@dataclass(frozen=True)
class AgentSpec:
    name: str
    family: str
    description: str
    origin: str = "original"
    reference: str = ""
    risk_penalty: float = 0.0
    filters: Tuple[Filter, ...] = ()
    weights: Tuple[Tuple[str, float], ...] = ()   # normalised to sum 1
    rank: Tuple[RankField, ...] = ()
    switch: Optional[Filter] = None               # when False, the alt_* scoring is used
    alt_weights: Tuple[Tuple[str, float], ...] = ()
    alt_rank: Tuple[RankField, ...] = ()

    @property
    def method(self) -> str:
        if self.switch is not None:
            return "switch"
        return "rank" if self.rank else "weights"

    @property
    def uses_rank(self) -> bool:
        return bool(self.rank or self.alt_rank)

    @property
    def alt_weights_map(self) -> Dict[str, float]:
        return dict(self.alt_weights)

    @property
    def weights_map(self) -> Dict[str, float]:
        return dict(self.weights)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["weights"] = self.weights_map
        d["filters"] = [asdict(f) for f in self.filters]
        d["rank"] = [asdict(r) for r in self.rank]
        d["switch"] = asdict(self.switch) if self.switch else None
        d["alt_weights"] = self.alt_weights_map
        d["alt_rank"] = [asdict(r) for r in self.alt_rank]
        return d

    # ---- construction & validation ---------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentSpec":
        known = {"name", "family", "description", "origin", "reference", "risk_penalty",
                 "filters", "weights", "rank", "switch", "alt_weights", "alt_rank"}
        unknown = set(raw) - known
        if unknown:
            raise SpecError(f"unknown keys: {sorted(unknown)}")
        name = str(raw.get("name", "")).strip()
        if not name or len(name) > 40:
            raise SpecError("name is required (1-40 characters)")
        family = raw.get("family", "")
        if family not in FAMILIES:
            raise SpecError(f"family must be one of {FAMILIES}")
        origin = raw.get("origin", "original")
        if origin not in ("known", "original"):
            raise SpecError("origin must be 'known' or 'original'")
        description = str(raw.get("description", "")).strip()
        if not description:
            raise SpecError("description is required")
        risk_penalty = _number(raw.get("risk_penalty", 0.0), "risk_penalty")
        if not 0 <= risk_penalty <= 1:
            raise SpecError("risk_penalty must be within [0, 1]")

        filters = tuple(_filter(f, i) for i, f in enumerate(raw.get("filters", []) or []))
        weights, rank = _scoring(raw.get("weights"), raw.get("rank"), "")
        switch, alt_weights, alt_rank = None, (), ()
        if raw.get("switch") is not None:
            switch = _filter(raw["switch"], 0, "switch")
            alt_weights, alt_rank = _scoring(raw.get("alt_weights"), raw.get("alt_rank"), "alt_")
        elif raw.get("alt_weights") or raw.get("alt_rank"):
            raise SpecError("alt_weights / alt_rank need a [switch] condition")
        return cls(name=name, family=family, description=description, origin=origin,
                   reference=str(raw.get("reference", "")).strip(),
                   risk_penalty=risk_penalty, filters=filters, weights=weights, rank=rank,
                   switch=switch, alt_weights=alt_weights, alt_rank=alt_rank)


def _scoring(weights_raw: Any, rank_raw: Any, prefix: str
             ) -> Tuple[Tuple[Tuple[str, float], ...], Tuple[RankField, ...]]:
    """Validate one scoring method: exactly one of [<prefix>weights] / [[<prefix>rank]]."""
    weights_raw = weights_raw or {}
    rank = tuple(_rank(r, i, f"{prefix}rank") for i, r in enumerate(rank_raw or []))
    if bool(weights_raw) == bool(rank):
        raise SpecError(f"define exactly one scoring method: [{prefix}weights] or "
                        f"[[{prefix}rank]]")
    if not weights_raw:
        return (), rank
    if not isinstance(weights_raw, Mapping):
        raise SpecError(f"{prefix}weights must be a table")
    bad = set(weights_raw) - set(WEIGHT_FACTORS)
    if bad:
        raise SpecError(f"unknown weight factors {sorted(bad)}; use {WEIGHT_FACTORS}")
    values = {k: _number(v, f"{prefix}weights.{k}") for k, v in weights_raw.items()}
    if any(v < 0 for v in values.values()) or sum(values.values()) <= 0:
        raise SpecError(f"{prefix}weights must be non-negative and not all zero")
    total = sum(values.values())
    return tuple(sorted((k, v / total) for k, v in values.items() if v > 0)), ()


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpecError(f"{what} must be a finite number")
    return float(value)


def _filter(raw: Any, i: int, what: str = "filters") -> Filter:
    where = what if what == "switch" else f"{what}[{i}]"
    if not isinstance(raw, Mapping):
        raise SpecError(f"{where} must be a table")
    f, op = raw.get("field"), raw.get("op")
    if f not in FIELDS:
        raise SpecError(f"{where}: unknown field {f!r}")
    if op not in OPS:
        raise SpecError(f"{where}: op must be one of {tuple(OPS)}")
    return Filter(f, op, _number(raw.get("value"), f"{where}.value"))


def _rank(raw: Any, i: int, what: str = "rank") -> RankField:
    if not isinstance(raw, Mapping):
        raise SpecError(f"{what}[{i}] must be a table")
    f = raw.get("field")
    if f not in FIELDS:
        raise SpecError(f"{what}[{i}]: unknown field {f!r}")
    prefer = raw.get("prefer", "high")
    if prefer not in PREFER:
        raise SpecError(f"{what}[{i}]: prefer must be 'high' or 'low'")
    weight = _number(raw.get("weight", 1.0), f"{what}[{i}].weight")
    if weight <= 0:
        raise SpecError(f"{what}[{i}]: weight must be > 0")
    return RankField(f, prefer, weight)


# ---- TOML output (tomllib can only read) ---------------------------------------------------

def _toml_value(v: Any) -> str:
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, float):
        return repr(round(v, 10))
    return str(v)


def to_toml(spec: AgentSpec) -> str:
    lines = [f"{k} = {_toml_value(getattr(spec, k))}"
             for k in ("name", "family", "origin", "description", "reference", "risk_penalty")]
    for f in spec.filters:
        lines += ["", "[[filters]]", f"field = {_toml_value(f.field)}",
                  f"op = {_toml_value(f.op)}", f"value = {_toml_value(float(f.value))}"]
    lines += _toml_scoring(spec.weights, spec.rank, "")
    if spec.switch is not None:
        s = spec.switch
        lines += ["", "[switch]", f"field = {_toml_value(s.field)}", f"op = {_toml_value(s.op)}",
                  f"value = {_toml_value(float(s.value))}"]
        lines += _toml_scoring(spec.alt_weights, spec.alt_rank, "alt_")
    return "\n".join(lines) + "\n"


def _toml_scoring(weights, rank, prefix: str) -> List[str]:
    lines: List[str] = []
    if weights:
        lines += ["", f"[{prefix}weights]"] + [f"{k} = {_toml_value(v)}" for k, v in weights]
    for r in rank:
        lines += ["", f"[[{prefix}rank]]", f"field = {_toml_value(r.field)}",
                  f"prefer = {_toml_value(r.prefer)}", f"weight = {_toml_value(r.weight)}"]
    return lines


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "agent"


def percentile_ranks(values: List[float]) -> List[float]:
    """0-100 percentile of each value (ties share their average rank; one value -> 50)."""
    n = len(values)
    if n == 1:
        return [50.0]
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return [r / (n - 1) * 100 for r in ranks]


__all__ = ["AgentSpec", "FAMILIES", "FIELDS", "Filter", "OPS", "RankField", "SpecError",
           "WEIGHT_FACTORS", "field_value", "percentile_ranks", "slug", "to_toml"]
