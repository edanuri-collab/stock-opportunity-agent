"""Deterministic synthetic market generator with signal and outcome kept separate.

Timeline of one period (decision date T):

1. Hidden state at T evolves (slow factors, persistent hidden alpha).
2. Features observed at T are produced from that state PLUS measurement noise, and from the
   price path up to T (momentum, 52w high, volatility). Nothing from after T is used.
3. Only then is the outcome after T drawn: market + sector shock + a SMALL signal loading on
   the state at T + a large idiosyncratic shock (fat tailed).

Consequences: features are a weak, noisy proxy of what drives returns (IC of a few %), each
agent concept gets the same small loading, loadings flip sign by regime at random (seeded),
and part of the edge is hidden alpha that no feature observes. A ``SignalConfig.null()``
dataset has no edge at all, so any "alpha" found there is pure luck.
Same arguments -> identical data, and growing ``n_periods`` never changes earlier periods.

The extended fundamentals (P/B, ROE, dividends, ...) are noisy proxies of the same hidden state
and use their own random stream. Low risk, short-term reversal and size get the same weak
loading as every other concept, so no strategy family is favoured or handicapped by design.
"""
from __future__ import annotations

import calendar
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from data.models import MarketDataset, StockSnapshot

REGIMES = ("bull", "bear", "sideways")
WARMUP_PERIODS = 12  # simulated history before the first emitted timestamp (not emitted)
_BASE_SIGNAL_KEYS = ("value", "quality", "catalyst", "momentum", "breakout", "squeeze",
                     "contrarian", "hidden")
_EXTRA_SIGNAL_KEYS = ("reversal", "low_vol", "size")
SIGNAL_KEYS = _BASE_SIGNAL_KEYS + _EXTRA_SIGNAL_KEYS

_REGIME_START_WEIGHTS = (0.45, 0.20, 0.35)
_REGIME_STAY_PROB = 0.7
_REGIME_MARKET = {"bull": (0.012, 0.035), "bear": (-0.015, 0.060), "sideways": (0.002, 0.040)}

_SECTORS = {
    "Technology": ("Software", "Semiconductors", "Hardware"),
    "Healthcare": ("Biotech", "Devices", "Services"),
    "Financials": ("Banks", "Insurance", "Asset Management"),
    "Energy": ("Oil & Gas", "Renewables", "Pipelines"),
    "Consumer": ("Retail", "Food & Beverage", "Leisure"),
    "Industrials": ("Machinery", "Logistics", "Aerospace"),
    "Utilities": ("Electric", "Water", "Gas"),
}
_NAME_PREFIX = ("Apex", "Blue", "Crest", "Delta", "Ember", "Falcon", "Granite", "Harbor",
                "Iron", "Juniper", "Kestrel", "Lumen", "Meridian", "Nova", "Orion",
                "Pioneer", "Quartz", "Ridge", "Summit", "Titan")
_NAME_SUFFIX = ("Systems", "Dynamics", "Labs", "Energy", "Foods", "Biotech", "Networks",
                "Industries", "Capital", "Logistics", "Materials", "Robotics")


@dataclass(frozen=True)
class SignalConfig:
    """Forward-return loading (monthly, per 1 std of the driver). Same for every agent concept."""

    value: float = 0.0015
    quality: float = 0.0015
    catalyst: float = 0.0015
    momentum: float = 0.0015
    breakout: float = 0.0015
    squeeze: float = 0.0015
    contrarian: float = 0.0015
    hidden: float = 0.0015         # edge that no feature observes
    reversal: float = 0.0015
    low_vol: float = 0.0015
    size: float = 0.0015
    regime_dispersion: float = 0.6  # std of the random per-regime multiplier of each loading
    noise_scale: float = 1.0        # multiplies idiosyncratic noise

    @classmethod
    def null(cls) -> "SignalConfig":
        """No relationship between anything observable and the outcome."""
        return cls(**{k: 0.0 for k in SIGNAL_KEYS})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _Stock:
    ticker: str
    name: str
    sector: str
    industry: str
    price0: float
    shares: float
    beta: float
    idio_sd: float
    turnover: float          # share of float traded per day
    value_f: float
    quality_f: float
    catalyst_f: float
    short_f: float
    hidden: float
    prices: List[float] = field(default_factory=list)  # price history incl. current
    pays_dividend: bool = False
    quality_history: List[float] = field(default_factory=list)  # last 13 months incl. current


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _month_end(start: datetime, offset: int) -> datetime:
    idx = start.month - 1 + offset
    year, month = start.year + idx // 12, idx % 12 + 1
    return datetime(year, month, calendar.monthrange(year, month)[1])


def _make_tickers(rng: random.Random, n: int) -> List[str]:
    letters = "ABCDEFGHIJKLMNOPRSTVWXYZ"
    seen: set = set()
    out: List[str] = []
    while len(out) < n:
        t = "".join(rng.choice(letters) for _ in range(rng.choice((3, 4))))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _init_stock(rng: random.Random, ticker: str, noise_scale: float) -> _Stock:
    sector = rng.choice(sorted(_SECTORS))
    price0 = rng.uniform(8, 400)
    vol_base = rng.gauss(0, 1)
    return _Stock(
        ticker=ticker,
        name=f"{rng.choice(_NAME_PREFIX)} {rng.choice(_NAME_SUFFIX)} Inc.",
        sector=sector,
        industry=rng.choice(_SECTORS[sector]),
        price0=price0,
        shares=10 ** _clip(rng.gauss(10.2, 0.4), 9.0, 11.5) / price0,
        beta=_clip(1 + 0.25 * vol_base + 0.15 * rng.gauss(0, 1), 0.4, 1.9),
        idio_sd=0.055 * math.exp(0.25 * vol_base) * noise_scale,
        turnover=0.012 * math.exp(0.4 * rng.gauss(0, 1)),
        value_f=rng.gauss(0, 1), quality_f=rng.gauss(0, 1), catalyst_f=rng.gauss(0, 1),
        short_f=rng.gauss(0, 1), hidden=rng.gauss(0, 1),
        prices=[price0],
    )


def _advance_state(rng: random.Random, s: _Stock) -> None:
    g = rng.gauss
    s.value_f = 0.85 * s.value_f + 0.5 * g(0, 1)
    s.quality_f = 0.9 * s.quality_f + math.sqrt(1 - 0.81) * g(0, 1)
    s.catalyst_f = 0.5 * s.catalyst_f + math.sqrt(1 - 0.25) * g(0, 1)
    s.short_f = 0.9 * s.short_f + math.sqrt(1 - 0.81) * g(0, 1)
    s.hidden = 0.7 * s.hidden + math.sqrt(1 - 0.49) * g(0, 1)


def _price_features(s: _Stock) -> Dict[str, float]:
    """Everything derivable from the price path up to and including today."""
    p = s.prices
    window = p[-13:]
    rets = [b / a - 1 for a, b in zip(window[:-1], window[1:])]
    mean = sum(rets) / len(rets) if rets else 0.0
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)) if rets else 0.1
    return {
        "momentum": p[-1] / p[max(0, len(p) - 7)] - 1,
        "distance_from_high": 1 - p[-1] / max(window),
        "volatility": sd * math.sqrt(12),
        "last_return": p[-1] / p[-2] - 1 if len(p) > 1 else 0.0,
        "momentum_12_1": p[-2] / p[max(0, len(p) - 13)] - 1 if len(p) > 1 else 0.0,
    }


def _estimate_beta(prices: List[float], market_history: List[float]) -> float:
    """OLS beta of the trailing monthly returns on the market, using data up to T only."""
    n = min(len(prices) - 1, len(market_history))
    if n < 3:
        return 1.0
    window = prices[-(n + 1):]
    y = [b / a - 1 for a, b in zip(window[:-1], window[1:])]
    x = market_history[-n:]
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return 1.0
    return _clip(sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx, -1.0, 4.0)


def _observe(rng: random.Random, s: _Stock, pf: Dict[str, float]) -> Dict[str, float]:
    """Noisy observation of the hidden state at T."""
    g = rng.gauss
    price = s.prices[-1]
    market_cap = price * s.shares
    cap0 = s.price0 * s.shares
    pe = _clip(25 * math.exp(-0.35 * s.value_f + 0.25 * g(0, 1)), 6, 80)
    if s.quality_f < -1.3 and rng.random() < 0.5:
        pe = -1.0  # negative earnings
    fcf = cap0 * (0.035 + 0.02 * (0.5 * s.quality_f + 0.5 * s.value_f) + 0.015 * g(0, 1))
    short_interest = _clip(0.06 * math.exp(0.5 * s.short_f + 0.2 * g(0, 1)), 0.005, 0.5)
    return {
        "company_name": s.name, "sector": s.sector, "industry": s.industry,
        "price": round(price, 2), "shares_outstanding": s.shares, "market_cap": market_cap,
        "revenue_growth": _clip(0.10 + 0.06 * s.quality_f + 0.07 * g(0, 1), -0.3, 0.8),
        "earnings_growth": _clip(0.10 + 0.08 * s.quality_f + 0.10 * g(0, 1), -0.6, 1.0),
        "free_cash_flow": fcf,
        "debt_to_equity": _clip(0.8 - 0.3 * s.quality_f + 0.3 * g(0, 1), 0.0, 4.0),
        "pe_ratio": pe,
        "price_to_sales": _clip(3.0 * math.exp(-0.4 * s.value_f + 0.3 * g(0, 1)), 0.3, 25.0),
        "momentum": pf["momentum"],
        "volume_ratio": _clip(math.exp(0.15 * abs(pf["last_return"]) / 0.05
                                       + 0.1 * s.catalyst_f + 0.3 * g(0, 1)), 0.4, 4.5),
        "volatility": _clip(pf["volatility"] * math.exp(0.05 * g(0, 1)), 0.10, 1.5),
        "catalyst_score": _clip(50 + 15 * s.catalyst_f + 10 * g(0, 1), 0, 100),
        "sentiment_score": _clip(50 + 10 * (0.5 * _clip(pf["momentum"] / 0.2, -3, 3)
                                            + 0.5 * s.catalyst_f) + 12 * g(0, 1), 0, 100),
        "short_interest": short_interest,
        "days_to_cover": _clip(short_interest / (s.turnover * math.exp(0.2 * g(0, 1))), 0.1, 30.0),
        "distance_from_high": _clip(pf["distance_from_high"], 0.0, 0.9),
    }


def _observe_extended(rng: random.Random, s: _Stock, pf: Dict[str, float],
                      market_history: List[float], pe_ratio: float) -> Dict[str, float]:
    """Extended fundamentals at T: noisy proxies of the hidden state, on their own stream."""
    g = rng.gauss
    quality_delta = s.quality_f - s.quality_history[0]
    roe = _clip(0.12 + 0.07 * s.quality_f + 0.03 * g(0, 1), -0.5, 0.6)
    roic = _clip(0.10 + 0.06 * s.quality_f + 0.03 * g(0, 1), -0.5, 0.6)
    if pe_ratio <= 0:  # loss-making companies cannot show positive returns on capital
        roe, roic = -abs(roe), -abs(roic)
    dividend = (_clip(0.025 * math.exp(0.35 * s.value_f + 0.2 * g(0, 1)), 0.002, 0.12)
                if s.pays_dividend else 0.0)
    trend = 1.0
    for r in market_history[-6:]:
        trend *= 1 + r
    return {
        "price_to_book": _clip(1.8 * math.exp(-0.4 * s.value_f + 0.25 * g(0, 1)), 0.2, 25.0),
        "roe": roe,
        "roic": roic,
        "gross_margin": _clip(0.40 + 0.09 * s.quality_f + 0.04 * g(0, 1), 0.05, 0.9),
        "dividend_yield": dividend,
        "earnings_surprise": _clip(0.02 + 0.04 * s.catalyst_f + 0.06 * g(0, 1), -0.6, 0.6),
        "momentum_12_1": pf["momentum_12_1"],
        "return_1m": pf["last_return"],
        "beta": _estimate_beta(s.prices, market_history),
        "market_trend": trend - 1,
        "roe_change": 0.07 * quality_delta + 0.02 * g(0, 1),
        "gross_margin_change": 0.09 * quality_delta + 0.015 * g(0, 1),
        "debt_change": -0.3 * quality_delta + 0.1 * g(0, 1),
    }


def _drivers(s: _Stock, pf: Dict[str, float]) -> Dict[str, float]:
    """Standardised state at T that the outcome loads on (all known at T)."""
    return {
        "value": s.value_f,
        "quality": s.quality_f,
        "catalyst": s.catalyst_f,
        "momentum": _clip(pf["momentum"] / 0.2, -3, 3),
        "breakout": _clip((0.15 - pf["distance_from_high"]) / 0.08, -3, 3),
        "squeeze": s.short_f,
        # high only when BOTH a deep drawdown and strong quality are present
        "contrarian": _clip(min((pf["distance_from_high"] - 0.15) / 0.10, s.quality_f), -3, 3),
        "hidden": s.hidden,
        "reversal": _clip(-pf["last_return"] / 0.08, -3, 3),
        "low_vol": _clip((0.35 - pf["volatility"]) / 0.15, -3, 3),
        "size": _clip((10.2 - math.log10(max(s.prices[-1] * s.shares, 1.0))) / 0.4, -3, 3),
    }


def _percentile_ranks(values: List[float]) -> List[float]:
    n = len(values)
    if n == 1:
        return [50.0]
    order = sorted(range(n), key=lambda i: (values[i], i))
    ranks = [0.0] * n
    for position, idx in enumerate(order):
        ranks[idx] = position / (n - 1) * 100
    return ranks


def generate_dataset(
    n_stocks: int = 120,
    n_periods: int = 24,
    seed: int = 42,
    start: datetime = datetime(2023, 1, 31),
    signal: Optional[SignalConfig] = None,
) -> MarketDataset:
    """Monthly synthetic dataset. Deterministic for a given argument set."""
    if n_stocks < 1 or n_periods < 1:
        raise ValueError("n_stocks and n_periods must be >= 1")
    cfg = signal or SignalConfig()
    strengths = {k: getattr(cfg, k) for k in SIGNAL_KEYS}
    rng = random.Random(seed)
    extended_rng = random.Random(seed * 7919 + 101)  # draws here never shift the main stream
    stocks = [_init_stock(rng, t, cfg.noise_scale) for t in _make_tickers(rng, n_stocks)]
    for s in stocks:
        s.pays_dividend = extended_rng.random() < 0.6
    # Random per-regime multiplier of every loading: which factor "works" depends on the seed.
    multipliers = {r: {k: _clip(1 + cfg.regime_dispersion * rng.gauss(0, 1), -1.0, 3.0)
                       for k in _BASE_SIGNAL_KEYS} for r in REGIMES}
    for r in REGIMES:
        multipliers[r].update({k: _clip(1 + cfg.regime_dispersion * extended_rng.gauss(0, 1),
                                        -1.0, 3.0) for k in _EXTRA_SIGNAL_KEYS})
    market_history: List[float] = []  # market returns up to T, never later

    snapshots: List[StockSnapshot] = []
    outcomes: Dict[Tuple[str, datetime], float] = {}
    regimes: Dict[datetime, str] = {}
    regime = rng.choices(REGIMES, weights=_REGIME_START_WEIGHTS)[0]

    for period in range(-WARMUP_PERIODS, n_periods):
        emit = period >= 0
        ts = _month_end(start, period) if emit else None
        if rng.random() >= _REGIME_STAY_PROB:
            regime = rng.choices(REGIMES, weights=_REGIME_START_WEIGHTS)[0]

        # (1)+(2): state and observable features at T. No outcome information used.
        observed, drivers = [], []
        for s in stocks:
            _advance_state(rng, s)
            s.quality_history = (s.quality_history + [s.quality_f])[-13:]
            pf = _price_features(s)
            drivers.append(_drivers(s, pf))
            feats = _observe(rng, s, pf) if emit else {}
            if emit:
                feats.update(_observe_extended(extended_rng, s, pf, market_history,
                                               feats["pe_ratio"]))
            observed.append(feats)
        if emit:
            ranks = _percentile_ranks([o["momentum"] for o in observed])
            for s, feats, rank in zip(stocks, observed, ranks):
                snapshots.append(StockSnapshot(timestamp=ts, ticker=s.ticker,
                                               relative_strength=rank, **feats))
            regimes[ts] = regime

        # (3): outcome after T, drawn only now.
        mean, sd = _REGIME_MARKET[regime]
        market = rng.gauss(mean, sd)
        market_history = (market_history + [market])[-12:]
        sector_shock = {sec: 0.025 * rng.gauss(0, 1) for sec in sorted(_SECTORS)}
        mult = multipliers[regime]
        for s, drv in zip(stocks, drivers):
            signal_part = sum(strengths[k] * mult[k] * drv[k] for k in SIGNAL_KEYS)
            tail = 2.5 if rng.random() < 0.04 else 1.0
            eps = s.idio_sd * tail * rng.gauss(0, 1)
            ret = max(-0.9, s.beta * market + sector_shock[s.sector] + signal_part + eps)
            if emit:
                outcomes[(s.ticker, ts)] = ret
            s.prices.append(s.prices[-1] * (1 + ret))
            s.prices = s.prices[-13:]
            s.value_f -= 2.0 * ret  # cheapness decays when the price rises

    return MarketDataset(snapshots, outcomes, regimes)
