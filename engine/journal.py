"""Tracking journal: what every agent chose, when, locked, and how it turned out later.

* ``record``: stores each agent's top-N picks at date T, using only the snapshots of T.
  An entry can never be changed or deleted (enforced by database triggers).
* ``resolve``: once a later date exists, measures each entry's equal-weight price return
  from T to that date, plus the equal-weight universe return as the benchmark.
* Entries recorded close to T are *forward* (genuine out-of-sample tracking); entries
  recorded long after T are *replay* (reconstructed history). With synthetic data everything
  is replay; real forward tracking starts once a real provider is connected.

Returns are price-based and before trading costs and dividends. No orders are placed.

Command line (e.g. for a monthly scheduled task)::

    python -m engine.journal replay   # record every date of the current dataset + resolve
    python -m engine.journal record   # record the latest date only
    python -m engine.journal resolve
    python -m engine.journal report
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agents.base_agent import BaseAgent
from engine.backtest import decide
from engine.stats import holm, t_pvalue
from engine.versioning import agent_version

FORWARD_GRACE = timedelta(days=3)  # recorded within 3 days of T counts as forward tracking

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL, dataset_version TEXT NOT NULL, as_of TEXT NOT NULL,
    agent TEXT NOT NULL, agent_version TEXT NOT NULL, top_n INTEGER NOT NULL,
    picks TEXT NOT NULL, eligible INTEGER NOT NULL, universe INTEGER NOT NULL,
    recorded_at TEXT NOT NULL, forward INTEGER NOT NULL,
    resolved_at TEXT, next_as_of TEXT, period_return REAL, priced INTEGER,
    UNIQUE (source, dataset_version, as_of, agent, agent_version)
);
CREATE TABLE IF NOT EXISTS periods (
    source TEXT NOT NULL, dataset_version TEXT NOT NULL, as_of TEXT NOT NULL,
    next_as_of TEXT NOT NULL, universe_return REAL NOT NULL,
    PRIMARY KEY (source, dataset_version, as_of)
);
CREATE TRIGGER IF NOT EXISTS entries_locked
BEFORE UPDATE OF source, dataset_version, as_of, agent, agent_version, top_n, picks,
                 eligible, universe, recorded_at, forward ON entries
BEGIN SELECT RAISE(ABORT, 'journal entries are immutable'); END;
CREATE TRIGGER IF NOT EXISTS entries_resolved_once
BEFORE UPDATE OF period_return ON entries WHEN OLD.period_return IS NOT NULL
BEGIN SELECT RAISE(ABORT, 'journal entries are resolved only once'); END;
CREATE TRIGGER IF NOT EXISTS entries_no_delete
BEFORE DELETE ON entries
BEGIN SELECT RAISE(ABORT, 'journal entries cannot be deleted'); END;
"""


@dataclass
class RecordResult:
    added: int = 0
    already_recorded: int = 0
    dates: int = 0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def is_forward(as_of: datetime, recorded_at: datetime) -> bool:
    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    return recorded_at <= as_of_utc + timedelta(days=1) + FORWARD_GRACE


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    # ---- writing -----------------------------------------------------------------------

    def record(self, provider: Any, agents: Sequence[BaseAgent], as_of: datetime,
               top_n: int, recorded_at: Optional[datetime] = None) -> RecordResult:
        """Store every agent's picks at ``as_of``. Uses only ``provider.snapshots_at(as_of)``."""
        snapshots = provider.snapshots_at(as_of)
        if not snapshots:
            raise ValueError(f"provider has no data at {as_of}")
        stamp = recorded_at or _now()
        forward = int(is_forward(as_of, stamp))
        prices = {s.ticker: s.price for s in snapshots}
        result = RecordResult(dates=1)
        decisions = decide(agents, snapshots, top_n)
        versions = {a.name: agent_version(a) for a in agents}
        with closing(self._connect()) as db, db:
            for name, ranked in decisions.items():
                eligible = [r for r in ranked if r.get("eligible", True)]
                picks = [{"ticker": r["ticker"], "score": r["opportunity_score"],
                          "price": prices[r["ticker"]]} for r in eligible[:top_n]]
                cur = db.execute(
                    "INSERT OR IGNORE INTO entries (source, dataset_version, as_of, agent, "
                    "agent_version, top_n, picks, eligible, universe, recorded_at, forward) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (provider.name, provider.version, _iso(as_of), name, versions[name], top_n,
                     json.dumps(picks), len(eligible), len(ranked), _iso(stamp), forward))
                if cur.rowcount:
                    result.added += 1
                else:
                    result.already_recorded += 1
        return result

    def replay(self, provider: Any, agents: Sequence[BaseAgent], top_n: int,
               recorded_at: Optional[datetime] = None) -> RecordResult:
        """Record every date of the provider (reconstructed history), then resolve."""
        total = RecordResult()
        for ts in provider.timestamps():
            r = self.record(provider, agents, ts, top_n, recorded_at)
            total.added += r.added
            total.already_recorded += r.already_recorded
            total.dates += 1
        self.resolve(provider)
        return total

    def resolve(self, provider: Any, resolved_at: Optional[datetime] = None) -> int:
        """Measure open entries whose next date is now available. Returns entries resolved."""
        dates = sorted(provider.timestamps())
        nxt = {_iso(a): b for a, b in zip(dates[:-1], dates[1:])}
        stamp = _iso(resolved_at or _now())
        done = 0
        with closing(self._connect()) as db, db:
            rows = db.execute(
                "SELECT id, as_of, picks FROM entries WHERE source=? AND dataset_version=? "
                "AND period_return IS NULL", (provider.name, provider.version)).fetchall()
            later_prices: Dict[str, Dict[str, float]] = {}
            for row in rows:
                if row["as_of"] not in nxt:
                    continue  # no later data yet: stays open
                end = nxt[row["as_of"]]
                if row["as_of"] not in later_prices:
                    start_px = {s.ticker: s.price for s in provider.snapshots_at(
                        datetime.fromisoformat(row["as_of"]))}
                    end_px = {s.ticker: s.price for s in provider.snapshots_at(end)}
                    later_prices[row["as_of"]] = end_px
                    common = [t for t in start_px if t in end_px and start_px[t] > 0]
                    universe = (sum(end_px[t] / start_px[t] - 1 for t in common) / len(common)
                                if common else 0.0)
                    db.execute("INSERT OR IGNORE INTO periods VALUES (?,?,?,?,?)",
                               (provider.name, provider.version, row["as_of"], _iso(end),
                                universe))
                end_px = later_prices[row["as_of"]]
                picks = json.loads(row["picks"])
                rets = [end_px[p["ticker"]] / p["price"] - 1 for p in picks
                        if p["ticker"] in end_px and p["price"] > 0]
                db.execute("UPDATE entries SET period_return=?, resolved_at=?, next_as_of=?, "
                           "priced=? WHERE id=?",
                           (sum(rets) / len(rets) if rets else 0.0, stamp, _iso(end), len(rets),
                            row["id"]))
                done += 1
        return done

    # ---- reading -----------------------------------------------------------------------

    def entries(self, source: Optional[str] = None, version: Optional[str] = None
                ) -> List[Dict[str, Any]]:
        query, args = "SELECT * FROM entries", []
        if source is not None:
            query += " WHERE source=? AND dataset_version=?"
            args = [source, version]
        with closing(self._connect()) as db:
            rows = db.execute(query + " ORDER BY as_of, agent", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["picks"] = json.loads(d["picks"])
            out.append(d)
        return out

    def universe_returns(self, source: str, version: str) -> Dict[str, float]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT as_of, universe_return FROM periods WHERE source=? AND "
                              "dataset_version=?", (source, version)).fetchall()
        return {r["as_of"]: r["universe_return"] for r in rows}

    def performance(self, source: str, version: str,
                    controls: Sequence[str] = ()) -> List[Dict[str, Any]]:
        """Per agent, over resolved entries: compounded return, hit rate and the excess over
        the equal-weight universe with a t-test, Holm-corrected across non-control agents."""
        bench = self.universe_returns(source, version)
        by_agent: Dict[str, List[Dict[str, Any]]] = {}
        for e in self.entries(source, version):
            if e["period_return"] is not None and e["as_of"] in bench:
                by_agent.setdefault(e["agent"], []).append(e)
        rows = []
        for agent, items in by_agent.items():
            rets = [e["period_return"] for e in items]
            excess = [e["period_return"] - bench[e["as_of"]] for e in items]
            n = len(excess)
            mean = sum(excess) / n
            sd = (math.sqrt(sum((x - mean) ** 2 for x in excess) / (n - 1)) if n > 1 else 0.0)
            t = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
            equity = 1.0
            for r in rets:
                equity *= 1 + r
            rows.append({
                "agent": agent, "periods": n, "total_return": equity - 1,
                "mean_return": sum(rets) / n, "hit_rate": sum(1 for x in excess if x > 0) / n,
                "mean_excess": mean, "t_stat": t,
                "p_value": t_pvalue(t, n - 1) if sd > 0 and n > 1 else 1.0,
                "forward_share": sum(e["forward"] for e in items) / n,
            })
        adjusted = holm({r["agent"]: r["p_value"] for r in rows if r["agent"] not in controls})
        for r in rows:
            r["p_adjusted"] = adjusted.get(r["agent"])
        rows.sort(key=lambda r: (-r["mean_excess"], r["agent"]))
        return rows


# ---- command line -------------------------------------------------------------------------

def _cli(argv: Optional[Sequence[str]] = None) -> None:
    from agents.registry import load_registry
    from config import Settings
    from data.providers import SyntheticProvider
    from data.sample_data import generate_dataset

    parser = argparse.ArgumentParser(description="Agent tracking journal (no trading).")
    parser.add_argument("command", choices=("record", "replay", "resolve", "report"))
    args = parser.parse_args(argv)
    cfg = Settings.from_env()
    provider = SyntheticProvider(generate_dataset(cfg.n_stocks, cfg.n_periods, cfg.seed))
    agents = load_registry().agents
    journal = Journal(cfg.journal_path)
    if args.command == "record":
        r = journal.record(provider, agents, provider.timestamps()[-1], cfg.top_n)
        print(f"recorded {r.added} entries ({r.already_recorded} already in the journal)")
    elif args.command == "replay":
        r = journal.replay(provider, agents, cfg.top_n)
        print(f"replayed {r.dates} dates: {r.added} new entries, "
              f"{r.already_recorded} already recorded")
    elif args.command == "resolve":
        print(f"resolved {journal.resolve(provider)} entries")
    else:
        controls = [a.name for a in agents if a.family == "control"]
        for row in journal.performance(provider.name, provider.version, controls):
            print(f"{row['agent']:24s} periods={row['periods']:3d} "
                  f"excess={row['mean_excess']:+.2%}/month t={row['t_stat']:+.2f} "
                  f"p_adj={row['p_adjusted'] if row['p_adjusted'] is not None else '-'}")


if __name__ == "__main__":
    _cli()
