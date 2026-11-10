"""Tracking journal: locked entries, point-in-time decisions, late resolution, any provider."""
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from agents import default_agents
from data.models import MarketDataset
from data.providers import SyntheticProvider
from data.sample_data import generate_dataset
from engine.journal import Journal, is_forward

AGENTS = default_agents()


@pytest.fixture
def provider():
    return SyntheticProvider(generate_dataset(40, 6, seed=21))


@pytest.fixture
def journal(tmp_path):
    return Journal(tmp_path / "j.db")


def test_record_stores_each_agents_top_n_at_that_date(journal, provider):
    ts = provider.timestamps()[2]
    r = journal.record(provider, AGENTS, ts, top_n=5)
    assert r.added == len(AGENTS) and r.already_recorded == 0
    entries = journal.entries(provider.name, provider.version)
    assert {e["agent"] for e in entries} == {a.name for a in AGENTS}
    prices = {s.ticker: s.price for s in provider.snapshots_at(ts)}
    for e in entries:
        assert e["as_of"] == ts.isoformat() and len(e["picks"]) == 5
        assert all(p["price"] == prices[p["ticker"]] for p in e["picks"])
        assert e["period_return"] is None  # nothing is known about the future yet


def test_recording_twice_adds_nothing(journal, provider):
    ts = provider.timestamps()[0]
    journal.record(provider, AGENTS, ts, 5)
    again = journal.record(provider, AGENTS, ts, 5)
    assert again.added == 0 and again.already_recorded == len(AGENTS)
    assert len(journal.entries()) == len(AGENTS)


def test_entries_cannot_be_changed_or_deleted(journal, provider):
    journal.replay(provider, AGENTS, 5)
    db = sqlite3.connect(journal.path)
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        db.execute("UPDATE entries SET picks='[]'")
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        db.execute("DELETE FROM entries")
    with pytest.raises(sqlite3.DatabaseError, match="only once"):
        db.execute("UPDATE entries SET period_return=0.5 WHERE period_return IS NOT NULL")
    db.close()


def test_decisions_do_not_depend_on_future_data(tmp_path, provider):
    ds = generate_dataset(40, 6, seed=21)
    cutoff = ds.timestamps[2]
    snaps = [s if s.timestamp <= cutoff else replace(s, price=s.price * 7, momentum=4.0)
             for ts in ds.timestamps for s in ds.snapshots_at(ts)]
    tampered = MarketDataset(snaps, {k: 0.0 for k in ds._outcomes}, {t: "bull" for t in ds.timestamps})
    a, b = Journal(tmp_path / "a.db"), Journal(tmp_path / "b.db")
    a.record(SyntheticProvider(ds), AGENTS, cutoff, 5)
    b.record(SyntheticProvider(tampered), AGENTS, cutoff, 5)
    picks = lambda j: [(e["agent"], e["picks"]) for e in j.entries()]
    assert picks(a) == picks(b)


def test_resolution_uses_the_next_dates_prices(journal, provider):
    dates = provider.timestamps()
    journal.record(provider, AGENTS, dates[1], 5)
    journal.record(provider, AGENTS, dates[-1], 5)
    assert journal.resolve(provider) == len(AGENTS)  # the last date has no "next" yet
    end = {s.ticker: s.price for s in provider.snapshots_at(dates[2])}
    for e in journal.entries():
        if e["as_of"] == dates[-1].isoformat():
            assert e["period_return"] is None
            continue
        expected = sum(end[p["ticker"]] / p["price"] - 1 for p in e["picks"]) / len(e["picks"])
        assert e["period_return"] == pytest.approx(expected)
        assert e["next_as_of"] == dates[2].isoformat()
    assert journal.resolve(provider) == 0  # resolved only once


def test_forward_versus_replay_flag(journal, provider):
    ts = provider.timestamps()[0]
    assert is_forward(ts, ts.replace(tzinfo=timezone.utc) + timedelta(days=2))
    assert not is_forward(ts, ts.replace(tzinfo=timezone.utc) + timedelta(days=30))
    journal.record(provider, AGENTS, ts, 5, recorded_at=ts.replace(tzinfo=timezone.utc))
    assert all(e["forward"] == 1 for e in journal.entries())
    journal.record(provider, AGENTS, provider.timestamps()[1], 5)  # recorded "now": replay
    assert {e["forward"] for e in journal.entries()} == {0, 1}


def test_performance_summary(journal, provider):
    journal.replay(provider, AGENTS, 5)
    rows = journal.performance(provider.name, provider.version, ["Random Agent"])
    assert {r["agent"] for r in rows} == {a.name for a in AGENTS}
    excess = [r["mean_excess"] for r in rows]
    assert excess == sorted(excess, reverse=True)
    for r in rows:
        assert r["periods"] == len(provider.timestamps()) - 1
        assert 0 <= r["hit_rate"] <= 1
        if r["agent"] == "Random Agent":
            assert r["p_adjusted"] is None
        else:
            assert r["p_adjusted"] >= r["p_value"] - 1e-12


def test_any_provider_works(journal):
    """The journal needs only name/version/timestamps()/snapshots_at(): a real feed fits."""
    ds = generate_dataset(30, 3, seed=2)

    class TinyFeed:
        name, version = "vendor-x", "live"

        def timestamps(self):
            return ds.timestamps

        def snapshots_at(self, ts):
            return ds.snapshots_at(ts)

    r = journal.replay(TinyFeed(), AGENTS[:2], 3)
    assert r.added == 2 * 3
    assert all(e["source"] == "vendor-x" for e in journal.entries())
    with pytest.raises(ValueError):
        journal.record(TinyFeed(), AGENTS[:2], datetime(1999, 1, 1), 3)
