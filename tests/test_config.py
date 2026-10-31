import pytest

from config import Settings


def test_defaults(monkeypatch):
    for k in ("SOA_SEED", "SOA_N_STOCKS", "SOA_N_PERIODS", "SOA_TOP_N"):
        monkeypatch.delenv(k, raising=False)
    assert Settings.from_env() == Settings()


def test_env_override_and_validation(monkeypatch):
    monkeypatch.setenv("SOA_SEED", "99")
    assert Settings.from_env().seed == 99
    monkeypatch.setenv("SOA_SEED", "abc")
    with pytest.raises(ValueError):
        Settings.from_env()
