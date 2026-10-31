from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard.context import PAGES  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def small_env(tmp_path, monkeypatch):
    for key, value in {"SOA_N_STOCKS": "40", "SOA_N_PERIODS": "8", "SOA_RANDOM_TRIALS": "20",
                       "SOA_EXPERIMENT_DIR": str(tmp_path)}.items():
        monkeypatch.setenv(key, value)
    return tmp_path


def test_app_runs_without_exceptions(small_env):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert list(small_env.glob("exp-*.json")), "experiment was not saved to history"


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(small_env, page):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, page
    assert any("SYNTHETIC DATA" in str(h.proto) for h in at.get("html")), page


def test_opportunity_filters_do_not_crash(small_env):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio(key="page").set_value("Stock Opportunities").run()
    at.text_input(key="opp_query").set_value("zzzz-no-match").run()
    assert not at.exception
    at.text_input(key="opp_query").set_value("").run()
    at.slider(key="opp_min").set_value(50).run()
    at.toggle(key="opp_picked").set_value(True).run()
    at.selectbox(key="opp_sort").set_value("Risk Score").run()
    assert not at.exception
