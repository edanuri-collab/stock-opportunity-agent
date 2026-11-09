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
    assert any("נתונים סינתטיים" in str(h.proto) for h in at.get("html")), page


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


def test_agent_builder_previews_and_saves(small_env, tmp_path, monkeypatch):
    import dashboard.views.catalog as catalog

    monkeypatch.setattr(catalog, "CUSTOM_DIR", tmp_path / "custom")
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio(key="page").set_value("Agent Catalog").run()
    assert not at.exception
    at.text_input(key="b_0_name").set_value("Smoke Agent").run()
    at.text_area(key="b_0_desc").set_value("test strategy").run()
    at.slider(key="b_0_w_value").set_value(60).run()
    assert not at.exception
    assert not [w for w in at.warning if "לא תקין" in w.value]
    at.button(key="b_0_save").click().run()
    assert not at.exception
    assert (tmp_path / "custom" / "smoke-agent.toml").exists()


def test_fair_ranking_runs_from_the_ui(small_env):
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.radio(key="page").set_value("Fair Ranking").run()
    at.slider(key="mc_seeds").set_value(5).run()
    at.button(key="mc_run").click().run()
    assert not at.exception
    assert any("יתרון עמיד" in str(h.proto) for h in at.get("html"))
    assert len(at.dataframe) >= 1
