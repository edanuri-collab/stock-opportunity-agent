"""One module per screen. Each exposes ``render(ctx: DashboardContext) -> None``."""
from __future__ import annotations

from datetime import datetime
from typing import List

import streamlit as st

from dashboard import theme
from dashboard.context import DashboardContext


def page_header(ctx: DashboardContext, title: str, subtitle: str) -> None:
    exp = ctx.exp
    meta = (f"{exp.experiment_id}<br>dataset {exp.dataset_version} &middot; "
            f"code {exp.code_version}")
    st.html(theme.header(title, subtitle, meta))
    st.html(theme.synthetic_banner(ctx.status.dataset_version, ctx.status.signal_mode))


def fmt_date(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d")


def timestamp_select(label: str, options: List[datetime], key: str, container=st) -> datetime:
    """As-of selector that survives option changes (e.g. a new experiment)."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[-1]
    return container.selectbox(label, options, key=key, format_func=fmt_date)
