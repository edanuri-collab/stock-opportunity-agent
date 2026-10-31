from __future__ import annotations

import streamlit as st

from dashboard import analytics as an
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import page_header
from engine.experiment import load_experiment_history


def render(ctx: DashboardContext) -> None:
    exp = ctx.exp
    page_header(ctx, "Experiments",
                "Every run is saved with its seed, dataset, code and agent versions. "
                "The same inputs always produce the same experiment id and results.")

    records = load_experiment_history(ctx.experiment_dir)
    st.markdown(f"#### History ({len(records)} saved in `{ctx.experiment_dir}/`)")
    if records:
        hist = an.history_frame(records)
        st.dataframe(hist, hide_index=True, width="stretch", column_config={
            "Best Alpha": st.column_config.NumberColumn(format="percent")})
    else:
        st.info("No saved experiments yet. Runs are saved automatically when they finish.")

    ids = [r["experiment_id"] for r in records] or [exp.experiment_id]
    if st.session_state.get("exp_pick_for") != exp.experiment_id or \
            st.session_state.get("exp_pick") not in ids:
        st.session_state["exp_pick"] = exp.experiment_id if exp.experiment_id in ids else ids[0]
        st.session_state["exp_pick_for"] = exp.experiment_id  # a new run selects itself
    chosen = st.selectbox("Inspect experiment", ids, key="exp_pick")
    record = next((r for r in records if r["experiment_id"] == chosen), exp.to_dict())

    p = record["parameters"]
    c = st.columns(5)
    c[0].html(theme.card("Seed", str(p.get("seed"))))
    c[1].html(theme.card("Dataset Version", record["dataset_version"]))
    c[2].html(theme.card("Code Version", record["code_version"]))
    c[3].html(theme.card("Universe", f"{p.get('n_stocks')} x {p.get('n_periods')}m",
                         "stocks x months"))
    c[4].html(theme.card("Top N", str(p.get("top_n")), "picks per agent"))

    st.markdown("#### Results")
    st.dataframe(an.record_results_frame(record), hide_index=True, width="stretch",
                 column_config={
                     **{k: st.column_config.NumberColumn(format="percent")
                        for k in ("Total Return", "Alpha", "Win Rate", "Max Drawdown")},
                     "t-stat": st.column_config.NumberColumn(format="%.2f"),
                     "Sharpe": st.column_config.NumberColumn(format="%.2f")})

    left, right = st.columns(2)
    with left:
        st.markdown("#### Parameters")
        st.json(p, expanded=1)
    with right:
        st.markdown("#### Agent versions")
        st.json(record.get("agent_versions", {}), expanded=True)
    st.caption(f"Run at {record['timestamp']}. Agent version = hash of the agent's weights, "
               "risk penalty and the shared scoring code.")
    if chosen == exp.experiment_id:
        st.download_button("Download experiment JSON", exp.to_json(),
                           file_name=f"{exp.experiment_id}.json", mime="application/json")
