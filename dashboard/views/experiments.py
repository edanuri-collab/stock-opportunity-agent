from __future__ import annotations

import streamlit as st

from dashboard import analytics as an
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import page_header
from engine.experiment import load_experiment_history


def render(ctx: DashboardContext) -> None:
    exp = ctx.exp
    page_header(ctx, "ניסויים",
                "כל הרצה נשמרת עם ה-Seed, גרסת הדאטה, גרסת הקוד וגרסאות הסוכנים. אותם קלטים "
                "תמיד מפיקים את אותו מזהה ניסוי ואת אותן תוצאות.")

    records = load_experiment_history(ctx.experiment_dir)
    st.markdown(f"#### היסטוריה ({len(records)} ניסויים שמורים בתיקייה `{ctx.experiment_dir}/`)")
    if records:
        hist = an.history_frame(records)
        hist["Rank By"] = hist["Rank By"].map(lambda k: he.RANK_KEYS.get(k, k))
        st.dataframe(he.frame(hist, ["Signal"]), hide_index=True, width="stretch",
                     column_config={he.col("Best Alpha"):
                                    st.column_config.NumberColumn(format="percent")})
    else:
        st.info("עדיין אין ניסויים שמורים. כל הרצה נשמרת אוטומטית בסיומה.")

    ids = [r["experiment_id"] for r in records] or [exp.experiment_id]
    if st.session_state.get("exp_pick_for") != exp.experiment_id or \
            st.session_state.get("exp_pick") not in ids:
        st.session_state["exp_pick"] = exp.experiment_id if exp.experiment_id in ids else ids[0]
        st.session_state["exp_pick_for"] = exp.experiment_id  # a new run selects itself
    chosen = st.selectbox("בחר ניסוי לבדיקה", ids, key="exp_pick")
    record = next((r for r in records if r["experiment_id"] == chosen), exp.to_dict())

    p = record["parameters"]
    c = st.columns(5)
    c[0].html(theme.card("Seed", str(p.get("seed"))))
    c[1].html(theme.card("גרסת דאטה", record["dataset_version"]))
    c[2].html(theme.card("גרסת קוד", record["code_version"]))
    c[3].html(theme.card("יקום מניות", f"{p.get('n_stocks')} x {p.get('n_periods')}",
                         "מניות x חודשים"))
    c[4].html(theme.card("Top N", str(p.get("top_n")), "בחירות לכל סוכן"))

    st.markdown("#### תוצאות")
    res = an.record_results_frame(record)
    st.dataframe(he.frame(res), hide_index=True, width="stretch", column_config={
        **{he.col(k): st.column_config.NumberColumn(format="percent")
           for k in ("Total Return", "Alpha", "Win Rate", "Max Drawdown")},
        he.col("t-stat"): st.column_config.NumberColumn(format="%.2f"),
        he.col("Sharpe"): st.column_config.NumberColumn(format="%.2f")})

    right, left = st.columns(2)
    with right:
        st.markdown("#### פרמטרים")
        st.json(p, expanded=1)
    with left:
        st.markdown("#### גרסאות הסוכנים")
        st.json(record.get("agent_versions", {}), expanded=True)
    st.caption(f"הורץ ב-{record['timestamp']}. גרסת סוכן = hash של המשקלים, קנס הסיכון וקוד "
               "הדירוג המשותף.")
    if chosen == exp.experiment_id:
        st.download_button("הורדת הניסוי כ-JSON", exp.to_json(),
                           file_name=f"{exp.experiment_id}.json", mime="application/json")
