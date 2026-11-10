from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import analytics as an
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import page_header
from data.providers import SyntheticProvider
from engine.journal import Journal

MESSAGE_KEY = "journal_message"


def _set_message(text: str) -> None:
    st.session_state[MESSAGE_KEY] = text


def render(ctx: DashboardContext) -> None:
    page_header(ctx, "יומן מעקב",
                "רישום נעול של מה שכל סוכן בחר ומתי, ומדידת התוצאה רק אחרי שהזמן עבר.")
    st.info("היומן רושם את הבחירות של כל סוכן בכל תאריך, והרישום ננעל: אי אפשר לשנות או "
            "למחוק אותו. התוצאה נמדדת רק כשמגיע התאריך הבא, לפי שינוי המחיר של המניות שנבחרו "
            "(לפני עמלות ודיבידנדים, ובלי לבצע שום עסקה). רישום שנעשה בזמן אמת מסומן "
            "'קדימה', ורישום של עבר מסומן 'שחזור'. עם נתונים סינתטיים הכל שחזור: מעקב אמיתי "
            "יתחיל כשיחובר מקור נתונים אמיתי.", icon=":material/menu_book:")

    provider = SyntheticProvider(ctx.exp.dataset)
    journal = Journal(ctx.journal_path)
    top_n = int(ctx.exp.parameters["top_n"])
    agents = ctx.registry.agents

    b1, b2, b3, _ = st.columns([1.3, 1.3, 1.1, 2])
    if b1.button("רשום את התאריך האחרון", icon=":material/edit_note:", key="j_record"):
        r = journal.record(provider, agents, provider.timestamps()[-1], top_n)
        _set_message(f"נרשמו {r.added} החלטות חדשות ({r.already_recorded} כבר היו ביומן).")
    if b2.button("שחזר את כל ההיסטוריה", icon=":material/history:", key="j_replay",
                 type="primary"):
        with st.spinner("רושם את כל התאריכים..."):
            r = journal.replay(provider, agents, top_n)
        _set_message(f"שוחזרו {r.dates} תאריכים: {r.added} החלטות חדשות, "
                     f"{r.already_recorded} כבר היו ביומן. התוצאות עודכנו.")
    if b3.button("עדכן תוצאות", icon=":material/refresh:", key="j_resolve"):
        _set_message(f"עודכנו {journal.resolve(provider)} החלטות שהתאריך הבא שלהן כבר הגיע.")
    if st.session_state.get(MESSAGE_KEY):
        st.success(st.session_state.pop(MESSAGE_KEY), icon=":material/check_circle:")

    entries = journal.entries(provider.name, provider.version)
    resolved = [e for e in entries if e["period_return"] is not None]
    forward = sum(e["forward"] for e in entries)
    c = st.columns(4)
    c[0].html(theme.card("מקור נתונים", "סינתטי", f"גרסת דאטה {provider.version}",
                         theme.WARNING))
    c[1].html(theme.card("החלטות ביומן", f"{len(entries):,}",
                         f"{len({e['as_of'] for e in entries})} תאריכים, {len({e['agent'] for e in entries})} סוכנים"))
    c[2].html(theme.card("נמדדו", f"{len(resolved):,}",
                         f"{len(entries) - len(resolved)} ממתינות לתאריך הבא"))
    c[3].html(theme.card("נרשמו בזמן אמת", f"{forward:,}",
                         "השאר שחזור של העבר" if entries else "",
                         theme.POSITIVE if forward else theme.NEUTRAL))
    if not entries:
        st.caption("היומן ריק עבור הדאטה הנוכחי. לחץ על 'שחזר את כל ההיסטוריה' כדי להתחיל.")
        st.caption(f"קובץ היומן: `{ctx.journal_path}`")
        return

    controls = [a.name for a in agents if a.family == "control"]
    perf = journal.performance(provider.name, provider.version, controls)
    if perf:
        st.markdown("#### ביצועי הסוכנים לפי היומן")
        st.caption("עודף = תשואת המניות שהסוכן בחר פחות התשואה הממוצעת של כל המניות באותו "
                   "חודש. p מתוקן (Holm) לכל הסוכנים.")
        df = pd.DataFrame(perf)
        view = df.rename(columns={
            "agent": "סוכן", "periods": "חודשים", "total_return": "תשואה מצטברת",
            "mean_return": "תשואה חודשית ממוצעת", "hit_rate": "חודשים מעל הממוצע",
            "mean_excess": "עודף חודשי ממוצע", "t_stat": "t-stat",
            "p_adjusted": "p מתוקן (Holm)", "forward_share": "רישום בזמן אמת"}) \
            .drop(columns=["p_value"])
        st.dataframe(
            view.style.format({"תשואה מצטברת": "{:+.1%}", "תשואה חודשית ממוצעת": "{:+.2%}",
                               "חודשים מעל הממוצע": "{:.0%}", "עודף חודשי ממוצע": "{:+.2%}",
                               "t-stat": "{:.2f}", "p מתוקן (Holm)": "{:.3f}",
                               "רישום בזמן אמת": "{:.0%}"}, na_rep="-")
                      .map(lambda v: f"color: {theme.tone(v)}", subset=["עודף חודשי ממוצע"]),
            hide_index=True, width="stretch", height=38 + 35 * min(len(view), 12))

        st.markdown("#### עקומת הון מהיומן")
        top = [r["agent"] for r in perf if r["agent"] not in controls][:5] + controls
        bench = journal.universe_returns(provider.name, provider.version)
        rows = []
        for name in top + ["__universe__"]:
            equity = 1.0
            series = ([(e["as_of"], e["period_return"]) for e in resolved if e["agent"] == name]
                      if name != "__universe__" else sorted(bench.items()))
            for as_of, r in sorted(series):
                equity *= 1 + r
                rows.append({"Date": pd.to_datetime(as_of), "Equity": equity,
                             "Series": "ממוצע כל המניות" if name == "__universe__" else name})
        chart = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2).encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Equity:Q", title="הון יחסי", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title=None),
            tooltip=[alt.Tooltip("Series", title="סדרה"), alt.Tooltip("Date:T", title="תאריך"),
                     alt.Tooltip("Equity:Q", title="הון", format=".3f")])
        st.altair_chart(theme.chart(chart, 300), width="stretch")
        st.caption("5 הסוכנים המובילים ביומן, הסוכן האקראי והממוצע של כל המניות.")

    st.markdown("#### הרישומים האחרונים")
    agent_filter = st.selectbox("סוכן", ["כל הסוכנים"] + sorted({e["agent"] for e in entries}),
                                key="j_agent")
    shown = [e for e in entries if agent_filter == "כל הסוכנים" or e["agent"] == agent_filter]
    shown = sorted(shown, key=lambda e: (e["as_of"], e["agent"]), reverse=True)[:300]
    table = pd.DataFrame([{
        "תאריך": e["as_of"][:10], "סוכן": e["agent"],
        "אופן רישום": "קדימה" if e["forward"] else "שחזור",
        "נרשם ב-": e["recorded_at"][:16].replace("T", " "),
        "מניות שנבחרו": ", ".join(p["ticker"] for p in e["picks"]) or "(מזומן)",
        "תוצאה": (an.ltr(f"{e['period_return']:+.2%}") if e["period_return"] is not None
                  else "ממתין"),
        "גרסת סוכן": e["agent_version"],
    } for e in shown])
    st.dataframe(table, hide_index=True, width="stretch", height=420)
    export = pd.DataFrame([{**{k: v for k, v in e.items() if k != "picks"},
                            "picks": " ".join(p["ticker"] for p in e["picks"])}
                           for e in entries])
    st.download_button("ייצוא היומן ל-CSV", export.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"journal_{provider.version}.csv", mime="text/csv")
    st.caption(f"קובץ היומן: `{ctx.journal_path}`. אפשר גם להריץ מהטרמינל: "
               "`python -m engine.journal replay` / `record` / `resolve` / `report`.")
