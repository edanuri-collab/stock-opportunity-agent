from __future__ import annotations

import altair as alt
import streamlit as st

from dashboard import analytics as an
from dashboard import components as ui
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import curve_labels, dash_scale, fmt_date, page_header
from engine.benchmarks import MARKET


def render(ctx: DashboardContext) -> None:
    exp, status = ctx.exp, ctx.status
    s = an.overview_stats(exp)
    page_header(ctx, "סקירה כללית", "מעבדת מחקר כמותית: אותם נתונים ואותה נקודת זמן לכל "
                                    "הסוכנים, בהשוואה לשוק ולמזל.")

    c = st.columns(6)
    c[0].html(theme.card("מניות שנסרקו", f"{s['stocks_scanned']:,}",
                         f"{s['snapshots_analyzed']:,} תצפיות ב-{status.periods} חודשים"))
    c[1].html(theme.card("הזדמנויות שנמצאו", f"{s['opportunities']}",
                         f"{s['high_conviction']} נבחרו ע\"י 3 סוכנים או יותר"))
    c[2].html(theme.card("סוכנים פעילים", f"{s['active_agents']}",
                         f"{s['strategy_agents']} אסטרטגיות + סוכן ביקורת אקראי"))
    c[3].html(theme.card("אלפא מובילה (מול השוק)", f"{s['best_alpha']:+.1%}",
                         f"{s['best_agent']}, t = {s['best_t']:.2f}", theme.tone(s["best_alpha"])))
    c[4].html(theme.card("עדכון אחרון", fmt_date(status.as_of),
                         f"הורץ {s['run_timestamp'][:16].replace('T', ' ')} UTC"))
    c[5].html(theme.card("מקור נתונים", "סינתטי",
                         f"אין חיבור לנתונים חיים · {status.dataset_version}", theme.WARNING))

    st.markdown("#### ביצועי המערכת")
    p = st.columns(5)
    n = s["strategy_agents"]
    p[0].html(theme.card("אלפא חיובית", f"{s['beat_market']} / {n}", "אסטרטגיות מול השוק"))
    p[1].html(theme.card("אלפא מובהקת", f"{s['significant']} / {n}",
                         f"אחרי תיקון לריבוי בדיקות ({s['significant_raw']} לפני תיקון)",
                         theme.POSITIVE if s["significant"] else theme.NEUTRAL))
    p[2].html(theme.card("עקפו את הסוכן האקראי", f"{s['beat_random_agent']} / {n}",
                         "תשואה כוללת מול פלסבו"))
    p[3].html(theme.card("שארפ ממוצע", f"{s['avg_sharpe']:.2f}", "אסטרטגיות, בחישוב שנתי",
                         theme.tone(s["avg_sharpe"])))
    p[4].html(theme.card("תשואת השוק", f"{s['market_return']:+.1%}",
                         f"פרוקסי סינתטי משוקלל שווי, {status.periods} חודשים",
                         theme.tone(s["market_return"])))
    if s["significant"] == 0:
        st.caption("אחרי תיקון לריבוי בדיקות, אף אסטרטגיה לא מובהקת בריצה הזו. כדי לדעת מי "
                   "באמת טובה צריך הרבה עולמות: מסך 'דירוג הוגן'.")

    right, left = st.columns([1.35, 1])
    with right:
        st.markdown("#### עקומת הון")
        eq = ui.equity_frame(exp)
        leaders = [m.agent for m in exp.metrics if not an.is_control(m.agent)][:3]
        keep = leaders + ["Random Agent", MARKET]
        long = curve_labels(an.long_curves(eq[keep], "Equity"))
        line = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Equity:Q", title="הון יחסי", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", title=None, sort=he.values(keep)),
            strokeDash=alt.StrokeDash("Kind:N", legend=None, scale=dash_scale()),
            tooltip=[alt.Tooltip("Series", title="סדרה"), alt.Tooltip("Date:T", title="תאריך"),
                     alt.Tooltip("Equity:Q", title="הון", format=".3f")])
        st.altair_chart(theme.chart(line, 300), width="stretch")
        st.caption("3 האסטרטגיות המובילות, הסוכן האקראי (מקווקו) ופרוקסי השוק (מנוקד).")
    with left:
        st.markdown(f"#### ההזדמנויות המובילות, {fmt_date(ctx.latest)}")
        st.caption("ציון = אחוזון ממוצע של המניה אצל כל סוכני האסטרטגיה (ציונים מנורמלים).")
        opp = an.opportunities_frame(exp, ctx.latest, ctx.scores).head(10)
        view = he.frame(opp[["Ticker", "Company", "Opportunity Score", "Top Agent",
                             "Picked By"]])
        st.dataframe(view, hide_index=True, width="stretch", height=388,
                     column_config={he.col("Opportunity Score"):
                                    st.column_config.ProgressColumn(format="%.1f", min_value=0,
                                                                    max_value=100)})
