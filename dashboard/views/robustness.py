from __future__ import annotations

from dataclasses import replace

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import analytics as an
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import page_header
from engine.robustness import MonteCarloResult, run_monte_carlo

MC_TRIALS = 50          # random-portfolio trials per world (enough for the benchmark series)
RESULTS_KEY = "mc_results"
VERDICT_COLORS = {"Robust edge": theme.POSITIVE, "Robustly worse": theme.NEGATIVE,
                  "Inconclusive": theme.NEUTRAL}


def summary_frame(mc: MonteCarloResult, families: dict) -> pd.DataFrame:
    rows = [{"Agent": s.agent, "Family": families.get(s.agent, ""), "Verdict": s.verdict,
             "Mean Alpha": s.mean_alpha, "SD Alpha": s.sd_alpha, "t-stat": s.t_stat,
             "Adjusted p": s.p_adjusted, "Seeds Positive": s.share_positive,
             "Beats Random Agent": s.share_beats_random_agent, "Mean Rank": s.mean_rank,
             "Alpha In": s.mean_alpha_in, "Alpha Out": s.mean_alpha_out}
            for s in mc.summaries]
    return pd.DataFrame(rows)


def render(ctx: DashboardContext) -> None:
    page_header(ctx, "דירוג הוגן",
                "אותם סוכנים בעשרות עולמות מדומים שונים, עם תיקון לריבוי בדיקות, בדיקת "
                "יציבות לאורך זמן ובחירה 'קדימה בזמן'.")
    st.info("איך לקרוא את המסך: ריצה בודדת יכולה להמליך מנצח במזל. כאן כל סוכן נבדק בהרבה "
            "עולמות, ונחשב 'יתרון עמיד' רק אם האלפא הממוצעת שלו חיובית באופן מובהק גם אחרי "
            "תיקון Holm לכל הסוכנים. חשוב: העולמות סינתטיים, ולכל סוג אסטרטגיה בסימולטור יש "
            "קשר חלש וחיובי בממוצע לתשואה. לכן המסך בודק שהשיטה עובדת, ולא מוכיח מה יעבוד "
            "בשוק האמיתי. בחירה ב'ללא סיגנל' בהגדרות היא בדיקת ביקורת: שם לא אמור להופיע "
            "שום יתרון עמיד.", icon=":material/info:")

    c1, c2 = st.columns([2, 1])
    n_seeds = c1.slider("מספר עולמות (Seeds)", 5, 50, 20, key="mc_seeds",
                        help="כל עולם הוא דאטה סינתטי שונה. יותר עולמות = תשובה אמינה יותר, "
                             "אבל ריצה ארוכה יותר (כ-1.5 שניות לעולם).")
    params = replace(ctx.params, random_trials=MC_TRIALS) if ctx.params else None
    key = (params, n_seeds, ctx.agents_version)
    results = st.session_state.setdefault(RESULTS_KEY, {})
    if c2.button("הרץ ניתוח", type="primary", icon=":material/play_arrow:", key="mc_run",
                 disabled=params is None):
        bar = st.progress(0.0, text="מריץ עולמות...")
        results[key] = run_monte_carlo(
            params, ctx.registry.agents, list(range(1, n_seeds + 1)),
            progress=lambda done, total: bar.progress(done / total,
                                                      text=f"עולם {done} מתוך {total}"))
        bar.empty()
    mc = results.get(key)
    if mc is None:
        st.caption(f"עוד לא הורץ ניתוח עבור ההגדרות הנוכחיות ({len(ctx.registry.agents)} סוכנים, "
                   f"{n_seeds} עולמות). לחץ על 'הרץ ניתוח'.")
        return
    _render_results(ctx, mc)


def _render_results(ctx: DashboardContext, mc: MonteCarloResult) -> None:
    families = {a.name: a.family for a in ctx.registry.agents}
    df = summary_frame(mc, families)
    robust = df[df["Verdict"] == "Robust edge"]
    rank_corr = mc.mean("rank_correlation")
    wf, avg = mc.mean("walk_forward"), mc.mean("walk_forward_average")

    c = st.columns(4)
    c[0].html(theme.card("עולמות שנבדקו", str(len(mc.runs)),
                         f"{len(df)} סוכנים, {mc.params.get('cost_bps', 0):g} bps לצד"))
    c[1].html(theme.card("יתרון עמיד", f"{len(robust)} / {len(df) - 1}",
                         "אלפא חיובית מובהקת אחרי תיקון Holm",
                         theme.POSITIVE if len(robust) else theme.NEUTRAL))
    c[2].html(theme.card("יציבות הדירוג", f"{rank_corr:+.2f}",
                         "מתאם בין הדירוג בחצי הראשון לשני (0 = אין קשר)",
                         theme.tone(rank_corr) if abs(rank_corr) >= 0.1 else theme.NEUTRAL))
    c[3].html(theme.card("לרדוף אחרי המוביל", f"{wf - avg:+.1%}",
                         f"מול החזקת כל האסטרטגיות ({mc.share(lambda r: r.walk_forward > r.walk_forward_average):.0%} מהעולמות)",
                         theme.tone(wf - avg)))

    st.markdown("#### מה אפשר להסיק")
    st.markdown("\n".join(f"- {line}" for line in _conclusions(df, mc)))

    st.markdown("#### דירוג לפי אלפא ממוצעת בכל העולמות")
    view = he.frame(df.assign(Family=df["Family"].map(he.FAMILIES)), ["Verdict"])
    col = he.col
    colors = {he.v(k): v for k, v in VERDICT_COLORS.items()}
    styled = (view.style
              .format({col("Mean Alpha"): "{:+.1%}", col("SD Alpha"): "{:.1%}",
                       col("t-stat"): "{:.2f}", col("Adjusted p"): "{:.3f}",
                       col("Seeds Positive"): "{:.0%}", col("Beats Random Agent"): "{:.0%}",
                       col("Mean Rank"): "{:.1f}", col("Alpha In"): "{:+.1%}",
                       col("Alpha Out"): "{:+.1%}"})
              .map(lambda v: f"color: {colors.get(v, theme.TEXT)}; font-weight: 600",
                   subset=[col("Verdict")])
              .map(lambda v: f"color: {theme.tone(v)}", subset=[col("Mean Alpha")]))
    st.dataframe(styled, hide_index=True, width="stretch", height=38 + 35 * min(len(df), 20))

    st.markdown("#### פיזור האלפא בין העולמות")
    points = pd.DataFrame([{"Agent": a, "Alpha": r.alpha[a], "Seed": r.seed}
                           for r in mc.runs for a in df["Agent"]])
    order = list(df["Agent"])
    dots = alt.Chart(points).mark_circle(size=28, opacity=.45, color=theme.ACCENT).encode(
        y=alt.Y("Agent:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("Alpha:Q", axis=alt.Axis(format="%"), title="אלפא שנתית בכל עולם"),
        tooltip=[alt.Tooltip("Agent", title="סוכן"), alt.Tooltip("Seed", title="עולם"),
                 alt.Tooltip("Alpha:Q", title="אלפא", format="+.1%")])
    means = alt.Chart(df).mark_tick(thickness=3, size=16, color=theme.WARNING).encode(
        y=alt.Y("Agent:N", sort=order), x="Mean Alpha:Q",
        tooltip=[alt.Tooltip("Agent", title="סוכן"),
                 alt.Tooltip("Mean Alpha:Q", title="ממוצע", format="+.2%")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=theme.MUTED,
                                                         strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(theme.chart(dots + means + zero, 22 * len(order) + 40), width="stretch")
    st.caption("כל נקודה היא עולם אחד; הקו הכתום הוא הממוצע. פיזור רחב משני צדי האפס = "
               "תוצאה שתלויה במזל של העולם.")

    st.markdown("#### בחירה קדימה בזמן (Walk-Forward)")
    st.markdown(f"בכל חודש עוברים לסוכן שהיה לו התשואה הטובה ביותר ב-12 החודשים הקודמים, "
                f"ומקבלים את התשואה שלו **בחודש הבא**. בממוצע: "
                f"{an.ltr(f'{wf:+.1%}')} מול {an.ltr(f'{avg:+.1%}')} להחזקת כל האסטרטגיות "
                f"ו-{an.ltr(format(mc.mean('walk_forward_random_agent'), '+.1%'))} לסוכן האקראי. "
                f"המנצח של החצי הראשון הגיע בחצי השני בממוצע לאחוזון "
                f"{an.ltr(format(mc.mean('winner_out_percentile'), '.0%'))} מבין האסטרטגיות.")


def _conclusions(df: pd.DataFrame, mc: MonteCarloResult) -> list:
    robust = list(df.loc[df["Verdict"] == "Robust edge", "Agent"])
    worse = list(df.loc[df["Verdict"] == "Robustly worse", "Agent"])
    lines = []
    if robust:
        lines.append(f"{len(robust)} סוכנים מראים יתרון עמיד בעולמות האלה: "
                     + ", ".join(an.ltr(a) for a in robust[:8])
                     + (" ועוד." if len(robust) > 8 else "."))
    else:
        lines.append("אף סוכן לא מראה יתרון עמיד אחרי תיקון לריבוי בדיקות.")
    if worse:
        lines.append("חלשים באופן עקבי: " + ", ".join(an.ltr(a) for a in worse[:8]) + ".")
    rc = mc.mean("rank_correlation")
    if abs(rc) < 0.15:
        lines.append("הדירוג בתוך ריצה בודדת כמעט לא חוזר על עצמו בהמשך: מי שמוביל בחצי "
                     "הראשון לא בהכרח מוביל בשני. לכן אין להסיק מסקנות מריצה אחת.")
    else:
        lines.append(f"יש יציבות מסוימת בדירוג (מתאם {an.ltr(f'{rc:+.2f}')}).")
    if mc.mean("walk_forward") < mc.mean("walk_forward_average"):
        lines.append("לרדוף אחרי הסוכן שהצליח לאחרונה הניב פחות מהחזקת כל האסטרטגיות יחד.")
    lines.append("כל זה על נתונים סינתטיים: זו בדיקה של השיטה. ההכרעה האמיתית תגיע רק עם "
                 "נתונים אמיתיים ומעקב לאורך זמן.")
    return lines
