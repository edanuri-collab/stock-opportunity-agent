from __future__ import annotations

from html import escape
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from agents.configurable import build_agent
from agents.registry import CUSTOM_DIR, save_spec, spec_of
from agents.spec import FAMILIES, FIELDS, OPS, WEIGHT_FACTORS, AgentSpec, SpecError
from dashboard import analytics as an
from dashboard import i18n as he
from dashboard import theme
from dashboard.context import DashboardContext
from dashboard.views import page_header

BLANK = "(סוכן חדש מאפס)"
FIELD_BY_LABEL = {he.FIELDS[k]: k for k in FIELDS}
PREFER_BY_LABEL = {v: k for k, v in he.PREFER.items()}
OP_BY_LABEL = {he.OPS[k]: k for k in OPS}


def render(ctx: DashboardContext) -> None:
    page_header(ctx, "קטלוג ובונה סוכנים",
                "כל הסוכנים במערכת, הכללים של כל אחד, ובונה להוספת סוכנים חדשים בלי קוד.")
    if ctx.registry.errors:
        with st.expander(f"{len(ctx.registry.errors)} קבצי סוכנים לא נטענו", expanded=True,
                         icon=":material/warning:"):
            for file, err in ctx.registry.errors.items():
                st.markdown(f"- `{file}`: {err}")
    t_catalog, t_builder = st.tabs(["קטלוג הסוכנים", "בונה סוכנים"])
    with t_catalog:
        _catalog(ctx)
    with t_builder:
        _builder(ctx)


# ---- catalog -------------------------------------------------------------------------------

def _catalog(ctx: DashboardContext) -> None:
    df = an.catalog_frame(ctx.exp, ctx.registry)
    by_name = {a.name: a for a in ctx.registry.agents}
    df["Description"] = df["Agent"].map(lambda n: he.description(by_name[n]))
    counts = df["Source"].value_counts()
    c = st.columns(4)
    c[0].html(theme.card("סה\"כ סוכנים", str(len(df))))
    c[1].html(theme.card("מובנים", str(counts.get("core", 0)), "סוכני הליבה בקוד"))
    c[2].html(theme.card("מהספרייה", str(counts.get("library", 0)), "אסטרטגיות מוכנות"))
    c[3].html(theme.card("הסוכנים שלי", str(counts.get("custom", 0)), "נוצרו בבונה"))

    families = sorted(df["Family"].unique(), key=lambda f: FAMILIES.index(f))
    chosen = st.multiselect("סינון לפי משפחה", families, format_func=he.FAMILIES.get,
                            key="cat_family", placeholder="כל המשפחות")
    view = df[df["Family"].isin(chosen)] if chosen else df
    table = view.assign(Family=view["Family"].map(he.FAMILIES),
                        Origin=view["Origin"].map(he.ORIGINS),
                        Method=view["Method"].map(he.METHODS),
                        Source=view["Source"].map(lambda s: he.SOURCES.get(s, s)))
    table = table.rename(columns={"Family": "משפחה", "Origin": "מקור הרעיון",
                                  "Method": "שיטה", "Source": "מיקום", "Description": "תיאור",
                                  "Rank": "מקום בדירוג"}).rename(columns=he.COLUMNS)
    st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table),
                 column_config={he.col("Alpha"): st.column_config.NumberColumn(format="percent"),
                                he.col("Alpha t-stat"): st.column_config.NumberColumn(
                                    format="%.2f"),
                                "תיאור": st.column_config.TextColumn(width="large")})

    st.markdown("#### הכללים של כל סוכן")
    agents = {a.name: a for a in ctx.registry.agents}
    for name in view["Agent"]:
        agent = agents[name]
        spec, exact = spec_of(agent)
        title = f"{name} · {he.FAMILIES[agent.family]} · {he.ORIGINS[agent.origin]}"
        with st.expander(title):
            st.markdown(escape(he.description(agent)))
            if agent.reference:
                st.caption(f"מקור: {agent.reference}")
            if an.is_control(name):
                st.markdown("- בחירה פסאודו-אקראית (hash של טיקר ותאריך). מתעלם מכל הנתונים.")
            elif an.is_meta(agent):
                st.markdown("- רץ אחרי כל הסוכנים, על ההחלטות שלהם באותו תאריך בלבד.\n"
                            "- הציון: אחוז סוכני האסטרטגיה שמחזיקים את המניה ב-Top N.\n"
                            "- נבחרות רק מניות שלפחות שני סוכנים בחרו. סוכן הביקורת לא "
                            "מצביע.")
            else:
                st.markdown("\n".join(f"- {line}" for line in an.rules_he(spec)))
            if not exact and not an.is_control(name) and not an.is_meta(agent):
                st.caption("סוכן ליבה שמוגדר בקוד Python (agents/). "
                           "אפשר לשכפל אותו כנקודת התחלה בבונה הסוכנים.")
            st.caption(f"גרסה: {ctx.exp.agent_versions.get(name, '-')} · מיקום: "
                       f"{he.SOURCES.get(ctx.registry.sources.get(name, ''), '')}")


# ---- builder -------------------------------------------------------------------------------

def _to_display(field: str, raw: float) -> float:
    return raw / he.UNIT_HINT[FIELDS[field][0]][1]


def _to_raw(field: str, shown: float) -> float:
    return float(shown) * he.UNIT_HINT[FIELDS[field][0]][1]


def _filters_frame(spec: AgentSpec | None) -> pd.DataFrame:
    rows = [{"שדה": he.FIELDS[f.field], "תנאי": he.OPS[f.op],
             "ערך": _to_display(f.field, f.value)} for f in (spec.filters if spec else ())]
    return pd.DataFrame(rows, columns=["שדה", "תנאי", "ערך"])


def _rank_frame(spec: AgentSpec | None) -> pd.DataFrame:
    rows = [{"שדה": he.FIELDS[r.field], "עדיפות": he.PREFER[r.prefer], "משקל": r.weight}
            for r in (spec.rank if spec else ())]
    if not rows:
        rows = [{"שדה": he.FIELDS["momentum"], "עדיפות": he.PREFER["high"], "משקל": 1.0}]
    return pd.DataFrame(rows, columns=["שדה", "עדיפות", "משקל"])


def _complete(row: Dict[str, Any], keys: List[str]) -> bool:
    return all(row.get(k) is not None and not (isinstance(row.get(k), float)
                                               and pd.isna(row.get(k))) for k in keys)


def _save(spec: AgentSpec) -> None:
    """Button callback (runs before the rerun, so it may reset the template widget)."""
    try:
        path = save_spec(spec, CUSTOM_DIR)
    except SpecError as exc:
        st.session_state["b_error"] = f"השמירה נכשלה: {exc}"
        return
    st.session_state["b_saved"] = (f"{spec.name} נשמר בקובץ agents/custom/{path.name}, "
                                   "והוא כבר מופיע בכל המסכים.")
    st.session_state["b_template"] = BLANK


def _builder(ctx: DashboardContext) -> None:
    exp = ctx.exp
    if st.session_state.get("b_saved"):
        st.success(st.session_state.pop("b_saved"), icon=":material/check_circle:")
    if st.session_state.get("b_error"):
        st.error(st.session_state.pop("b_error"))
    st.markdown("בנה סוכן חדש: בחר **תנאי סינון** (אופציונלי) ושיטת ניקוד אחת. "
                "התצוגה המקדימה מריצה את הסוכן על אותם נתונים, ולאחר השמירה הוא מצטרף "
                "לכל המסכים.")
    templates = [BLANK] + [a.name for a in ctx.registry.agents
                           if not an.is_control(a.name) and not an.is_meta(a)]
    start = st.selectbox("להתחיל מ...", templates, key="b_template",
                         help="שכפול של סוכן קיים כנקודת התחלה לווריאציה")
    base = None
    if start != BLANK:
        base, _ = spec_of(next(a for a in ctx.registry.agents if a.name == start))
    k = f"b_{templates.index(start)}_"  # widget keys reset when the template changes
    if base is not None and base.switch is not None:
        st.info("לסוכן הזה יש החלפה לפי מצב השוק. הבונה עורך רק את השיטה הראשית, ולכן העותק "
                "יישמר בלי ההחלפה. אפשר להוסיף אותה ידנית בקובץ ה-TOML.")

    c1, c2, c3 = st.columns([1.4, 1, 1])
    name = c1.text_input("שם הסוכן (באנגלית, עד 40 תווים)",
                         value=f"{base.name} v2"[:40] if base else "", key=k + "name")
    family = c2.selectbox("משפחה", [f for f in FAMILIES if f != "control"],
                          index=[f for f in FAMILIES if f != "control"].index(base.family)
                          if base and base.family != "control" else 0,
                          format_func=he.FAMILIES.get, key=k + "family")
    origin = c3.radio("מקור הרעיון", ["original", "known"], format_func=he.ORIGINS.get,
                      index=1 if base and base.origin == "known" else 0, horizontal=True,
                      key=k + "origin")
    base_agent = next((a for a in ctx.registry.agents if a.name == start), None)
    description = st.text_area("תיאור האסטרטגיה",
                               value=he.description(base_agent) if base_agent else "",
                               key=k + "desc", height=80)
    reference = st.text_input("מקור / השראה (אופציונלי)", value=base.reference if base else "",
                              key=k + "ref")

    st.markdown("##### 1. תנאי סינון (אופציונלי)")
    st.caption("רק מניות שעומדות בכל התנאים יכולות להיבחר. אחוזים מקלידים כאחוז (15 = 15%), "
               "ושווי שוק במיליארדי דולרים.")
    filters = st.data_editor(
        _filters_frame(base), num_rows="dynamic", width="stretch", key=k + "filters",
        column_config={
            "שדה": st.column_config.SelectboxColumn(options=list(FIELD_BY_LABEL),
                                                    required=True, width="large"),
            "תנאי": st.column_config.SelectboxColumn(options=list(OP_BY_LABEL),
                                                     required=True),
            "ערך": st.column_config.NumberColumn(required=True, format="%.4g"),
        })

    st.markdown("##### 2. שיטת ניקוד")
    method = st.radio("שיטה", ["weights", "rank"], format_func=he.METHODS.get, horizontal=True,
                      index=1 if base and base.rank else 0, key=k + "method",
                      label_visibility="collapsed")
    weights: Dict[str, float] = {}
    rank_rows = pd.DataFrame()
    if method == "weights":
        st.caption("כמה משקל לתת לכל גורם (0-100). המשקלים מנורמלים אוטומטית לסכום 100%.")
        base_w = base.weights_map if base and base.weights else {}
        cols = st.columns(4)
        for i, f in enumerate(WEIGHT_FACTORS):
            weights[f] = cols[i % 4].slider(he.FACTORS[f], 0, 100,
                                            int(round(base_w.get(f, 0) * 100)),
                                            key=k + "w_" + f)
    else:
        st.caption("כל מניה מקבלת אחוזון (0-100) בכל שדה מול שאר המניות שעברו את הסינון, "
                   "והציון הוא ממוצע משוקלל של האחוזונים.")
        rank_rows = st.data_editor(
            _rank_frame(base), num_rows="dynamic", width="stretch", key=k + "rank",
            column_config={
                "שדה": st.column_config.SelectboxColumn(options=list(FIELD_BY_LABEL),
                                                        required=True, width="large"),
                "עדיפות": st.column_config.SelectboxColumn(options=list(PREFER_BY_LABEL),
                                                           required=True),
                "משקל": st.column_config.NumberColumn(min_value=0.1, required=True,
                                                      format="%.2f"),
            })
    risk_penalty = st.slider("3. קנס סיכון (כמה להוריד לפי ציון הסיכון)", 0.0, 1.0,
                             float(base.risk_penalty) if base else 0.0, 0.05, key=k + "risk")

    raw: Dict[str, Any] = {
        "name": name, "family": family, "origin": origin, "description": description,
        "reference": reference, "risk_penalty": risk_penalty,
        "filters": [{"field": FIELD_BY_LABEL[r["שדה"]], "op": OP_BY_LABEL[r["תנאי"]],
                     "value": _to_raw(FIELD_BY_LABEL[r["שדה"]], r["ערך"])}
                    for r in filters.to_dict("records") if _complete(r, ["שדה", "תנאי", "ערך"])],
    }
    if method == "weights":
        raw["weights"] = {f: float(w) for f, w in weights.items() if w > 0}
    else:
        raw["rank"] = [{"field": FIELD_BY_LABEL[r["שדה"]],
                        "prefer": PREFER_BY_LABEL[r["עדיפות"]], "weight": float(r["משקל"])}
                       for r in rank_rows.to_dict("records")
                       if _complete(r, ["שדה", "עדיפות", "משקל"])]

    try:
        spec = AgentSpec.from_dict(raw)
    except SpecError as exc:
        st.warning(f"הסוכן עדיין לא תקין: {exc}")
        return
    if spec.name in ctx.registry.sources:
        st.warning(f"כבר קיים סוכן בשם {spec.name}. בחר שם אחר.")
        return

    st.markdown("##### תצוגה מקדימה (על הנתונים של הניסוי הנוכחי)")
    st.markdown("\n".join(f"- {line}" for line in an.rules_he(spec)))
    prev = an.preview_agent(exp, build_agent(spec))
    m, a = prev["metrics"], prev["alpha"]
    p = st.columns(5)
    p[0].html(theme.card("תשואה כוללת", f"{m.total_return:+.1%}",
                         f"הסוכן האקראי: {prev['random_agent_return']:+.1%}",
                         theme.tone(m.total_return)))
    p[1].html(theme.card("אלפא מול השוק", f"{a.alpha:+.1%}", f"t = {a.t_stat:.2f}",
                         theme.tone(a.alpha)))
    p[2].html(theme.card("שארפ", f"{m.sharpe:.2f}", f"ירידה מקסימלית {m.max_drawdown:.1%}"))
    p[3].html(theme.card("עסקאות", str(m.num_trades),
                         f"בממוצע {prev['avg_picks']:.1f} מניות בחודש"))
    p[4].html(theme.card("עוברות סינון היום", f"{prev['eligible_latest']} / {prev['universe']}",
                         "מניות שמותר לבחור"))
    if prev["eligible_latest"] == 0:
        st.info("אף מניה לא עוברת את התנאים בתאריך האחרון. כדאי לרכך את הסינון.")
    st.dataframe(he.frame(prev["picks"]), hide_index=True, width="stretch",
                 column_config={he.col("Opportunity Score"): st.column_config.ProgressColumn(
                     format="%.1f", min_value=0, max_value=100)})
    st.caption("ריצה בודדת על נתונים סינתטיים. תוצאה טובה כאן עדיין לא מוכיחה יתרון: "
               "שווה לבדוק בכמה Seeds שונים.")

    st.button("שמירת הסוכן", type="primary", icon=":material/save:", key=k + "save",
              on_click=_save, args=(spec,))
