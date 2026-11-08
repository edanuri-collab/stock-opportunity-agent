"""Hebrew UI text. Internal keys (column names, statuses, page ids) stay English; views
translate at render time, so logic and tests do not depend on display language.

Agent names, company names, tickers and standard abbreviations (P/E, CAGR, t-stat) are kept
in English, as is common in Hebrew financial products.
"""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from engine.benchmarks import BUY_HOLD, MARKET, RANDOM

PAGES = {
    "Overview": "סקירה כללית",
    "Agents Leaderboard": "דירוג סוכנים",
    "Agent Catalog": "קטלוג ובונה סוכנים",
    "Stock Opportunities": "הזדמנויות במניות",
    "Stock Detail": "פרטי מניה",
    "Performance & Backtesting": "ביצועים ובקטסט",
    "Experiments": "ניסויים",
}

COLUMNS = {
    # leaderboard / metrics
    "Rank": "דירוג", "Agent": "סוכן", "Type": "סוג", "Status": "סטטוס", "Family": "משפחה",
    "Opportunity Score": "ציון הזדמנות", "Alpha": "אלפא", "Alpha t-stat": "t-stat אלפא",
    "Win Rate": "אחוז הצלחה", "Sharpe": "שארפ", "Max Drawdown": "ירידה מקסימלית",
    "Trades": "עסקאות", "Total Return": "תשואה כוללת", "Beats Random": "עוקף תיקים אקראיים",
    "CAGR": "CAGR", "Avg Win": "רווח ממוצע", "Avg Loss": "הפסד ממוצע",
    "Expectancy": "תוחלת לעסקה", "Profit Factor": "Profit Factor",
    "Alpha (vs market)": "אלפא (מול השוק)", "Beta": "בטא",
    "Benchmark": "מדד ייחוס", "Benchmark Return": "תשואת מדד הייחוס",
    "Excess CAGR": "CAGR עודף", "Beats Random Portfolios": "עוקף תיקים אקראיים",
    "Verdict": "מסקנה",
    # opportunities
    "Ticker": "טיקר", "Company": "חברה", "Sector": "סקטור", "Price": "מחיר",
    "Market Cap": "שווי שוק", "Value Score": "ציון ערך", "Catalyst Score": "ציון קטליזטור",
    "Momentum Score": "ציון מומנטום", "Risk Score": "ציון סיכון", "Top Agent": "סוכן מוביל",
    "Picked By": "מספר בוחרים", "Agents": "סוכנים שבחרו",
    # stock detail
    "Picked": "נבחרה",
    # experiments
    "Experiment": "ניסוי", "Run At": "מועד הרצה", "Seed": "Seed",
    "Dataset Version": "גרסת דאטה", "Code Version": "גרסת קוד", "Agent Version": "גרסת סוכנים",
    "Stocks": "מניות", "Months": "חודשים", "Top N": "Top N", "Signal": "סיגנל",
    "Rank By": "דירוג לפי", "Best Agent": "הסוכן המוביל", "Best Alpha": "אלפא מובילה",
    "t-stat": "t-stat", "Version": "גרסה",
    # regimes (columns of the regime table)
    "bull": "שוק עולה", "bear": "שוק יורד", "sideways": "שוק מדשדש",
}

VALUES = {
    "Strategy": "אסטרטגיה", "Control": "ביקורת", "Agent": "סוכן", "Benchmark": "מדד ייחוס",
    "Outperforming": "עדיף על השוק", "Underperforming": "נחות מהשוק",
    "Inconclusive": "לא מובהק",
    "Alpha likely real": "כנראה אלפא אמיתית",
    "Not distinguishable from luck": "לא ניתן להבדיל ממזל",
    "Significantly worse than benchmark": "גרוע מובהקות מהמדד",
    "Weak": "חלש", "Null": "אפס (ללא סיגנל)",
    MARKET: "מדד השוק", BUY_HOLD: "קנה והחזק", RANDOM: "בחירה אקראית",
    # sectors and industries of the synthetic universe
    "Technology": "טכנולוגיה", "Healthcare": "בריאות", "Financials": "פיננסים",
    "Energy": "אנרגיה", "Consumer": "צריכה", "Industrials": "תעשייה", "Utilities": "תשתיות",
    "Software": "תוכנה", "Semiconductors": "שבבים", "Hardware": "חומרה", "Biotech": "ביוטק",
    "Devices": "מכשור רפואי", "Services": "שירותים", "Banks": "בנקים", "Insurance": "ביטוח",
    "Asset Management": "ניהול נכסים", "Oil & Gas": "נפט וגז", "Renewables": "אנרגיה מתחדשת",
    "Pipelines": "צנרת", "Retail": "קמעונאות", "Food & Beverage": "מזון ומשקאות",
    "Leisure": "פנאי", "Machinery": "מכונות", "Logistics": "לוגיסטיקה",
    "Aerospace": "תעופה וחלל", "Electric": "חשמל", "Water": "מים", "Gas": "גז",
}

FACTORS = {
    "value": "ערך", "quality": "איכות", "momentum": "מומנטום", "catalyst": "קטליזטור",
    "breakout": "פריצה", "squeeze": "סקוויז", "contrarian": "קונטרריאן", "risk": "סיכון",
    "noise": "רעש",
}

RANK_KEYS = {"alpha": "אלפא", "sharpe": "שארפ", "total_return": "תשואה כוללת",
             "cagr": "CAGR", "expectancy": "תוחלת לעסקה"}

SIGNAL_MODES = {"Weak signal": "סיגנל חלש", "Null (no signal)": "ללא סיגנל (אפס)"}


def v(value: object) -> object:
    """Translate a display value if a translation exists."""
    return VALUES.get(value, value) if isinstance(value, str) else value


def col(name: str) -> str:
    return COLUMNS.get(name, name)


def frame(df: pd.DataFrame, value_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Hebrew copy of ``df``: translated values in ``value_columns``, then column names."""
    out = df.copy()
    for c in value_columns:
        if c in out.columns:
            out[c] = out[c].map(v)
    if out.index.name:
        out.index.name = col(out.index.name)
    return out.rename(columns=COLUMNS)


def values(items: Iterable[str]) -> List[str]:
    return [v(i) for i in items]


# ---- agent catalog & builder ----------------------------------------------------------------

FAMILIES = {
    "value": "ערך", "quality": "איכות", "momentum": "מומנטום", "growth": "צמיחה",
    "low_risk": "סיכון נמוך", "event": "אירועים", "contrarian": "קונטרריאני",
    "size": "גודל", "multi_factor": "רב-גורמי", "control": "ביקורת",
}
ORIGINS = {"core": "ליבה (קוד)", "known": "אסטרטגיה מוכרת", "original": "רעיון מקורי"}
SOURCES = {"core": "מובנה", "library": "ספרייה", "custom": "הסוכנים שלי", "meta": "סוכן-על"}
METHODS = {"weights": "שקלול גורמים", "rank": "דירוג לפי נתונים",
           "switch": "מחליף לפי מצב השוק", "meta": "שילוב החלטות של סוכנים"}
PREFER = {"high": "גבוה עדיף", "low": "נמוך עדיף"}
OPS = {"<": "<", "<=": "≤", ">": ">", ">=": "≥"}

FIELDS = {
    "market_cap": "שווי שוק", "pe_ratio": "מכפיל רווח (P/E)",
    "earnings_yield": "תשואת רווח (1/PE)", "price_to_sales": "מכפיל מכירות (P/S)",
    "fcf_yield": "תשואת תזרים חופשי", "revenue_growth": "צמיחה בהכנסות",
    "earnings_growth": "צמיחה ברווח", "debt_to_equity": "חוב להון",
    "momentum": "מומנטום 6 חודשים", "relative_strength": "חוזק יחסי",
    "distance_from_high": "מרחק משיא 52 שבועות", "volume_ratio": "יחס מחזור מסחר",
    "volatility": "תנודתיות", "catalyst_score": "ציון קטליזטור",
    "sentiment_score": "ציון סנטימנט", "short_interest": "שורט מהמניות הצפות",
    "days_to_cover": "ימים לכיסוי שורט",
    "price_to_book": "מכפיל הון (P/B)", "roe": "תשואה על ההון (ROE)",
    "roic": "תשואה על ההון המושקע (ROIC)", "gross_margin": "שולי רווח גולמי",
    "dividend_yield": "תשואת דיבידנד", "earnings_surprise": "הפתעת רווחים",
    "momentum_12_1": "מומנטום 12-1 חודשים", "return_1m": "תשואת החודש האחרון",
    "beta": "בטא", "market_trend": "מגמת השוק (6 חודשים)",
    "roe_change": "שינוי ב-ROE בשנה", "gross_margin_change": "שינוי בשולי הרווח בשנה",
    "debt_change": "שינוי בחוב להון בשנה", "f_score": "F-Score מקורב (0-7)",
    **{f"factor_{k}": f"גורם {v}" for k, v in FACTORS.items() if k != "noise"},
}
# how a field's value is typed in the builder: raw value = typed value x factor
UNIT_HINT = {"pct": ("%", 0.01), "money": ("מיליארד $", 1e9), "ratio": ("", 1.0),
             "score": ("0-100", 1.0)}

# Hebrew descriptions of the core (Python) agents; spec agents carry their own text.
CORE_DESCRIPTIONS = {
    "Value Hunter": "מחפש מניות זולות ביחס לרווח, למכירות ולתזרים המזומנים החופשי.",
    "Momentum Hunter": "מחפש מניות במגמת עלייה חזקה עם חוזק יחסי גבוה.",
    "Catalyst Hunter": "מחפש אירועים שעשויים להזיז את המניה: קטליזטורים וסנטימנט חיובי.",
    "Quality Hunter": "מחפש חברות איכותיות: צמיחה, תזרים חופשי ומאזן חזק.",
    "Contrarian": "מחפש מניות שירדו חדות בזמן שהנתונים העסקיים שלהן נשארו חזקים.",
    "Breakout Hunter": "מחפש מניות קרובות לשיא 52 שבועות עם מחזור מסחר גבוה ומומנטום.",
    "Short Squeeze Hunter": "מחפש מניות עם שורט גבוה, מחזור מסחר עולה ומומנטום (סקוויז).",
    "Balanced": "משקל שווה לערך, איכות, מומנטום וקטליזטור.",
    "Random Agent": "סוכן ביקורת: בוחר באקראי (באופן דטרמיניסטי) ומתעלם מכל הנתונים.",
    "Consensus": "סוכן-על: בוחר את המניות שהכי הרבה סוכני אסטרטגיה מחזיקים באותו חודש "
                 "(לפחות שניים).",
}


def description(agent) -> str:
    return CORE_DESCRIPTIONS.get(agent.name, agent.description)
