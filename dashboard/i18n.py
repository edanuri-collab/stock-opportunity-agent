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
    "Stock Opportunities": "הזדמנויות במניות",
    "Stock Detail": "פרטי מניה",
    "Performance & Backtesting": "ביצועים ובקטסט",
    "Experiments": "ניסויים",
}

COLUMNS = {
    # leaderboard / metrics
    "Rank": "דירוג", "Agent": "סוכן", "Type": "סוג", "Status": "סטטוס",
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
