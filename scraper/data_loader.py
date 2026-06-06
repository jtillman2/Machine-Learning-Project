"""
Loads and caches the existing UFC dataset (covers fights through Nov 2024).

Provides fighter history lookup so the scraper only fetches genuinely new fights.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"


_WC_KEYWORDS = [
    "strawweight", "flyweight", "bantamweight", "featherweight",
    "lightweight", "welterweight", "middleweight", "light heavyweight",
    "heavyweight",
]


def _normalize_weight_class(wc: str) -> str | None:
    """Extract canonical weight class name from a raw weight_class string."""
    wc_lower = wc.lower()
    for name in _WC_KEYWORDS:  # "light heavyweight" before "heavyweight"
        if name in wc_lower:
            return name
    return None


@lru_cache(maxsize=1)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load fights, fighters, events CSVs (cached)."""
    fights = pd.read_csv(RAW / "ufc_event_fight_stats.csv")
    fighters = pd.read_csv(RAW / "ufc_fighters.csv")
    events = pd.read_csv(RAW / "ufc_events.csv")

    # Merge event date onto fights
    fights = fights.merge(
        events[["url_link", "event_date"]],
        left_on="event_url",
        right_on="url_link",
        how="left",
    )
    fights["event_date"] = pd.to_datetime(fights["event_date"])
    fights = fights.sort_values("event_date")

    fights["f1_ctrl_time"] = fights["f1_ctrl_time"].fillna(0)
    fights["f2_ctrl_time"] = fights["f2_ctrl_time"].fillna(0)

    return fights, fighters, events


def get_fighter_profile(name: str) -> dict | None:
    """
    Look up physical attributes from the fighters CSV by name.
    Returns a dict compatible with scrape_fighter_profile(), or None if not found.
    Includes 'gender' ('male'/'female'/None) and 'weight_class' derived from fight history.
    """
    fights_df, fighters_df, _ = load_data()
    name_lower = name.strip().lower()
    mask = fighters_df["fighter_name"].str.lower().str.strip() == name_lower
    rows = fighters_df[mask]

    if rows.empty:
        # Try partial match
        tokens = name_lower.split()
        for _, row in fighters_df.iterrows():
            if all(t in row["fighter_name"].lower() for t in tokens):
                rows = fighters_df[fighters_df.index == row.name]
                break

    if rows.empty:
        return None

    r = rows.iloc[0]
    fighter_id = int(r["fighter_id"])

    dob = None
    try:
        parsed = pd.to_datetime(r["fighter_dob"])
        if parsed is not pd.NaT and not pd.isnull(parsed):
            dob = parsed.date()
    except Exception:
        pass

    # Gender and weight class from actual fight history (most reliable source)
    fighter_fights = fights_df[
        (fights_df["f1_id"] == fighter_id) | (fights_df["f2_id"] == fighter_id)
    ]
    gender = None
    weight_class = None
    if not fighter_fights.empty:
        wc_series = fighter_fights["weight_class"].fillna("")
        gender = "female" if wc_series.str.contains("Women", case=False).any() else "male"
        weight_class = _normalize_weight_class(wc_series.iloc[-1])

    return {
        "name": r["fighter_name"],
        "height_cm": r.get("fighter_height_cm") if not pd.isna(r.get("fighter_height_cm", float("nan"))) else None,
        "weight_lbs": r.get("fighter_weight_lbs") if not pd.isna(r.get("fighter_weight_lbs", float("nan"))) else None,
        "reach_cm": r.get("fighter_reach_cm") if not pd.isna(r.get("fighter_reach_cm", float("nan"))) else None,
        "stance": r.get("fighter_stance") if not pd.isna(r.get("fighter_stance", float("nan"))) else None,
        "dob": dob,
        "fighter_url": r.get("fighter_url"),
        "gender": gender,
        "weight_class": weight_class,
    }


def get_fighter_history(name: str) -> list[dict]:
    """
    Return per-fight stats from the existing dataset for the named fighter,
    sorted chronologically (oldest first).

    Stat keys match STAT_COLS in features.py, plus 'result' and 'date'.
    """
    fights_df, fighters_df, _ = load_data()

    # Find fighter_id
    name_lower = name.strip().lower()
    mask = fighters_df["fighter_name"].str.lower().str.strip() == name_lower
    rows = fighters_df[mask]

    if rows.empty:
        tokens = name_lower.split()
        for _, row in fighters_df.iterrows():
            if all(t in row["fighter_name"].lower() for t in tokens):
                rows = fighters_df[fighters_df.index == row.name]
                break

    if rows.empty:
        return []

    fighter_id = int(rows.iloc[0]["fighter_id"])

    # Find all fights for this fighter as f1 or f2
    f1_fights = fights_df[fights_df["f1_id"] == fighter_id].copy()
    f2_fights = fights_df[fights_df["f2_id"] == fighter_id].copy()

    stat_cols = [
        "knockdowns", "sig_strike_atts", "sig_strikes", "tot_strike_atts", "tot_strikes",
        "takedown_atts", "takedowns", "submissions", "reversals", "ctrl_time",
        "head_strike_atts", "head_strikes", "body_strike_atts", "body_strikes",
        "leg_strike_atts", "leg_strikes", "dist_strike_atts", "dist_strikes",
        "clinch_atts", "clinchs", "ground_atts", "grounds", "total_fight_time",
    ]

    history = []

    for _, row in f1_fights.iterrows():
        result_id = str(row.get("result", "")).strip()
        result = "win" if result_id == str(fighter_id) else ("draw" if not result_id or result_id == "0" else "loss")
        entry = {"date": row["event_date"].date(), "result": result}
        for s in stat_cols:
            entry[s] = float(row.get(f"f1_{s}", 0) or 0)
        history.append(entry)

    for _, row in f2_fights.iterrows():
        result_id = str(row.get("result", "")).strip()
        result = "win" if result_id == str(fighter_id) else ("draw" if not result_id or result_id == "0" else "loss")
        entry = {"date": row["event_date"].date(), "result": result}
        for s in stat_cols:
            entry[s] = float(row.get(f"f2_{s}", 0) or 0)
        history.append(entry)

    history.sort(key=lambda e: e["date"])
    return history
