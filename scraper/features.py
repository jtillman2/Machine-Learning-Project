"""
Feature engineering for fight prediction.

Replicates the feature pipeline from notebooks/feature_engineering.ipynb:
  - Career per-fight averages (cumulative sum / experience)
  - Last-4-fight rolling averages
  - W/L/D record and experience count
  - Physical attribute differentials

All computation is done from a list of raw per-fight stat dicts
(sorted chronologically, oldest first).
"""

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

STAT_COLS = [
    "knockdowns", "sig_strike_atts", "sig_strikes", "tot_strike_atts", "tot_strikes",
    "takedown_atts", "takedowns", "submissions", "reversals", "ctrl_time",
    "head_strike_atts", "head_strikes", "body_strike_atts", "body_strikes",
    "leg_strike_atts", "leg_strikes", "dist_strike_atts", "dist_strikes",
    "clinch_atts", "clinchs", "ground_atts", "grounds", "total_fight_time",
]

WEIGHT_CLASS_MAP = {
    # maps (lower_bound_exclusive, upper_bound_inclusive) -> mens_class_encoded
    (0, 125): 0,    # Flyweight
    (125, 135): 1,  # Bantamweight
    (135, 145): 2,  # Featherweight
    (145, 155): 3,  # Lightweight
    (155, 170): 4,  # Welterweight
    (170, 185): 5,  # Middleweight
    (185, 205): 6,  # Light Heavyweight
    (205, 9999): 7, # Heavyweight
}


def weight_to_class(weight_lbs: float | None) -> int:
    if weight_lbs is None:
        return -1
    for (lo, hi), enc in WEIGHT_CLASS_MAP.items():
        if lo < weight_lbs <= hi:
            return enc
    return 7  # default heavyweight


def compute_fighter_stats(fights: list[dict]) -> dict:
    """
    Given a chronologically sorted list of past fights (each a dict with
    STAT_COLS keys plus 'result'), return computed features:

      career_avg_{stat}  — career per-fight average going into next fight
      last4_{stat}       — rolling mean of last 4 fights
      experience         — number of prior fights
      wins / losses / draws
    """
    if not fights:
        return {
            **{f"career_avg_{s}": 0.0 for s in STAT_COLS},
            **{f"last4_{s}": 0.0 for s in STAT_COLS},
            "experience": 0, "wins": 0, "losses": 0, "draws": 0,
        }

    n = len(fights)
    career = {s: sum(f.get(s, 0.0) or 0.0 for f in fights) / n for s in STAT_COLS}

    last4 = fights[-4:]
    l4n = len(last4)
    last4_avg = {s: sum(f.get(s, 0.0) or 0.0 for f in last4) / l4n for s in STAT_COLS}

    wins = sum(1 for f in fights if f.get("result") == "win")
    losses = sum(1 for f in fights if f.get("result") == "loss")
    draws = n - wins - losses

    return {
        **{f"career_avg_{s}": career[s] for s in STAT_COLS},
        **{f"last4_{s}": last4_avg[s] for s in STAT_COLS},
        "experience": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }


def build_feature_row(
    f1_stats: dict,
    f1_profile: dict,
    f2_stats: dict,
    f2_profile: dict,
    fight_date: date | None = None,
) -> pd.DataFrame:
    """
    Build the 63-feature row expected by the XGBoost model.

    f1_stats / f2_stats: output of compute_fighter_stats()
    f1_profile / f2_profile: output of scrape_fighter_profile()
    fight_date: upcoming fight date (used to compute age; defaults to today)

    Returns a single-row DataFrame with the correct column order.
    """
    if fight_date is None:
        fight_date = date.today()

    def _age(dob) -> float:
        if dob is None or dob is np.nan:
            return 30.0  # median fallback
        try:
            if pd.isnull(dob):
                return 30.0
        except (TypeError, ValueError):
            pass
        if isinstance(dob, str):
            try:
                dob = datetime.strptime(dob, "%Y-%m-%d").date()
            except ValueError:
                return 30.0
        return (fight_date - dob).days / 365.25

    def _dob_days(dob) -> float:
        """DOB as days since epoch (used for diff_fighter_dob)."""
        if dob is None or dob is np.nan:
            return 0.0
        try:
            if pd.isnull(dob):
                return 0.0
        except (TypeError, ValueError):
            pass
        if isinstance(dob, str):
            try:
                dob = datetime.strptime(dob, "%Y-%m-%d").date()
            except ValueError:
                return 0.0
        return float((dob - date(1970, 1, 1)).days)

    def _stance_dummies(stance: str | None) -> dict:
        s = (stance or "").strip().lower()
        return {
            "orthodox": int(s == "orthodox"),
            "southpaw": int(s == "southpaw"),
            "switch":   int(s == "switch"),
        }

    f1_stance = _stance_dummies(f1_profile.get("stance"))
    f2_stance = _stance_dummies(f2_profile.get("stance"))

    # Weight class from heavier fighter's weight (fight-level context)
    w1 = f1_profile.get("weight_lbs") or 0
    w2 = f2_profile.get("weight_lbs") or 0
    mens_class = weight_to_class(max(w1, w2) if (w1 or w2) else None)

    # Differentials: f1 value - f2 value
    def diff(key1, key2=None):
        k2 = key2 or key1
        return (f1_stats.get(key1) or 0.0) - (f2_stats.get(k2) or 0.0)

    row = {
        # Ages
        "f1_age_during": _age(f1_profile.get("dob")),
        "f2_age_during": _age(f2_profile.get("dob")),

        # Weight class
        "mens_class_encoded": mens_class,
        "womens_class_encoded": -1,  # men's only

        # Stance OHE
        "f1_orthodox": f1_stance["orthodox"],
        "f1_southpaw": f1_stance["southpaw"],
        "f1_switch":   f1_stance["switch"],
        "f2_orthodox": f2_stance["orthodox"],
        "f2_southpaw": f2_stance["southpaw"],
        "f2_switch":   f2_stance["switch"],

        # Physical diffs
        "diff_fighter_dob": _dob_days(f1_profile.get("dob")) - _dob_days(f2_profile.get("dob")),
        "diff_height_cm": (f1_profile.get("height_cm") or 0) - (f2_profile.get("height_cm") or 0),
        "diff_fighter_weight_lbs": (f1_profile.get("weight_lbs") or 0) - (f2_profile.get("weight_lbs") or 0),
        "diff_fighter_reach_cm": (f1_profile.get("reach_cm") or 0) - (f2_profile.get("reach_cm") or 0),

        # Career average diffs
        "diff_knockdowns":        diff("career_avg_knockdowns"),
        "diff_sig_strike_atts":   diff("career_avg_sig_strike_atts"),
        "diff_sig_strikes":       diff("career_avg_sig_strikes"),
        "diff_tot_strike_atts":   diff("career_avg_tot_strike_atts"),
        "diff_tot_strikes":       diff("career_avg_tot_strikes"),
        "diff_takedown_atts":     diff("career_avg_takedown_atts"),
        "diff_takedowns":         diff("career_avg_takedowns"),
        "diff_submissions":       diff("career_avg_submissions"),
        "diff_reversals":         diff("career_avg_reversals"),
        "diff_ctrl_time":         diff("career_avg_ctrl_time"),
        "diff_head_strike_atts":  diff("career_avg_head_strike_atts"),
        "diff_head_strikes":      diff("career_avg_head_strikes"),
        "diff_body_strike_atts":  diff("career_avg_body_strike_atts"),
        "diff_body_strikes":      diff("career_avg_body_strikes"),
        "diff_leg_strike_atts":   diff("career_avg_leg_strike_atts"),
        "diff_leg_strikes":       diff("career_avg_leg_strikes"),
        "diff_dist_strike_atts":  diff("career_avg_dist_strike_atts"),
        "diff_dist_strikes":      diff("career_avg_dist_strikes"),
        "diff_clinch_atts":       diff("career_avg_clinch_atts"),
        "diff_clinchs":           diff("career_avg_clinchs"),
        "diff_ground_atts":       diff("career_avg_ground_atts"),
        "diff_grounds":           diff("career_avg_grounds"),
        "diff_total_fight_time":  diff("career_avg_total_fight_time"),

        # Record diffs
        "diff_experience": (f1_stats.get("experience") or 0) - (f2_stats.get("experience") or 0),
        "diff_wins":       (f1_stats.get("wins") or 0)       - (f2_stats.get("wins") or 0),
        "diff_losses":     (f1_stats.get("losses") or 0)     - (f2_stats.get("losses") or 0),

        # Last-4 diffs
        **{f"diff_{s}_last4": diff(f"last4_{s}") for s in STAT_COLS},
    }

    # Canonical column order (must match training data exactly)
    col_order = [
        "f1_age_during", "f2_age_during",
        "mens_class_encoded", "womens_class_encoded",
        "f1_orthodox", "f1_southpaw", "f1_switch",
        "f2_orthodox", "f2_southpaw", "f2_switch",
        "diff_fighter_dob", "diff_height_cm",
        "diff_knockdowns", "diff_sig_strike_atts", "diff_sig_strikes",
        "diff_tot_strike_atts", "diff_tot_strikes",
        "diff_takedown_atts", "diff_takedowns",
        "diff_submissions", "diff_reversals", "diff_ctrl_time",
        "diff_head_strike_atts", "diff_head_strikes",
        "diff_body_strike_atts", "diff_body_strikes",
        "diff_leg_strike_atts", "diff_leg_strikes",
        "diff_dist_strike_atts", "diff_dist_strikes",
        "diff_clinch_atts", "diff_clinchs",
        "diff_ground_atts", "diff_grounds",
        "diff_total_fight_time",
        "diff_experience", "diff_wins", "diff_losses",
        "diff_fighter_weight_lbs", "diff_fighter_reach_cm",
    ] + [f"diff_{s}_last4" for s in STAT_COLS]

    return pd.DataFrame([row], columns=col_order)
