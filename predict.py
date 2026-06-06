"""
UFC fight outcome predictor.

Given two fighter names, fetches their current stats (from local dataset +
ufcstats.com for any new fights), computes differential features, and
returns win probabilities from the XGBoost model.

Usage:
  python predict.py "Jon Jones" "Stipe Miocic"

  Or import and call predict_fight() directly.
"""

from __future__ import annotations

import sys
import time
import threading
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from playwright.sync_api import sync_playwright, Browser, Page

from scraper import ufcstats, features, data_loader

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

AVAILABLE_MODELS: dict[str, dict] = {
    "random_forest": {
        "label": "Random Forest",
        "accuracy": 0.7284,
        "model_path": MODELS_DIR / "random_forest.joblib",
        "features_path": None,  # uses model.feature_names_in_
    },
    "xgboost": {
        "label": "XGBoost",
        "accuracy": 0.6489,
        "model_path": MODELS_DIR / "xgboost_exp1.joblib",
        "features_path": MODELS_DIR / "xgboost_exp1_features.joblib",
    },
}

# ---------------------------------------------------------------------------
# Fighter data cache  { name_lower: (profile, history, timestamp) }
# Avoids re-scraping ufcstats.com within the same server session.
# ---------------------------------------------------------------------------

CACHE_TTL = 3600  # seconds
_fighter_cache: dict[str, tuple] = {}
_cache_lock = threading.Lock()


def _cache_get(name: str) -> tuple[dict, list] | None:
    key = name.strip().lower()
    with _cache_lock:
        entry = _fighter_cache.get(key)
    if entry and (time.time() - entry[2]) < CACHE_TTL:
        return entry[0], entry[1]
    return None


def _cache_set(name: str, profile: dict, history: list) -> None:
    key = name.strip().lower()
    with _cache_lock:
        _fighter_cache[key] = (profile, history, time.time())


_model_cache: dict[str, tuple] = {}
_model_cache_lock = threading.Lock()


def _load_model(key: str = "xgboost") -> tuple:
    with _model_cache_lock:
        if key not in _model_cache:
            cfg = AVAILABLE_MODELS.get(key)
            if cfg is None:
                raise ValueError(f"Unknown model '{key}'. Valid options: {list(AVAILABLE_MODELS)}")
            model = joblib.load(cfg["model_path"])
            features = (joblib.load(cfg["features_path"])
                        if cfg["features_path"] else list(model.feature_names_in_))
            _model_cache[key] = (model, features)
    return _model_cache[key]


def _resolve_fighter(page: Page, name: str) -> tuple[dict, list[dict]]:
    """
    Return (profile, fight_history) for a named fighter.

    Order of precedence:
      1. In-memory cache (TTL = 1 hour)
      2. Local dataset (no network)
      3. ufcstats.com scrape for missing profile or post-cutoff fights
    """
    cached = _cache_get(name)
    if cached:
        return cached

    profile = data_loader.get_fighter_profile(name)
    history = data_loader.get_fighter_history(name)

    profile_url = (profile or {}).get("fighter_url")
    if not profile_url:
        profile_url = ufcstats.search_fighter(page, name)
        if not profile_url:
            raise ValueError(f"Fighter not found: '{name}'")

    if not profile:
        profile = ufcstats.scrape_fighter_profile(page, profile_url)

    new_fights = ufcstats.get_new_fights(page, profile_url, name)
    if new_fights:
        history = history + new_fights

    _cache_set(name, profile, history)
    return profile, history


def predict_fight(
    f1_name: str,
    f2_name: str,
    fight_date: date | None = None,
    browser: Browser | None = None,
    model_key: str = "random_forest",
) -> dict:
    """
    Predict the outcome of f1_name vs f2_name.

    Returns:
      {
        "f1_name":       str,
        "f2_name":       str,
        "f1_win_prob":   float,   # probability f1 wins (0–1)
        "f2_win_prob":   float,   # probability f2 wins (0–1)
        "f1_experience": int,
        "f2_experience": int,
        "f1_record":     str,     # e.g. "22-2-0"
        "f2_record":     str,
      }

    Raises ValueError if either fighter cannot be found.
    """
    model, feature_cols = _load_model(model_key)

    def _validate_matchup(p1: dict, p2: dict) -> None:
        g1, g2 = p1.get("gender"), p2.get("gender")
        if g1 and g2 and g1 != g2:
            raise ValueError(
                f"Cannot match {p1.get('name', f1_name)} ({g1}) "
                f"against {p2.get('name', f2_name)} ({g2}): gender mismatch."
            )
        wc1, wc2 = p1.get("weight_class"), p2.get("weight_class")
        if wc1 and wc2 and wc1 != wc2:
            raise ValueError(
                f"Cannot match {p1.get('name', f1_name)} ({wc1}) "
                f"against {p2.get('name', f2_name)} ({wc2}): different weight classes."
            )

    def _run(page: Page) -> dict:
        f1_profile, f1_history = _resolve_fighter(page, f1_name)
        f2_profile, f2_history = _resolve_fighter(page, f2_name)
        _validate_matchup(f1_profile, f2_profile)

        f1_stats = features.compute_fighter_stats(f1_history)
        f2_stats = features.compute_fighter_stats(f2_history)

        for fname, stats in ((f1_name, f1_stats), (f2_name, f2_stats)):
            if stats["experience"] < 5:
                raise ValueError(
                    f"{fname} has only {stats['experience']} UFC fight(s) — "
                    f"a minimum of 5 is required for a reliable prediction."
                )

        row = features.build_feature_row(f1_stats, f1_profile, f2_stats, f2_profile, fight_date)
        row = row[feature_cols]

        proba = model.predict_proba(row)[0]  # [p_f2_wins, p_f1_wins]
        f1_win_prob = float(proba[1])
        f2_win_prob = float(proba[0])

        def _record(s):
            return f"{s['wins']}-{s['losses']}-{s['draws']}"

        return {
            "f1_name":       f1_profile.get("name") or f1_name,
            "f2_name":       f2_profile.get("name") or f2_name,
            "f1_win_prob":   round(f1_win_prob, 4),
            "f2_win_prob":   round(f2_win_prob, 4),
            "f1_experience": f1_stats["experience"],
            "f2_experience": f2_stats["experience"],
            "f1_record":     _record(f1_stats),
            "f2_record":     _record(f2_stats),
            "model_key":     model_key,
            "model_label":   AVAILABLE_MODELS[model_key]["label"],
        }

    if browser is not None:
        page = browser.new_page()
        try:
            return _run(page)
        finally:
            page.close()
    else:
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            page = b.new_page()
            try:
                return _run(page)
            finally:
                b.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py \"Fighter One\" \"Fighter Two\"")
        sys.exit(1)

    f1, f2 = sys.argv[1], sys.argv[2]
    print(f"\nPredicting: {f1} vs {f2} ...\n")

    result = predict_fight(f1, f2)

    width = max(len(result["f1_name"]), len(result["f2_name"])) + 2
    print(f"{'Fighter':<{width}}  {'Win %':>7}  {'Record':>10}")
    print("-" * (width + 22))
    print(f"{result['f1_name']:<{width}}  {result['f1_win_prob']*100:>6.1f}%  {result['f1_record']:>10}")
    print(f"{result['f2_name']:<{width}}  {result['f2_win_prob']*100:>6.1f}%  {result['f2_record']:>10}")
    print()


if __name__ == "__main__":
    main()
