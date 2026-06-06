"""
Scraper for ufcstats.com using Playwright.

Hybrid approach:
  - Existing dataset covers fights through ~Nov 2024
  - Only scrapes fights not in the local dataset
  - Uses a single shared Playwright browser instance
"""

import re
from datetime import datetime, date

from bs4 import BeautifulSoup
from playwright.sync_api import Browser, Page


DATASET_CUTOFF = date(2024, 11, 9)  # last date covered by ufc_event_fight_stats.csv
BASE_URL = "http://www.ufcstats.com"

STAT_COLS = [
    "knockdowns", "sig_strike_atts", "sig_strikes", "tot_strike_atts", "tot_strikes",
    "takedown_atts", "takedowns", "submissions", "reversals", "ctrl_time",
    "head_strike_atts", "head_strikes", "body_strike_atts", "body_strikes",
    "leg_strike_atts", "leg_strikes", "dist_strike_atts", "dist_strikes",
    "clinch_atts", "clinchs", "ground_atts", "grounds", "total_fight_time",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_of(s: str) -> tuple[int, int]:
    """'96 of 119' -> (landed=96, attempts=119)"""
    m = re.match(r"(\d+)\s+of\s+(\d+)", s.strip())
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _parse_ctrl(s: str) -> float:
    """'3:51' -> 231.0 seconds"""
    m = re.match(r"(\d+):(\d{2})", s.strip())
    return float(int(m.group(1)) * 60 + int(m.group(2))) if m else 0.0


def _parse_height_cm(h: str) -> float | None:
    """\"6' 4\"\" -> 193.04 cm"""
    m = re.match(r"(\d+)'\s*(\d+)", h)
    if m:
        return round((int(m.group(1)) * 12 + int(m.group(2))) * 2.54, 2)
    return None


def _parse_weight_lbs(w: str) -> float | None:
    """'248 lbs.' -> 248.0"""
    m = re.match(r"(\d+\.?\d*)\s*lbs", w)
    return float(m.group(1)) if m else None


def _parse_reach_cm(r: str) -> float | None:
    """'84\"' or '84.0\"' -> cm"""
    m = re.match(r"(\d+\.?\d*)[\"']", r.strip())
    return round(float(m.group(1)) * 2.54, 2) if m else None


def _parse_date(s: str) -> date | None:
    """'Nov. 16, 2024' or 'Nov 16, 2024' -> date object"""
    s = s.strip().replace(".", "").replace(",", "")
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def _total_fight_seconds(round_num: int, time_str: str) -> float:
    """Round 3 + '4:29' -> 869 seconds total"""
    m = re.match(r"(\d+):(\d{2})", time_str.strip())
    if not m:
        return 0.0
    secs_in_round = int(m.group(1)) * 60 + int(m.group(2))
    return float((round_num - 1) * 300 + secs_in_round)


def _weight_to_gender(weight_lbs: float | None) -> str | None:
    """Infer gender from weight (heuristic for scraped profiles not in local dataset).
    Strawweight (≤115 lbs) is women-only in UFC; Lightweight+ (≥155 lbs) is men-only.
    125/135/145 lbs are ambiguous — both divisions exist."""
    if weight_lbs is None:
        return None
    if weight_lbs <= 115:
        return "female"
    if weight_lbs >= 155:
        return "male"
    return None


def _weight_to_class_name(weight_lbs: float | None) -> str | None:
    """Map fighter weight to UFC weight class name."""
    if weight_lbs is None:
        return None
    if weight_lbs <= 115:  return "strawweight"
    if weight_lbs <= 125:  return "flyweight"
    if weight_lbs <= 135:  return "bantamweight"
    if weight_lbs <= 145:  return "featherweight"
    if weight_lbs <= 155:  return "lightweight"
    if weight_lbs <= 170:  return "welterweight"
    if weight_lbs <= 185:  return "middleweight"
    if weight_lbs <= 205:  return "light heavyweight"
    return "heavyweight"


def _fuzzy_name_match(target: str, candidates: list[str]) -> str | None:
    """Case-insensitive substring match; returns best candidate or None."""
    target_lower = target.strip().lower()
    # exact
    for c in candidates:
        if c.strip().lower() == target_lower:
            return c
    # target tokens all appear in candidate
    tokens = target_lower.split()
    for c in candidates:
        if all(t in c.lower() for t in tokens):
            return c
    return None


def _get_soup(page: Page, url: str) -> BeautifulSoup:
    page.goto(url, wait_until="networkidle", timeout=30000)
    return BeautifulSoup(page.content(), "html.parser")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_fighter(page: Page, name: str) -> str | None:
    """
    Search ufcstats.com by last name (+ optional first name) and return the
    best-matching fighter's profile URL, or None if not found.
    """
    parts = name.strip().split()
    last = parts[-1] if parts else name
    url = f"{BASE_URL}/statistics/fighters/search?query={'+'.join(parts)}"
    soup = _get_soup(page, url)

    rows = soup.select("tr.b-statistics__table-row")
    best_url = None
    best_score = -1

    target_lower = name.strip().lower()

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        first = cells[0].get_text(strip=True)
        last_name = cells[1].get_text(strip=True)
        nickname = cells[2].get_text(strip=True)
        full_name = f"{first} {last_name}".strip()

        link = row.find("a", href=re.compile(r"fighter-details"))
        if not link:
            continue
        profile_url = link["href"]

        # Score: exact full name match > last name match > partial
        full_lower = full_name.lower()
        score = 0
        if full_lower == target_lower:
            score = 3
        elif all(t in full_lower for t in target_lower.split()):
            score = 2
        elif last.lower() in full_lower:
            score = 1

        if score > best_score:
            best_score = score
            best_url = profile_url

    return best_url if best_score > 0 else None


def scrape_fighter_profile(page: Page, profile_url: str) -> dict:
    """
    Return physical attributes from a fighter's profile page.

    Keys: name, height_cm, weight_lbs, reach_cm, stance, dob
    """
    soup = _get_soup(page, profile_url)

    # Name
    name_tag = soup.select_one(".b-content__title-highlight")
    name = name_tag.get_text(strip=True) if name_tag else ""

    profile = {"name": name, "height_cm": None, "weight_lbs": None,
               "reach_cm": None, "stance": None, "dob": None,
               "gender": None, "weight_class": None, "image_url": None}

    for li in soup.select(".b-list__box-list-item"):
        text = li.get_text(separator="|", strip=True)
        if "Height:" in text:
            val = text.split("|", 1)[-1].strip()
            profile["height_cm"] = _parse_height_cm(val)
        elif "Weight:" in text:
            val = text.split("|", 1)[-1].strip()
            profile["weight_lbs"] = _parse_weight_lbs(val)
        elif "Reach:" in text:
            val = text.split("|", 1)[-1].strip()
            profile["reach_cm"] = _parse_reach_cm(val)
        elif "STANCE:" in text:
            val = text.split("|", 1)[-1].strip()
            profile["stance"] = val if val and val != "--" else None
        elif "DOB:" in text:
            val = text.split("|", 1)[-1].strip()
            profile["dob"] = _parse_date(val)

    profile["gender"] = _weight_to_gender(profile["weight_lbs"])
    profile["weight_class"] = _weight_to_class_name(profile["weight_lbs"])

    # Fighter photo — try known selectors, fall back to any /media/fighters img
    img_tag = (
        soup.select_one(".b-fighter-page__fighter-img") or
        soup.select_one(".b-content__intro-image img")
    )
    if not img_tag:
        for img in soup.find_all("img"):
            if "/media/fighters" in img.get("src", ""):
                img_tag = img
                break
    profile["image_url"] = img_tag["src"].strip() if img_tag and img_tag.get("src") else None

    return profile


def scrape_fight_history(page: Page, profile_url: str) -> list[dict]:
    """
    Return a list of all fights from a fighter's profile, sorted oldest → newest.

    Each dict: {date, fight_url, result, round, time}
    result is 'win', 'loss', or 'draw'.
    """
    soup = _get_soup(page, profile_url)
    fights = []

    for row in soup.select("tr.b-fight-details__table-row__hover"):
        fight_url = row.get("data-link", "").strip()
        if not fight_url:
            continue

        cells = row.find_all("td")
        if len(cells) < 9:
            continue

        # Result
        flag = row.select_one(".b-flag__text")
        result_raw = flag.get_text(strip=True).lower() if flag else ""
        if result_raw == "win":
            result = "win"
        elif result_raw == "loss":
            result = "loss"
        else:
            result = "draw"

        # Date (col 6, second <p> tag is the date; first is event name)
        event_paras = cells[6].find_all("p")
        fight_date = None
        for p in reversed(event_paras):  # date is usually the last <p>
            d = _parse_date(p.get_text(strip=True))
            if d:
                fight_date = d
                break

        # Round (col 8) — each cell may have <p> tags
        def _cell_text(cell, p_idx=0):
            paras = cell.find_all("p")
            if paras and len(paras) > p_idx:
                return paras[p_idx].get_text(strip=True)
            return cell.get_text(strip=True)

        try:
            round_num = int(_cell_text(cells[8]))
        except (ValueError, IndexError):
            round_num = 0

        # Time in round (col 9)
        time_str = _cell_text(cells[9]) if len(cells) > 9 else "0:00"

        fights.append({
            "date": fight_date,
            "fight_url": fight_url,
            "result": result,
            "round": round_num,
            "time": time_str,
        })

    # Sort chronologically
    fights.sort(key=lambda f: f["date"] or date.min)
    return fights


def scrape_fight_stats(page: Page, fight_url: str, fighter_name: str) -> dict | None:
    """
    Scrape per-fight stats for `fighter_name` from a fight detail page.

    Returns a dict with keys matching STAT_COLS, or None on failure.
    """
    soup = _get_soup(page, fight_url)
    tables = soup.find_all("table")
    if len(tables) < 3:
        return None

    def _get_fighter_idx(table, name: str) -> int | None:
        """Return 0 or 1 for which <p> index matches this fighter."""
        first_data_row = None
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if cells:
                first_data_row = cells
                break
        if not first_data_row:
            return None
        name_cell = first_data_row[0]
        paras = name_cell.find_all("p")
        names_on_page = [p.get_text(strip=True) for p in paras]
        matched = _fuzzy_name_match(name, names_on_page)
        if matched is None:
            return None
        return names_on_page.index(matched)

    # ---- Table 0: Totals ----
    t0 = tables[0]
    idx = _get_fighter_idx(t0, fighter_name)
    if idx is None:
        return None

    data_row = None
    for row in t0.find_all("tr"):
        if row.find("td"):
            data_row = row
            break
    if data_row is None:
        return None

    cells0 = data_row.find_all("td")

    def _val(cell, i: int) -> str:
        paras = cell.find_all("p")
        return paras[i].get_text(strip=True) if len(paras) > i else "0"

    kd = int(_val(cells0[1], idx) or 0)
    sig_landed, sig_att = _parse_of(_val(cells0[2], idx))
    tot_landed, tot_att = _parse_of(_val(cells0[4], idx))
    td_landed, td_att = _parse_of(_val(cells0[5], idx))
    sub_att = int(_val(cells0[7], idx) or 0)
    reversals = int(_val(cells0[8], idx) or 0)
    ctrl = _parse_ctrl(_val(cells0[9], idx))

    # ---- Table 2: Sig strikes by location ----
    t2 = tables[2]
    sig_data_row = None
    for row in t2.find_all("tr"):
        if row.find("td"):
            sig_data_row = row
            break

    head_l = head_a = body_l = body_a = 0
    leg_l = leg_a = dist_l = dist_a = 0
    clinch_l = clinch_a = gnd_l = gnd_a = 0

    if sig_data_row:
        cells2 = sig_data_row.find_all("td")
        if len(cells2) >= 9:
            head_l, head_a = _parse_of(_val(cells2[3], idx))
            body_l, body_a = _parse_of(_val(cells2[4], idx))
            leg_l, leg_a = _parse_of(_val(cells2[5], idx))
            dist_l, dist_a = _parse_of(_val(cells2[6], idx))
            clinch_l, clinch_a = _parse_of(_val(cells2[7], idx))
            gnd_l, gnd_a = _parse_of(_val(cells2[8], idx))

    return {
        "knockdowns": float(kd),
        "sig_strike_atts": float(sig_att),
        "sig_strikes": float(sig_landed),
        "tot_strike_atts": float(tot_att),
        "tot_strikes": float(tot_landed),
        "takedown_atts": float(td_att),
        "takedowns": float(td_landed),
        "submissions": float(sub_att),
        "reversals": float(reversals),
        "ctrl_time": ctrl,
        "head_strike_atts": float(head_a),
        "head_strikes": float(head_l),
        "body_strike_atts": float(body_a),
        "body_strikes": float(body_l),
        "leg_strike_atts": float(leg_a),
        "leg_strikes": float(leg_l),
        "dist_strike_atts": float(dist_a),
        "dist_strikes": float(dist_l),
        "clinch_atts": float(clinch_a),
        "clinchs": float(clinch_l),
        "ground_atts": float(gnd_a),
        "grounds": float(gnd_l),
        # total_fight_time filled in by caller using round/time from fight history
        "total_fight_time": 0.0,
    }


def get_new_fights(
    page: Page,
    profile_url: str,
    fighter_name: str,
    cutoff: date = DATASET_CUTOFF,
) -> list[dict]:
    """
    Return per-fight stats for all fights strictly after `cutoff`.
    Includes the `date` and `result` fields in addition to STAT_COLS.
    """
    history = scrape_fight_history(page, profile_url)
    new = [f for f in history if f["date"] and f["date"] > cutoff]

    result = []
    for fight in new:
        stats = scrape_fight_stats(page, fight["fight_url"], fighter_name)
        if stats is None:
            continue
        stats["total_fight_time"] = _total_fight_seconds(fight["round"], fight["time"])
        stats["date"] = fight["date"]
        stats["result"] = fight["result"]
        result.append(stats)

    return result
