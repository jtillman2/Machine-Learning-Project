"""
UFC Fight Predictor — FastAPI backend.

Endpoints:
  GET  /              → frontend HTML
  GET  /search?q=...  → fighter name autocomplete (from local dataset)
  POST /predict       → { fighter1, fighter2 } → win probabilities

Architecture:
  Playwright's sync API cannot run inside an asyncio event loop.
  A single dedicated worker thread owns the browser for its lifetime.
  Prediction requests are sent to it via a threading.Queue, so the
  async endpoint never touches Playwright directly.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import random as _random
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from predict import predict_fight, AVAILABLE_MODELS
from scraper.data_loader import load_data, _normalize_weight_class

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "static"

# ---------------------------------------------------------------------------
# Browser worker thread
# ---------------------------------------------------------------------------
# The worker owns the Playwright browser. Callers put a (future, f1, f2) tuple
# onto _job_queue; the worker runs predict_fight and resolves the future.

_job_queue: queue.Queue = queue.Queue()
_worker_thread: threading.Thread | None = None


def _browser_worker() -> None:
    """Runs in a dedicated non-asyncio thread. Owns the Playwright browser.
    Each job is a (future, browser_fn) pair; browser_fn(browser) -> result."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            while True:
                job = _job_queue.get()
                if job is None:       # shutdown sentinel
                    break
                cf_future, browser_fn = job
                try:
                    result = browser_fn(browser)
                    cf_future.set_result(result)
                except Exception as exc:
                    cf_future.set_exception(exc)
        finally:
            browser.close()


def _start_worker() -> None:
    global _worker_thread
    _worker_thread = threading.Thread(target=_browser_worker, daemon=True)
    _worker_thread.start()


def _stop_worker() -> None:
    _job_queue.put(None)          # send shutdown sentinel
    if _worker_thread:
        _worker_thread.join(timeout=10)


async def _run_in_browser(browser_fn) -> Any:
    """Submit any browser job to the worker thread and await the result."""
    cf_future: concurrent.futures.Future = concurrent.futures.Future()
    _job_queue.put((cf_future, browser_fn))
    return await asyncio.wrap_future(cf_future)


async def _run_prediction(f1: str, f2: str, model_key: str = "random_forest") -> dict:
    return await _run_in_browser(lambda b: predict_fight(f1, f2, browser=b, model_key=model_key))


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_data()       # pre-load dataset into memory
    _start_worker()   # start browser in background thread
    yield
    _stop_worker()


app = FastAPI(title="UFC Fight Predictor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Fighter image
# ---------------------------------------------------------------------------

_image_cache: dict[str, str | None] = {}
_image_cache_lock = threading.Lock()

_ESPN_SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


async def _fetch_espn_image(name: str) -> str | None:
    """
    Search ESPN's internal API for an MMA fighter and return their CDN headshot URL.
    The search result includes the image URL directly — no ID construction needed.
    """
    import httpx
    async with httpx.AsyncClient(headers=_ESPN_SEARCH_HEADERS, timeout=8.0) as client:
        r = await client.get(
            "https://site.web.api.espn.com/apis/search/v2",
            params={"query": name, "limit": 5, "region": "us",
                    "lang": "en", "platform": "web"},
        )
        if r.status_code != 200:
            return None
        for section in r.json().get("results", []):
            for item in section.get("contents", []):
                if item.get("sport") == "mma" or item.get("description") == "MMA":
                    img = item.get("image", {}).get("default")
                    if img:
                        return img
    return None


@app.get("/fighter-image")
async def get_fighter_image(name: str = "") -> dict:
    if not name.strip():
        return {"image_url": None}
    key = name.strip().lower()
    with _image_cache_lock:
        if key in _image_cache:
            return {"image_url": _image_cache[key]}
    url = None
    try:
        url = await _fetch_espn_image(name.strip())
    except Exception:
        pass
    with _image_cache_lock:
        _image_cache[key] = url
    return {"image_url": url}


# ---------------------------------------------------------------------------
# Fighter search (autocomplete from local dataset)
# ---------------------------------------------------------------------------

_fighter_index: list[dict] | None = None
_fighter_index_lock = threading.Lock()


def _build_fighter_index() -> list[dict]:
    fights_df, fighters_df, _ = load_data()

    women_ids: set[int] = (
        set(fights_df.loc[fights_df["weight_class"].str.contains("Women", na=False), "f1_id"])
        | set(fights_df.loc[fights_df["weight_class"].str.contains("Women", na=False), "f2_id"])
    )
    sorted_f = fights_df.sort_values("event_date")
    last_wc_f1 = sorted_f.groupby("f1_id")["weight_class"].last().to_dict()
    last_wc_f2 = sorted_f.groupby("f2_id")["weight_class"].last().to_dict()
    last_wc = {**last_wc_f2, **last_wc_f1}

    # Count fights per fighter from fights_df — same source predict.py uses
    fight_counts = (
        fights_df.groupby("f1_id").size()
        .add(fights_df.groupby("f2_id").size(), fill_value=0)
        .astype(int)
    )

    out = []
    for _, row in fighters_df.iterrows():
        fid = int(row["fighter_id"])
        if int(fight_counts.get(fid, 0)) < 5:
            continue
        w = int(row.get("fighter_wins", 0) or 0)
        l = int(row.get("fighter_losses", 0) or 0)
        d = int(row.get("fighter_draws", 0) or 0)
        gender = "female" if fid in women_ids else "male"
        wc_raw = last_wc.get(fid) or ""
        weight_class = _normalize_weight_class(str(wc_raw)) if wc_raw else None
        out.append({
            "name": row["fighter_name"],
            "record": f"{w}-{l}-{d}",
            "gender": gender,
            "weight_class": weight_class,
        })
    return out


def _get_fighter_index() -> list[dict]:
    global _fighter_index
    if _fighter_index is None:
        with _fighter_index_lock:
            if _fighter_index is None:
                _fighter_index = _build_fighter_index()
    return _fighter_index


@app.get("/search")
def search_fighters(q: str = "") -> list[dict]:
    """
    Return up to 10 fighters matching the query, ranked by relevance:
      0 — exact full name match
      1 — name starts with query
      2 — any name token starts with query
      3 — name contains query anywhere
    """
    if len(q) < 2:
        return []
    q_lower = q.strip().lower()
    tokens = q_lower.split()

    def _rank(f: dict) -> int:
        name = f["name"].lower()
        if name == q_lower:
            return 0
        if name.startswith(q_lower):
            return 1
        if any(t.startswith(q_lower) or name.startswith(t) for t in tokens):
            return 2
        if q_lower in name:
            return 3
        return 99

    ranked = [(f, _rank(f)) for f in _get_fighter_index() if q_lower in f["name"].lower()]
    ranked.sort(key=lambda x: (x[1], x[0]["name"]))
    return [f for f, _ in ranked[:10]]


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------

@app.get("/models")
def list_models() -> list[dict]:
    return [
        {"key": k, "label": v["label"], "accuracy": v["accuracy"]}
        for k, v in AVAILABLE_MODELS.items()
    ]


@app.get("/random-fighter")
def get_random_fighter(gender: str = "", weight_class: str = "") -> dict:
    index = _get_fighter_index()
    pool = index
    if gender:
        pool = [f for f in pool if f.get("gender") == gender]
    if weight_class and weight_class != "any":
        pool = [f for f in pool if f.get("weight_class") == weight_class]
    if not pool:
        raise HTTPException(status_code=404, detail="No fighters found matching those criteria.")
    f = _random.choice(pool)
    return {"name": f["name"], "record": f["record"], "gender": f.get("gender"), "weight_class": f.get("weight_class")}


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    fighter1: str
    fighter2: str
    model_key: str = "random_forest"


class PredictResponse(BaseModel):
    f1_name: str
    f2_name: str
    f1_win_prob: float
    f2_win_prob: float
    f1_experience: int
    f2_experience: int
    f1_record: str
    f2_record: str
    model_key: str
    model_label: str


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(req: PredictRequest) -> Any:
    if not req.fighter1.strip() or not req.fighter2.strip():
        raise HTTPException(status_code=422, detail="Both fighter names are required.")
    if req.fighter1.strip().lower() == req.fighter2.strip().lower():
        raise HTTPException(status_code=422, detail="Please enter two different fighters.")
    if req.model_key not in AVAILABLE_MODELS:
        raise HTTPException(status_code=422, detail=f"Unknown model '{req.model_key}'.")

    try:
        result = await _run_prediction(req.fighter1.strip(), req.fighter2.strip(), req.model_key)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
