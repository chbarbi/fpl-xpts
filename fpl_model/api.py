""" All API calls to the FPL API are made here. """

import logging
import time

import requests

from fpl_model.config import API_DELAY, FPL_BASE

logger = logging.getLogger(__name__)

_session = requests.Session()

def _fetch(url: str, retries: int = 3) -> dict | list | None:
    """
    GET a URL and return parsed JSON.

    - Retries on network/timeout errors with exponential backoff.
    - Does NOT retry on HTTP errors (404, 403, etc.) — logs and returns None.
    - Returns None on complete failure so callers can handle gracefully.
    """

    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error fetching %s: %s", url, e)
            return None
        except requests.exceptions.RequestException as e:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, e)
            if attempt < retries:
                time.sleep(2 ** attempt)
    logger.error("All %d attempts failed for %s", retries, url)
    return None

def fetch_bootstrap() -> dict:
    """Fetch the bootstrap-static endpoint (teams, players, positions, events)."""
    url = f"{FPL_BASE}/bootstrap-static/"
    logger.info("Fetching bootstrap: %s", url)
    data = _fetch(url)
    if data is None:
        raise RuntimeError("Failed to fetch bootstrap. Check your internet connection.")
    return data


def fetch_fixtures() -> list:
    """Fetch all season fixtures. Returns raw dict."""
    url = f"{FPL_BASE}/fixtures/"
    logger.info("Fetching fixtures: %s", url)
    data = _fetch(url)
    if data is None:
        raise RuntimeError("Failed to fetch fixtures. Check your internet connection.")
    return data

def fetch_element_summary(player_id: int) -> dict | None:
    """
    Fetch per-match history and upcoming fixtures for one player.
    Returns dict with keys 'history' and 'fixtures', or None on failure.
    Applies a short delay to avoid hammering the API.
    """
    url  = f"{FPL_BASE}/element-summary/{player_id}/"
    data = _fetch(url)
    if data is None:
        logger.warning("Could not fetch element-summary for player %d.", player_id)
    time.sleep(API_DELAY)
    return data


def fetch_gw_live(gw: int) -> dict | None:
    """
    Fetch post-match stats for a completed gameweek.
    Returns dict with key 'elements', each containing a 'stats' sub-dict.
    """
    url  = f"{FPL_BASE}/event/{gw}/live/"
    logger.info("Fetching live data for gameweek %d: %s", gw, url)
    data = _fetch(url)
    if data is None:
        logger.warning("Could not fetch GW %d live data.", gw)
    return data