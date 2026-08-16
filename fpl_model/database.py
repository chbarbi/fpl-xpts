"""SQLite setup for prediction-outcome validation."""

import logging
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from fpl_model.config import COLD_START_GWS, DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    run_timestamp   TEXT    NOT NULL,
    gameweek        INTEGER NOT NULL,
    fixture_id      INTEGER NOT NULL,
    player_id       INTEGER NOT NULL,
    web_name        TEXT,
    position        TEXT,
    team_name       TEXT,
    opponent_team   TEXT,
    is_home         INTEGER,
    price_m         REAL,
    xpts            REAL,
    ep_next         REAL,
    is_cold_start   INTEGER,
    PRIMARY KEY (run_timestamp, fixture_id, player_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
    gameweek        INTEGER NOT NULL,
    player_id       INTEGER NOT NULL,
    minutes         INTEGER,
    total_points    INTEGER,
    PRIMARY KEY (gameweek, player_id)
);
"""

logger = logging.getLogger(__name__)

def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def init_db() -> None:
    """Create tables if they don't exist. Safe to call every run."""
    with _connect() as conn:
        conn.executescript(_SCHEMA) # the CREATE table statements above as a string.
    logger.info('Database initialized at %s', DB_PATH)

def save_predictions(future_predictions: pd.DataFrame, next_gw: int | None) -> None:
    """Append a timestamped snapshot of predictions."""
    if future_predictions.empty:
        logger.info('No predictions to save.')
        return

    df = future_predictions.copy()
    df['run_timestamp'] = datetime.now(timezone.utc).isoformat()
    df['is_cold_start'] = (df['gameweek'] <= COLD_START_GWS).astype(int)

    cols = ['run_timestamp', 'gameweek', 'fixture_id', 'player_id', 'web_name',
            'position', 'team_name', 'opponent_team', 'is_home', 'price_m',
            'xPts', 'ep_next', 'is_cold_start']
    df = df[[c for c in cols if c in df.columns]].rename(columns={'xPts': 'xpts'})

    with _connect() as conn:
        df.to_sql('predictions', conn, if_exists='append', index=False)
    logger.info('Saved %d prediction rows for GW%s.', len(df), next_gw)

def save_outcomes(outcomes: pd.DataFrame) -> None:
    """Insert actual gameweek results, ignoring dupes.
    
    Outcomes are inserted after the GW is complete, so we don't expect to overwrite any existing rows.
    INSERT or IGNORE skips rows where an entry already exists for that (gameweek, player_id) pair.
    """
    if outcomes.empty:
        logger.info('No outcomes to save.')
        return

    rows = outcomes[['gameweek', 'player_id', 'minutes', 'total_points']]
    with _connect() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO outcomes (gameweek, player_id, minutes, total_points) "
            "VALUES (?, ?, ?, ?)",
            rows.itertuples(index=False, name=None)
        )
    logger.info('Saved outcomes for %d player-GW rows.', len(rows))