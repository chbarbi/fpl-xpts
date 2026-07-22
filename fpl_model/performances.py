""" Build the Performances DataFrame: one row per (player, completed fixture). """

import logging

import pandas as pd

from fpl_model.api import fetch_element_summary, fetch_gw_live
from fpl_model.bootstrap import Bootstrap

logger = logging.getLogger(__name__)

"""
    Columns pulled from element-summary history, with their JSON keys.
    Declared once here rather than inline so the extraction loop stays readable
    and the schema is documented in one place.
"""

_SUMMARY_FIELDS = {
    "goals_scored": 0, "assists": 0, "clean_sheets": 0, "goals_conceded": 0,
    "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
    "yellow_cards": 0, "red_cards": 0, "saves": 0, "bonus": 0, "bps": 0,
    "minutes": 0, "total_points": 0,
}
_SUMMARY_FLOATS = {
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists",
    "expected_goals_conceded", "expected_goal_involvements",
}

def _fetch_summary_rows(player_ids: list[int]) -> pd.DataFrame:
    """Step 1: per-player element summary history -> one row per (player, fixture)."""
    rows = []
    for i, pid in enumerate(player_ids):
        if i % 50 == 0:
            logger.info('element-summary: %d/%d', i, len(player_ids))

        data = fetch_element_summary(pid)
        if not data or not data.get('history'):
            continue

        for m in data['history']:
            row = {
                'player_id': pid,
                'fixture_id': m.get('fixture'),
                'gameweek': m.get('round'),
                'opponent_team_id': m.get('opponent_team'),
                'was_home': m.get('was_home'),
            }
            for key, default in _SUMMARY_FIELDS.items():
                row[key] = m.get(key, default)
            for key in _SUMMARY_FLOATS:
                row[key] = float(m.get(key, 0))
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info('element-summary: %d player-fixture rows.', len(df))
    return df

def _fetch_live_rows(completed_gws: list[int]) -> pd.DataFrame:
    """Step 2: per-GW live endpoint -> defensive stats (CBI, tackles, recoveries)."""
    rows = []
    for gw in completed_gws:
        data = fetch_gw_live(gw)
        if not data:
            logger.warning('Skipping GW %d, no live data returned', gw)
            continue
        for elem in data['elements']:
            s = elem.get('stats', {})
            rows.append({
                "player_id": elem["id"],
                "gameweek": gw,
                "cbi": s.get("clearances_blocks_interceptions", 0),
                "recoveries": s.get("recoveries", 0),
                "tackles": s.get("tackles", 0),
                "defensive_contribution": s.get("defensive_contribution", 0),
            })

    df = pd.DataFrame(rows)
    logger.info('GW live: %d player-GW rows.', len(df))
    return df

def build_performances_df(bootstrap: Bootstrap, fixtures: pd.DataFrame) -> pd.DataFrame:
    """
    Builds performances dataframe, one row per (player,fixture) for all completed gameweeks.

    Step 1: element-summary per player  -> rich per-player stats
    Step 2: event/live per GW           -> CBI, recoveries, tackles
    Step 3: merge on (player_id, gameweek), enrich with metadata
    """
    players = bootstrap.players

    df_summary = _fetch_summary_rows(players['id'].tolist())
    if df_summary.empty:
        logger.warning('No completed player history found, returning empty performances.')
        return df_summary
    
    completed_gws = sorted(fixtures.loc[fixtures['fixtures'], 'event'].unique())
    df_live = _fetch_live_rows(completed_gws)

    """
    Step 3: merge. element-summary has 2 fixture rows per double-GW;
    live has 1 GW-aggregated row, so defensive stats attach to both
    fixtures via the left merge — acceptable approximation for now.
    """
    df = df_summary.merge(df_live, on=['player_id', 'gameweek'], how='left')

    # Rename the FPL API's expected_* columns to short names the model uses.
    df = df.rename(columns={
        'expected_goals': 'xG', 'expected_assists': 'xA',
        'expected_goals_conceded': 'xGC', 'expected_goal_involvements': 'xGI',
    })

    meta = players[[
        'id', 'first_name', 'second_name', 'web_name',
        'positions', 'team_name', 'team',
    ]]
    df = df.merge(meta, left_on='player_id', right_on='id', how='left')
    df['opponent_team'] = df['opponent_team_id'].map(bootstrap.team_id_to_name)

    df = df[df['minutes'] > 0].copy()

    for col in ['cbi', 'recoveries', 'tackles', 'defensive_contribution']:
        df[col] = df[col].fillna(0).astype(int)

    col_order = [
        'player_id', 'first_name', 'second_name', 'web_name',
        'position', 'team_name', 'gameweek',
        'fixture_id', 'opponent_team', 'was_home',
        'minutes', 'goals_scored', 'assists', 'clean_sheets',
        'goals_conceded', 'own_goals', 'penalties_saved', 'penalties_missed',
        'yellow_cards', 'red_cards', 'saves',
        'cbi', 'recoveries', 'tackles', 'defensive_contribution',
        'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index',
        'xG', 'xA', 'xGC', 'xGI',
        'total_points',
    ]
    df = df[col_order].reset_index(drop=True)

    logger.info('Performances built: %s', df.shape)
    return df