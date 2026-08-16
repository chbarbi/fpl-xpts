"""Parses FPL API live gameweek data into per-player outcome rows."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

def parse_gw_outcomes(live_data: dict, gameweek: int) -> pd.DataFrame:
    """Turn an event/{gw}/live response into per-player outcome rows."""
    if not live_data or 'elements' not in live_data:
        logger.info('No valid live data for GW %d.', gameweek)
        return pd.DataFrame(columns=['gameweek', 'player_id', 'minutes', 'total_points'])

    rows = []
    for elem in live_data['elements']:
        stats = elem.get('stats', {})
        rows.append({
            'gameweek': gameweek,
            'player_id': elem['id'],
            'minutes': stats.get('minutes', 0),
            'total_points': stats.get('total_points', 0),
        })
    return pd.DataFrame(rows)