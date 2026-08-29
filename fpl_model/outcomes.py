"""Parses FPL API live gameweek data into per-player outcome rows."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

def parse_gw_outcomes(live_data: dict, gameweek: int) -> pd.DataFrame:
    """Turn an event/{gw}/live response into per-player outcome rows."""
    cols = ['gameweek', 'player_id', 'minutes', 'total_points', 'goals_scored',
            'assists', 'clean_sheets', 'goals_conceded', 'saves', 'bonus',
            'defensive_contribution']
    if not live_data or 'elements' not in live_data:
        logger.info('No valid live data for GW %d.', gameweek)
        return pd.DataFrame(columns=cols)
    
    rows = []
    for elem in live_data['elements']:
        stats = elem.get('stats', {})
        rows.append({
            'gameweek': gameweek,
            'player_id': elem['id'],
            'minutes': stats.get('minutes', 0),
            'total_points': stats.get('total_points', 0),
            'goals_scored': stats.get('goals_scored', 0),
            'assists': stats.get('assists', 0),
            'clean_sheets': stats.get('clean_sheets', 0),
            'goals_conceded': stats.get('goals_conceded', 0),
            'saves': stats.get('saves', 0),
            'bonus': stats.get('bonus', 0),
            'defensive_contribution': stats.get('defensive_contribution', 0),
        })
    return pd.DataFrame(rows)