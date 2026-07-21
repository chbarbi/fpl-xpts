""" Team Strength scores from Big-5 league xG data. """

import logging

import pandas as pd
import numpy as np

from fpl_model.config import FBREF_TO_FPL

logger = logging.getLogger(__name__)

def compute_team_strength(big5_xg_path, season: str) -> pd.DataFrame:
    """
    Load top-5 league xG data from 2017-18 to 2025-26 and compute log-ratio strength scores,
    return a one-row-per-team DataFrame for the given PL season.

    Log-ratio formula (centred at 50, league average = 50):
        attack_score  = 50 + 50 * log(team_xG  / mean_xG_across_all_data)
        defense_score = 50 + 50 * log(team_xGA / mean_xGA_across_all_data)
            
        then INVERTED: defense_score = 100 - raw_score
        so that higher = better defence (fewer goals conceded)

    Scores are unbounded but in practice sit between ~20 and ~80.
    50 = exactly league average. Above 50 = better than average.

    Args:
        big5_xg_path: path to CSV with top-5 league xG data
        season: e.g. '2025-26'
    """
    df = pd.read_csv(big5_xg_path)

    # Per-league baselines: each row gets its OWN league's mean xG/xGA.
    # transform() returns a Series aligned to df's rows (same length),
    # so every team is scored against the league it actually plays in.
    league_mean_xG  = df.groupby('League')['xG'].transform('mean')
    league_mean_xGA = df.groupby('League')['xGA'].transform('mean')

    df['attack_score']  = 50 + 50 * np.log(df['xG']  / league_mean_xG)
    df['defense_score'] = 100 - (50 + 50 * np.log(df['xGA'] / league_mean_xGA))
    #                     ^--- invert: higher xGA = weaker defence, we flip it
    
    pl = df[(df['League'] == 'Premier League') & (df['Season'] == season)].copy()
    if pl.empty:
        available = df['Season'].unique().tolist()
        raise ValueError(f"No Premier League data for season {season}. Available seasons: {available}")
    
    # Extract Home / Away from team_id (format: "Arsenal_Home_202526")
    pl['home_away'] = pl['team_id'].str.extract(r'_(Home|Away)_')

    # Pivot: one row per team, separate home/away columns
    pl_pivot = pl.pivot_table(
        index='Squad',
        columns='home_away',
        values=['attack_score', 'defense_score']
    )

    pl_pivot.columns = [
        f'{ha.lower()}_{metric.replace("_score", "")}'
        for metric, ha in pl_pivot.columns
    ]

    pl_pivot = pl_pivot.round(2).reset_index()
    pl_pivot.rename(columns={"Squad": "squad"}, inplace=True)

    # Mapping FBREF team names to FPL team names
    pl_pivot['fpl_name'] = pl_pivot['squad'].replace(FBREF_TO_FPL)

    logger.info("Team strength computed: %d PL teams for %s.", len(pl_pivot), season)
    return pl_pivot