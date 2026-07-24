"""Poisson rates computation (Bayesian-shrunk) and opponent strength adjustment."""

import logging

import pandas as pd

from fpl_model.bootstrap import Bootstrap
from fpl_model.config import N_PRIOR

logger = logging.getLogger(__name__)

_RATE_COLS = [
    "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "saves", "yellow_cards", "red_cards",
    "penalties_saved", "penalties_missed",
    "cbi", "recoveries", "tackles", "defensive_contribution",
    "xG", "xA", "xGC", "minutes",
]

def compute_player_rates(
        performances: pd.DataFrame,
        bootstrap: Bootstrap,
        n_prior: int = N_PRIOR,
) -> pd.DataFrame:
    """
    Compute Bayesian-shrunk Poisson rate (λ) for each player for each stat.

    Formula: 
        λ_player = (player_season_total + n_prior * position_avg_rate)
                    / (player_MP + n_prior)

    Early in the season (where MP is low), rates lean on the positional average.
    Later, the player's own data dominates.

    Returns: a DataFrame indexed by player_id.
    """
    totals = performances.groupby('player_id')[_RATE_COLS].sum()
    totals['MP'] = performances[performances['minutes'] > 0].groupby('player_id').size()

    pos_lookup = bootstrap.players.set_index('id')['position']
    totals['position'] = totals.index.map(pos_lookup)

    pos_totals = totals.groupby('position')[_RATE_COLS].sum()
    pos_mp = totals.groupby('position')['MP'].sum()
    pos_rate = pos_totals.div(pos_mp, axis=0)

    rates = pd.DataFrame(index=totals.index)
    rates['position'] = totals['position']
    rates['MP'] = totals['MP']
    rates['avg_minutes'] = (totals['minutes'] / totals['MP']).round(1)

    for col in _RATE_COLS:
        if col == 'minutes':
            continue
        prior = totals['position'].map(pos_rate[col])
        rates[f'lam_{col}'] = (
            (totals[col] + n_prior * prior) / (totals['MP'] + n_prior)
        ).round(4)

    logger.info('Player rates computed for %d players.', len(rates))
    return rates

def build_team_strength_lookup(
    team_strength: pd.DataFrame,
    bootstrap: Bootstrap
) -> pd.DataFrame:
    """
    Merge team strength scores onto FPL teams, indexed by FPL team id.
    Warns if any team is missing from the strength data.
    """
    teams = bootstrap.teams[['id', 'name']].copy()
    teams = teams.merge(
        team_strength[['fpl_name', 'home_attack', 'home_defense', 'away_attack', 'away_defense']],
        left_on='name', right_on='fpl_name', how='left'
    )
    missing = teams[teams['home_attack'].isna()]['name'].tolist()
    if missing:
        logger.warning("No strength data for: %s. Add entries to FBREF_TO_FPL or check big5_xG_data.csv.", missing)
    return teams.set_index('id')


def adjust_rates_for_opponent(
    rates_row: pd.Series,
    opp_team_id: int,
    is_home: bool,
    team_lookup: pd.DataFrame
) -> pd.Series:
    """
    Scale a player's Poisson rates for a specific opponent and home/away context.

    When the player is at home, the opponent uses their away ratings.
    When the player is away, the opponent uses their home ratings.

    Attacking rates (goals, assists, xG, xA) scaled by:
        (100 - opp_defense) / 50    ← weaker opp defence = easier to score

    Defensive rates (goals_conceded, xGC) scaled by:
        opp_attack / 50             ← stronger opp attack = more goals conceded
    """
    row = rates_row.copy()

    if opp_team_id not in team_lookup.index:
        logger.warning("opp_team_id %d not in team_lookup, no adjustment applied.", opp_team_id)
        return row

    opp = team_lookup.loc[opp_team_id]

    if opp[["home_attack", "away_attack", "home_defense", "away_defense"]].isna().any():
        logger.warning("Strength data for opp_team_id %d is incomplete — no adjustment applied.", opp_team_id)
        return row

    # Opponent uses away context when player is home, and vice versa
    opp_att = opp['away_attack']   if is_home else opp['home_attack']
    opp_def = opp['away_defense']  if is_home else opp['home_defense']

    att_mult = (100 - opp_def) / 50
    def_mult = opp_att / 50

    for col in ['lam_goals_scored', 'lam_assists', 'lam_xG', 'lam_xA']:
        if col in row.index:
            row[col] = max(0.0, float(row[col]) * att_mult)

    for col in ['lam_goals_conceded', 'lam_xGC']:
        if col in row.index:
            row[col] = max(0.0, float(row[col]) * def_mult)

    return row