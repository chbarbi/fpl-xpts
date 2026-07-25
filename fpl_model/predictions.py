"""Future predictions: per-fixture projected xPts for upcoming gameweeks."""

import logging

import pandas as pd

from fpl_model.bootstrap import Bootstrap
from fpl_model.rates import adjust_rates_for_opponent
from fpl_model.xpts import compute_xpts

logger = logging.getLogger(__name__)

FUTURE_PRED_COLUMNS = [
    'id', 'web_name', 'first_name', 'second_name', 'position',
    'team_name', 'price_m', 'gameweek', 'fixture_id',
    'opponent_team', 'is_home',
    'avg_minutes', 'MP', 'prob_play_60', 'chance_of_playing_next_round',
    'xPts',
    'xPts_mins', 'xPts_cs', 'xPts_gc', 'xPts_goals', 'xPts_assists',
    'xPts_saves', 'xPts_yellows', 'xPts_reds',
    'xPts_pen_save', 'xPts_pen_miss', 'xPts_defcon',
    'lam_goals_scored', 'lam_assists', 'lam_goals_conceded',
    'lam_saves', 'lam_yellow_cards', 'lam_red_cards',
    'lam_xG', 'lam_xA', 'lam_xGC',
]

def _build_fixture_long(future_fx: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, fixture): each fixture yields home and away row."""
    home = future_fx[['id', 'event', 'team_h', 'team_a']].rename(
        columns={'team_h': 'team_id', 'team_a': 'opp_team_id'}
    )
    home['is_home'] = True

    away = future_fx[['id', 'event', 'team_h', 'team_a']].rename(
        columns={'team_a': 'team_id', 'team_h': 'opp_team_id'}
    )
    away['is_home'] = False

    long = pd.concat([home, away], ignore_index=True)
    return long.rename(columns={'id': 'fixture_id', 'event': 'gameweek'})

def build_future_predictions(
        player_rates: pd.DataFrame,
        bootstrap: Bootstrap,
        fixtures: pd.DataFrame,
        team_lookup: pd.DataFrame,
        next_gw: int | None,
        n_gws: int = 5,
) -> pd.DataFrame:
    """
    Builds Future Predictions DataFrame: one row per (player, fixture) for the next n_gws.
    
    Returns empty (correctly-columned) DataFrame when season is complete (next_gw is None)
    so downstream code can proceed uniformly.
    """
    if next_gw is None:
        logger.info('No future predictions: season complete.')
        return pd.DataFrame(columns=FUTURE_PRED_COLUMNS)

    gw_range = range(next_gw, next_gw + n_gws)
    future_fx = fixtures[
        fixtures['event'].isin(gw_range) & ~fixtures['finished']
    ].copy()

    if future_fx.empty:
        logger.info('No future predictions: no upcoming fixtures in next %d gameweeks.', list(gw_range))
        return pd.DataFrame(columns=FUTURE_PRED_COLUMNS)

    fx_long = _build_fixture_long(future_fx)

    player_meta = bootstrap.players[[
        'id', 'web_name', 'first_name', 'second_name',
        'position', 'team', 'team_name', 'price_m',
        'chance_of_playing_next_round',
    ]].copy()
    player_meta['chance_of_playing_next_round'] = (
        player_meta['chance_of_playing_next_round'].fillna(100.0)
    )

    pred_df = player_meta.merge(fx_long, left_on='team', right_on='team_id', how='inner')
    pred_df = pred_df.merge(
        player_rates.drop(columns=['position']),
        left_on='id', right_index=True, how='left',
    )
    pred_df['opponent_team'] = pred_df['opp_team_id'].map(bootstrap.team_id_to_name)

    # Per-fixture opponent adjusted xPts
    xpts_results = []
    for _, row in pred_df.iterrows():
        adj_row = adjust_rates_for_opponent(
            row,
            opp_team_id=int(row['opp_team_id']),
            is_home=bool(row['is_home']),
            team_lookup=team_lookup,
        )
        xpts_results.append(compute_xpts(adj_row, mode='projected'))

    pred_df = pd.concat([pred_df, pd.DataFrame(xpts_results, index=pred_df.index)], axis=1)

    # Scale only the next GW by availability; later GWs are too far out for
    # today's injury flags to be meaningful.
    is_next = pred_df['gameweek'] == next_gw
    avail = pred_df['chance_of_playing_next_round'] / 100.0
    pred_df.loc[is_next, 'xPts'] = (pred_df.loc[is_next, 'xPts'] * avail[is_next]).round(2)

    pred_df = pred_df[[c for c in FUTURE_PRED_COLUMNS if c in pred_df.columns]]
    pred_df = pred_df.sort_values(['id', 'gameweek', 'fixture_id']).reset_index(drop=True)

    logger.info("Future predictions built: %s", pred_df.shape)
    return pred_df