"""Player Summary: one row per player, aggregating season stats and projections."""

import logging

import pandas as pd

from fpl_model.bootstrap import Bootstrap
from fpl_model.xpts import compute_xpts

logger = logging.getLogger(__name__)

def _pivot_gw_xpts(
        future_predictions: pd.DataFrame,
        gw_cols: list[str],
        next_gw: int | None,
        n_gws: int,
) -> pd.DataFrame:
    """
    Pivot future_predictions to one row per player, with columns for each GW's xPts.

    Double GWs: two fixtures in one GW are summed
    Blank GWs: = 0 (no fixture for that player in that GW)
    Season over: returns just a player_id column.
    """
    if next_gw is None or future_predictions.empty:
        return pd.DataFrame(columns=['player_id'])

    gw_xpts = (
        future_predictions.groupby(['id', 'gameweek'])['xPts']
        .sum() # this is where the DGW gets summed
        .reset_index()
        .rename(columns={'id': 'player_id'})
    )

    gw_pivot = (
        gw_xpts.pivot(index='player_id', columns='gameweek', values='xPts')
        .reset_index()
    )
    gw_pivot.columns.name = None
    gw_pivot.columns = [
        'player_id' if c == 'player_id' else f'xPts_gw{c}'
        for c in gw_pivot.columns
    ]

    # Ensures every GW exists, BGW -> 0
    for col in gw_cols:
        if col not in gw_pivot.columns:
            gw_pivot[col] = 0.0
        else:
            gw_pivot[col] = gw_pivot[col].fillna(0.0).round(2)

    gw_pivot['xPts_next5_sum'] = gw_pivot[gw_cols].sum(axis=1).round(2)
    gw_pivot['xPts_next5_avg'] = gw_pivot[gw_cols].mean(axis=1).round(2)
    return gw_pivot

def build_player_summary(
        bootstrap: Bootstrap,
        performances: pd.DataFrame,
        future_predictions: pd.DataFrame,
        player_rates: pd.DataFrame,
        next_gw: int | None,
        n_gws: int = 5,
) -> pd.DataFrame:
    """
    Build the Player Summary DataFrame: one row per player, aggregating season stats and projections.
    """
    # Season totals
    totals = performances.groupby('player_id').agg(
        MP=('minutes', lambda x: (x > 0).sum()),
        total_minutes=('minutes', 'sum'),
        goals=('goals_scored', 'sum'),
        assists=('assists', 'sum'),
        clean_sheets=('clean_sheets', 'sum'),
        goals_conceded=('goals_conceded', 'sum'),
        saves=('saves', 'sum'),
        yellow_cards=('yellow_cards', 'sum'),
        red_cards=('red_cards', 'sum'),
        bonus=('bonus', 'sum'),
        bps_total=('bps', 'sum'),
        total_xG=('xG', 'sum'),
        total_xA=('xA', 'sum'),
        total_xGC=('xGC', 'sum'),
        total_points=('total_points', 'sum'),
    ).reset_index()

    # Per-90 stats, checking for div by 0 error
    nineties = (totals['total_minutes'] / 90).replace(0, pd.NA)
    totals['90s'] = (totals['total_minutes'] / 90).round(1)
    totals['goals_p90'] = (totals['goals'] / nineties).round(3)
    totals['assists_p90'] = (totals['assists'] / nineties).round(3)
    totals['xG_p90'] = (totals['total_xG'] / nineties).round(3)
    totals['xA_p90'] = (totals['total_xA'] / nineties).round(3)
    totals['xGC_p90'] = (totals['total_xGC'] / nineties).round(3)
    totals['pts_p90'] = (totals['total_points'] / nineties).round(3)

    # Latest rolling form
    form = (
        performances.sort_values('gameweek')
        .groupby('player_id')[['xForm5', 'xForm10']]
        .last()
        .reset_index()
    )

    # Baseline xPts (no adjustment for opponent strength)
    baseline = player_rates.copy()
    baseline['xPts_baseline'] = baseline.apply(
        lambda r: compute_xpts(r, mode='projected')['xPts'], axis=1
    )

    # Per GW projections
    gw_cols = (
        [f'xPts_gw{gw}' for gw in range(next_gw, next_gw + n_gws)]
        if next_gw is not None else []
    )
    gw_pivot = _pivot_gw_xpts(future_predictions, gw_cols, next_gw, n_gws)

    # Base player info
    base = bootstrap.players[[
        'id', 'first_name', 'second_name', 'web_name',
        'position', 'team', 'team_name', 'price_m',
        'selected_by_percent',
        'chance_of_playing_this_round', 'chance_of_playing_next_round',
        'form', 'points_per_game',
    ]].copy()
    base['chance_of_playing_this_round'] = base['chance_of_playing_this_round'].fillna(100.0)
    base['chance_of_playing_next_round'] = base['chance_of_playing_next_round'].fillna(100.0)
    base['points_per_game'] = pd.to_numeric(base['points_per_game'], errors='coerce')

    # Assemble
    summary = (
        base
        .merge(totals, left_on='id', right_on='player_id', how='left')
        .merge(form, left_on='id', right_on='player_id', how='left')
        .merge(baseline[['xPts_baseline']], left_on='id', right_index=True, how='left')
        .merge(gw_pivot, left_on='id', right_on='player_id', how='left')
    )
    summary.drop(columns=[c for c in summary.columns if c == 'player_id'], inplace=True)

    # Value metrics
    ## TO BE FIXED LATER
    next_gw_col = f'xPts_GW{next_gw}'
    if next_gw is not None and next_gw_col in summary.columns:
        summary['xVAPM'] = ((summary[next_gw_col] - 2) / summary['price_m']).round(3)
    else:
        summary['xVAPM'] = pd.NA
    summary['VAPM'] = ((summary['points_per_game'] - 2) / summary['price_m']).round(3)

    # Final column order
    id_cols = ['id', 'first_name', 'second_name', 'web_name', 'position',
               'team_name', 'price_m', 'selected_by_percent',
               'chance_of_playing_this_round', 'chance_of_playing_next_round']
    stat_cols = ['MP', '90s', 'total_minutes', 'goals', 'assists', 'clean_sheets',
                 'goals_conceded', 'saves', 'yellow_cards', 'red_cards',
                 'bonus', 'bps_total', 'total_xG', 'total_xA', 'total_xGC', 'total_points']
    p90_cols = ['goals_p90', 'assists_p90', 'xG_p90', 'xA_p90', 'xGC_p90', 'pts_p90']
    fpl_cols = ['form', 'points_per_game']
    mdl_cols = ['xForm5', 'xForm10', 'xPts_baseline', 'xVAPM', 'VAPM']
    proj_cols = gw_cols + (['xPts_next5_sum', 'xPts_next5_avg'] if gw_cols else [])

    all_cols = id_cols + stat_cols + p90_cols + fpl_cols + mdl_cols + proj_cols
    summary = summary[[c for c in all_cols if c in summary.columns]].round(2)

    logger.info('Player Summary built: %s', summary.shape)
    return summary