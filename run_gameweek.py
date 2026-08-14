"""
End-to-end pipeline entry point: fetch -> model -> project -> export.
"""

import logging

import pandas as pd

from fpl_model.config import BIG5_XG_PATH, STRENGTH_SEASON, OUTPUT_DIR
from fpl_model.logging_setup import setup_logging
from fpl_model.team_strength import compute_team_strength
from fpl_model.bootstrap import build_bootstrap
from fpl_model.fixtures import build_fixtures_df
from fpl_model.performances import build_performances_df
from fpl_model.rates import compute_player_rates, build_team_strength_lookup
from fpl_model.xpts import compute_xpts
from fpl_model.predictions import build_future_predictions
from fpl_model.summary import build_player_summary
from fpl_model.database import init_db, save_predictions

logger = logging.getLogger(__name__)

N_GWS = 5


def _apply_historic_xpts(performances: pd.DataFrame) -> pd.DataFrame:
    """
    Historic xPts per performance row + rolling form.

    Summary depends on the xForm5/xForm10 columns this produces, so it must run before build_player_summary.
    """
    if performances.empty:
        logger.info("No performances to score, skipping historic xPts and form.")
        return performances

    performances = performances.sort_values(['player_id', 'gameweek']).reset_index(drop=True)

    hist = performances.apply(lambda r: compute_xpts(r, mode='historic'), axis=1)
    performances = pd.concat(
        [performances, pd.DataFrame(hist.tolist(), index=performances.index)], axis=1
    )

    for window, col in [(5, 'xForm5'), (10, 'xForm10')]:
        performances[col] = (
            performances.groupby('player_id')['xPts']
            .transform(lambda x: x.rolling(window, min_periods=1).mean().round(2))
        )
    return performances

def main() -> None:
    setup_logging()
    logger.info("=== Pipeline start ===")

    init_db()

    # Reference data
    team_strength = compute_team_strength(BIG5_XG_PATH, STRENGTH_SEASON)
    bootstrap = build_bootstrap()
    fixtures, next_gw = build_fixtures_df(bootstrap)
    team_lookup = build_team_strength_lookup(team_strength, bootstrap)

    # Historic model + model inputs
    performances = build_performances_df(bootstrap, fixtures)

    # If running during pre-season (no matches played -> no data)
    if performances.empty:
        logger.warning('No completed performances yet, writing bootstrap-only outputs.')
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        bootstrap.players.to_csv(OUTPUT_DIR / "player_summary.csv", index=False)
        logger.info("=== Pipeline complete (pre-season): %d players ===", len(bootstrap.players))
        return

    performances = _apply_historic_xpts(performances)
    player_rates = compute_player_rates(performances, bootstrap)

    # Projections
    future_predictions = build_future_predictions(
        player_rates, bootstrap, fixtures, team_lookup, next_gw, n_gws=N_GWS
    )
    player_summary = build_player_summary(
        bootstrap, performances, future_predictions, player_rates, next_gw, n_gws=N_GWS
    )

    # Saving snapshot of predictions to SQLite database
    save_predictions(future_predictions, next_gw)

    # Persist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    performances.to_csv(OUTPUT_DIR / 'performances.csv', index=False)
    future_predictions.to_csv(OUTPUT_DIR / 'future_predictions.csv', index=False)
    player_summary.to_csv(OUTPUT_DIR / 'player_summary.csv', index=False)

    gw_label = f'GW{next_gw}' if next_gw is not None else 'season_complete'
    logger.info('Outputs written to %s (label: %s)', OUTPUT_DIR, gw_label)
    logger.info('=== Pipeline complete: %d players ===', len(player_summary))

if __name__ == '__main__':
    main()