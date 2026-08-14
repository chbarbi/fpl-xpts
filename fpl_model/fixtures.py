""" Fixtures parsing and next GW detection. """

import logging

import pandas as pd

from fpl_model.api import fetch_fixtures
from fpl_model.bootstrap import Bootstrap

logger = logging.getLogger(__name__)

def _parse_fixtures(raw: list, bootstrap: Bootstrap) -> tuple[pd.DataFrame, int | None]:
    """ Takes raw fixture data, returns (fixtures_df, next_gw). Separated from build_fixtures_df so it can be
    tested without hitting the API."""
    raw = pd.DataFrame(raw)

    raw['home_team'] = raw['team_h'].map(bootstrap.team_id_to_name)
    raw['away_team'] = raw['team_a'].map(bootstrap.team_id_to_name)
    raw.rename(
        columns={'team_h_score': 'home_score', 'team_a_score': 'away_score'},
        inplace=True,
    )

    keep = [
        "id", "event", "finished",
        "team_h", "team_a", "home_team", "away_team",
        "home_score", "away_score",
        "team_h_difficulty", "team_a_difficulty",
        "kickoff_time",
    ]
    fixtures = raw[keep].dropna(subset=['event']).copy()
    fixtures['event'] = fixtures['event'].astype(int)

    unplayed = fixtures[~fixtures['finished']]
    if unplayed.empty:
        logger.info('No unfinished fixtures, season complete')
        next_gw = None
    else:
        next_gw = int(unplayed['event'].min())

    n_done = int(fixtures['finished'].sum())
    logger.info(
        "Fixtures: %d total, %d completed. Next GW: %s.",
        len(fixtures), n_done, next_gw,
    )
    return fixtures, next_gw


def build_fixtures_df(bootstrap: Bootstrap) -> tuple[pd.DataFrame, int | None]:
    """Fetch fixtures and parse them."""
    return _parse_fixtures(fetch_fixtures(), bootstrap)