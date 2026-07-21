""" Bootstrap-static parsing: teams, positions, players, and the lookups derived from them. """

import logging
from dataclasses import dataclass

import pandas as pd

from fpl_model.api import fetch_bootstrap

logger = logging.getLogger(__name__)

@dataclass
class Bootstrap:
    """ Parsed bootstrap-static data and lookups derived from it. 
        Bundles the reference tables and id→name maps that the rest of the pipeline needs,
        so functions can accept one `Bootstrap` instead of four loose dicts.
    """
    teams: pd.DataFrame
    positions: pd.DataFrame
    players: pd.DataFrame
    team_id_to_name: dict
    team_name_to_id: dict
    pos_map: dict

def build_bootstrap() -> Bootstrap:
    """Fetch bootstrap-static and parse it into a Bootstrap object."""
    raw = fetch_bootstrap()

    teams = pd.DataFrame(raw['teams'])
    positions = pd.DataFrame(raw['element_types'])

    team_id_to_name = dict(zip(teams['id'], teams['name']))
    team_name_to_id = dict(zip(teams['name'], teams['id']))
    pos_map = dict(zip(positions['id'], positions['singular_name_short']))

    players = pd.DataFrame(raw['elements'])
    players = players[
        (players['can_select']) & (players['minutes'] > 0)
    ].copy()
    players['position'] = players['element_type'].map(pos_map)
    players['team_name'] = players['team'].map(team_id_to_name)
    players['price_m'] = players['now_cost'] / 10

    logger.info(
        "Bootstrap parsed: %d eligible players, %d teams.", len(players), len(teams)
    )
    return Bootstrap(
        teams=teams,
        positions=positions,
        players=players,
        team_id_to_name=team_id_to_name,
        team_name_to_id=team_name_to_id,
        pos_map=pos_map
    )