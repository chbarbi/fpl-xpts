"""Tests for fixtures parsing and next-gameweek detection."""

import pandas as pd

from fpl_model.bootstrap import Bootstrap
from fpl_model.fixtures import _parse_fixtures


def _make_bootstrap():
    """Minimal Bootstrap — _parse_fixtures only needs team_id_to_name."""
    teams = pd.DataFrame({'id': [1, 2], 'name': ['Team A', 'Team B']})
    return Bootstrap(
        teams=teams, positions=pd.DataFrame(), players=pd.DataFrame(),
        team_id_to_name={1: 'Team A', 2: 'Team B'},
        team_name_to_id={'Team A': 1, 'Team B': 2},
        pos_map={},
    )


def _raw_fixture(fixture_id, event, finished):
    """One raw fixture dict, matching the FPL API's fixture shape (minimal)."""
    return {
        'id': fixture_id, 'event': event, 'finished': finished,
        'team_h': 1, 'team_a': 2,
        'team_h_score': 1 if finished else None,
        'team_a_score': 0 if finished else None,
        'team_h_difficulty': 3, 'team_a_difficulty': 3,
        'kickoff_time': '2026-08-15T14:00:00Z',
    }


def test_next_gw_is_earliest_unfinished():
    """With GW1 finished and GW2, GW3 unfinished, next_gw should be 2."""
    raw = [
        _raw_fixture(1, 1, True),
        _raw_fixture(2, 2, False),
        _raw_fixture(3, 3, False),
    ]
    _, next_gw = _parse_fixtures(raw, _make_bootstrap())
    assert next_gw == 2


def test_season_complete_returns_none():
    """With every fixture finished, next_gw should be None."""
    raw = [
        _raw_fixture(1, 1, True),
        _raw_fixture(2, 2, True),
    ]
    _, next_gw = _parse_fixtures(raw, _make_bootstrap())
    assert next_gw is None


def test_all_unfinished_returns_first():
    """Pre-season: nothing finished. next_gw should be the earliest GW."""
    raw = [
        _raw_fixture(1, 1, False),
        _raw_fixture(2, 2, False),
    ]
    _, next_gw = _parse_fixtures(raw, _make_bootstrap())
    assert next_gw == 1