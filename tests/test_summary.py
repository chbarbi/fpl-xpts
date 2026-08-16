"""Tests for player summary aggregation in double/blank gameweeks."""

import pandas as pd

from fpl_model.summary import _pivot_gw_xpts


def _pred_row(player_id, gameweek, fixture_id, xpts):
    """One row shaped like future_predictions (onyl columns that pivot uses)."""
    return {'id': player_id, 'gameweek': gameweek, 'fixture_id': fixture_id, 'xPts': xpts}

def test_single_gameweek_passes_through():
    """Each player has one fixture per GW, xPts appears unchanged."""
    preds = pd.DataFrame([
        _pred_row(100, 5, 1, 6.0),
        _pred_row(101, 5, 1, 4.0),
    ])
    gw_cols = ['xPts_gw5']
    result = _pivot_gw_xpts(preds, gw_cols, next_gw=5, n_gws=1)
    # Player 100 should have xPts_gw5 == 6.0
    # Player 101 should have xPts_gw5 == 4.0
    row = result[result['player_id'] == 100].iloc[0]

    print(row)
    print(row['xPts_gw5'])

    assert row['xPts_gw5'] == 6.0


def test_double_gameweek_sums_fixtures():
    """A player with TWO fixtures in a GW has xPts summed."""
    preds = pd.DataFrame([
        _pred_row(100, 5, 1, 6.0), # fixture 1
        _pred_row(100, 5, 2, 4.0), # fixture 2
    ])
    gw_cols = ['xPts_gw5']
    result = _pivot_gw_xpts(preds, gw_cols, next_gw=5, n_gws=1)
    # Player 100 should have xPts_gw5 == 10.0 (6 + 4)
    row = result[result['player_id'] == 100].iloc[0]

    print(row)
    print(row['xPts_gw5'])

    assert row['xPts_gw5'] == 10.0


def test_blank_gameweek_fills_zero():
    """A player with a fixture in GW5 but NONE in GW6 has xPts == 0 in GW6 (not NaN)."""
    preds = pd.DataFrame([
        _pred_row(100, 5, 1, 6.0), # fixture 1
        _pred_row(101, 6, 2, 4.0), # fixture 2
    ])
    gw_cols = ['xPts_gw5', 'xPts_gw6']
    result = _pivot_gw_xpts(preds, gw_cols, next_gw=5, n_gws=2)
    # Player 100 should have xPts_gw6 == 0.0 (no fixture)
    row = result[result['player_id'] == 100].iloc[0]

    print(row)
    print(row['xPts_gw6'])

    assert row['xPts_gw6'] == 0.0