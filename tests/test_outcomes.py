"""Test for outcome parsing and database storage."""

from fpl_model.outcomes import parse_gw_outcomes


def test_parse_gw_outcomes_extracts_points():
    fake_live = {'elements': [
        {'id': 100, 'stats': {'minutes': 90, 'total_points': 6}},
        {'id': 101, 'stats': {'minutes': 45, 'total_points': 2}},
    ]}
    df = parse_gw_outcomes(fake_live, gameweek=5)
    assert len(df) == 2
    assert df[df['player_id'] == 100]['total_points'].iloc[0] == 6
    assert (df['gameweek'] == 5).all()

def test_parse_gw_outcomes_empty_input():
    df = parse_gw_outcomes({}, gameweek=5)
    assert df.empty