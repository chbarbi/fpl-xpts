""" Tests opponent adjustment functions. """

import pandas as pd
import pytest

from fpl_model.rates import adjust_rates_for_opponent

def _make_team_lookup():
    """ Minimal team_lookup with specified opponent (id=7). """
    return pd.DataFrame(
        {
            'home_attack': [50.0],
            'home_defense': [50.0],
            'away_attack': [50.0],
            'away_defense': [50.0],
        },
        index=[7],
    )

def test_average_opponent_leaves_rates_unchanged():
    """ An opponent with all-50 (i.e. league average) ratings gives 1.0 multiplier. """
    rates = pd.Series({'lam_goals_scored': 0.5, 'lam_assists': 0.3})
    lookup = _make_team_lookup()

    result = adjust_rates_for_opponent(rates, opp_team_id=7, is_home=True, team_lookup=lookup)

    assert result['lam_goals_scored'] == pytest.approx(0.5)
    assert result['lam_assists'] == pytest.approx(0.3)

def test_weak_defence_boosts_attacking_rates():
    """ An opponent with weak away defence (30) should raise attacking rates. """
    lookup = _make_team_lookup()
    lookup.loc[7, 'away_defense'] = 30  # weak away defence

    rates = pd.Series({'lam_goals_scored': 0.5})
    result = adjust_rates_for_opponent(rates, opp_team_id=7, is_home=True, team_lookup=lookup)

    # attack multiplier = (100 - 30) / 50 = 1.4
    assert result['lam_goals_scored'] == pytest.approx(0.5 * 1.4)

def test_home_player_uses_opponent_away_ratings():
    """ When the player is home, the opponent's away ratings are used. """
    lookup = _make_team_lookup()
    lookup.loc[7, 'away_defense'] = 30.0  # weak away defence
    lookup.loc[7, 'home_defense'] = 70.0  # strong home defence

    rates = pd.Series({'lam_goals_scored': 0.5})
    result = adjust_rates_for_opponent(rates, opp_team_id=7, is_home=True, team_lookup=lookup)

    # player home -> opponent away defence (30) used, not home (70).
    # (100 - 30) / 50 = 1.4
    assert result['lam_goals_scored'] == pytest.approx(0.5 * 1.4)

def test_missing_team_returns_unchanged():
    """ An opponent not in the lookup gets no adjustment (first guard). """
    rates = pd.Series({'lam_goals_scored': 0.5})
    result = adjust_rates_for_opponent(rates, opp_team_id=999, is_home=True, team_lookup=_make_team_lookup())
    assert result['lam_goals_scored'] == pytest.approx(0.5)

def test_nan_strength_returns_unchanged():
    """ An opponent present with NaN strength data gets no adjustment (second guard). """
    lookup = _make_team_lookup()
    lookup.loc[7, 'away_defense'] = float('nan')  # promoted team case for first COLD_START_GWS gameweeks

    rates = pd.Series({'lam_goals_scored': 0.5})
    result = adjust_rates_for_opponent(rates, opp_team_id=7, is_home=True, team_lookup=lookup)
    assert result['lam_goals_scored'] == pytest.approx(0.5)