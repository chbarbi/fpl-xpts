"""xPts computation functions."""

import pandas as pd
import numpy as np

from scipy.stats import poisson

from fpl_model.config import DEFCON_THRESHOLD, SCORING

def compute_xpts(row: pd.Series, mode: str = 'projected') -> dict:
    """
    Compute xPts and all component contributions for one player-fixture.

    Parameters
    ----------
    row  : pd.Series — either a player_rates row (projected) or performances row (historic)
    mode : 'projected' | 'historic'

    Returns
    -------
    dict with keys: xPts, xPts_mins, xPts_cs, xPts_gc, xPts_goals, xPts_assists,
                    xPts_saves, xPts_yellows, xPts_reds, xPts_pen_save,
                    xPts_pen_miss, xPts_defcon
    """
    pos = row.get('position')
    if pos not in SCORING:
        return {'xPts': 0.0}

    s   = SCORING[pos]
    out = {}

    # ── Playing time ──────────────────────────────────────────────────────────
    if mode == 'projected':
        avg_mins     = float(row.get('avg_minutes', 60))
        prob_plays60 = 1.0 - poisson.cdf(59, mu=avg_mins)
        # 1pt for playing at all (assumed starter), +1pt if 60+ mins
        out['xPts_mins'] = round(
            s['play_u60'] + (s['play_60plus'] - s['play_u60']) * prob_plays60, 3
        )
    else:
        mins = float(row.get('minutes', 0))
        out['xPts_mins'] = (
            s['play_u60'] * int(mins > 0) +
            (s['play_60plus'] - s['play_u60']) * int(mins >= 60)
        )

    # ── Clean sheet ───────────────────────────────────────────────────────────
    # P(CS) = P(goals_conceded = 0) = exp(-λ_gc)
    # Conditional on playing 60+ minutes (necessary condition for CS award).
    if mode == 'projected':
        lam_gc  = float(row.get('lam_goals_conceded', 0))
        prob_cs = np.exp(-lam_gc) * prob_plays60
    else:
        # Use Opta xGC as the best estimate of expected goals conceded in that match
        lam_gc  = float(row.get('xGC', 0))
        prob_cs = np.exp(-lam_gc) * int(float(row.get('minutes', 0)) >= 60)
    out['xPts_cs'] = round(s['clean_sheet'] * prob_cs, 3)

    # ── Goals conceded (GKP/DEF penalty) ─────────────────────────────────────
    if s['goals_conceded'] != 0:
        gc_lam = float(row.get('lam_goals_conceded' if mode == 'projected' else 'goals_conceded', 0))
        # E[goals conceded] = λ; -0.5 pts per goal (-1 per 2)
        out['xPts_gc'] = round(s['goals_conceded'] * gc_lam, 3)
    else:
        out['xPts_gc'] = 0.0

    # ── Goals scored ──────────────────────────────────────────────────────────
    # Projected: use shrunk goal-scoring rate
    # Historic:  use Opta xG as λ (better reflection of quality of chances)
    lam_g          = float(row.get('lam_goals_scored' if mode == 'projected' else 'xG', 0))
    out['xPts_goals'] = round(lam_g * s['goal'], 3)

    # ── Assists ───────────────────────────────────────────────────────────────
    lam_a             = float(row.get('lam_assists' if mode == 'projected' else 'xA', 0))
    out['xPts_assists'] = round(lam_a * s['assist'], 3)

    # ── Saves (GKP only) ──────────────────────────────────────────────────────
    # FPL: 1pt per 3 saves = 1/3 pt per save
    if s['save_per_3'] > 0:
        lam_sv         = float(row.get('lam_saves' if mode == 'projected' else 'saves', 0))
        out['xPts_saves'] = round(lam_sv * (1/3), 3)
    else:
        out['xPts_saves'] = 0.0

    # ── Yellow cards ──────────────────────────────────────────────────────────
    # P(exactly 1 yellow) — 2-yellow reds appear in red_cards, not here.
    lam_y              = float(row.get('lam_yellow_cards' if mode == 'projected' else 'yellow_cards', 0))
    out['xPts_yellows'] = round(s['yellow'] * poisson.pmf(1, mu=lam_y), 3)

    # ── Red cards ─────────────────────────────────────────────────────────────
    # Captures both straight reds and 2-yellow reds (both recorded as red_cards).
    lam_r            = float(row.get('lam_red_cards' if mode == 'projected' else 'red_cards', 0))
    out['xPts_reds'] = round(s['red'] * poisson.pmf(1, mu=lam_r), 3)

    # ── Penalty saves ─────────────────────────────────────────────────────────
    if s['pen_save'] > 0:
        lam_ps              = float(row.get('lam_penalties_saved' if mode == 'projected' else 'penalties_saved', 0))
        out['xPts_pen_save'] = round(lam_ps * s['pen_save'], 3)
    else:
        out['xPts_pen_save'] = 0.0

    # ── Penalty misses ────────────────────────────────────────────────────────
    lam_pm              = float(row.get('lam_penalties_missed' if mode == 'projected' else 'penalties_missed', 0))
    out['xPts_pen_miss'] = round(lam_pm * s['pen_miss'], 3)

    # ── Defensive contribution bonus ──────────────────────────────────────────
    # 2 bonus pts if total defensive actions exceed position-specific threshold.
    # GKPs are explicitly excluded.
    if s['defcon_eligible']:
        threshold       = DEFCON_THRESHOLD.get(pos, 999)
        lam_dc          = float(row.get(
            'lam_defensive_contribution' if mode == 'projected' else 'defensive_contribution', 0
        ))
        prob_defcon     = 1.0 - poisson.cdf(threshold - 1, mu=lam_dc)
        out['xPts_defcon'] = round(2.0 * prob_defcon, 3)
    else:
        out['xPts_defcon'] = 0.0

    # ── Total ─────────────────────────────────────────────────────────────────
    component_keys = [
        'xPts_mins', 'xPts_cs', 'xPts_gc', 'xPts_goals', 'xPts_assists',
        'xPts_saves', 'xPts_yellows', 'xPts_reds',
        'xPts_pen_save', 'xPts_pen_miss', 'xPts_defcon'
    ]
    out['xPts'] = round(sum(out[k] for k in component_keys), 2)
    return out