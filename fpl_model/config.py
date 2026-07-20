""" Configuration: paths, API settings, scoring rules, model constants. """

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
OUTPUT_DIR   = DATA_DIR / 'outputs'

BIG5_XG_PATH = RAW_DATA_DIR / 'big5_xG_data.csv'
LOG_PATH     = PROJECT_ROOT / 'fpl_xpts.log'

# ── FPL API  ────────────────────────────────────────────────────────────────────
FPL_BASE = 'https://fantasy.premierleague.com/api/'
API_DELAY = 0.5  # seconds between API calls

# ── Season ──────────────────────────────────────────────────────────────────────
CURRENT_SEASON = '2025-26'

# ── Team name mapping / scoring rules / thresholds / prior ───────────────────────────
FBREF_TO_FPL = {
    'Manchester City':  'Man City',
    'Manchester Utd':   'Man Utd',
    'Newcastle Utd':    'Newcastle',
    "Nott'ham Forest":  "Nott'm Forest",
    'Leeds United':     'Leeds',
    'Tottenham':        'Spurs',
}

# Points per event by position.
# play_u60 / play_60plus are cumulative: a player who plays 60+ gets both.
SCORING = {
    'GKP': {
        'play_u60':        1,
        'play_60plus':     2,    # total for 60+ (1 base + 1 bonus)
        'goal':            10,
        'assist':          3,
        'clean_sheet':     4,
        'goals_conceded': -0.5,  # -1 per 2 goals conceded
        'yellow':         -1,
        'red':            -3,
        'save_per_3':      1,    # 1pt per 3 saves
        'pen_save':        5,
        'pen_miss':       -2,
        'own_goal':       -2,
        'defcon_eligible': False,
    },
    'DEF': {
        'play_u60':        1,
        'play_60plus':     2,
        'goal':            6,
        'assist':          3,
        'clean_sheet':     4,
        'goals_conceded': -0.5,
        'yellow':         -1,
        'red':            -3,
        'save_per_3':      0,
        'pen_save':        0,
        'pen_miss':       -2,
        'own_goal':       -2,
        'defcon_eligible': True,
    },
    'MID': {
        'play_u60':        1,
        'play_60plus':     2,
        'goal':            5,
        'assist':          3,
        'clean_sheet':     1,
        'goals_conceded':  0,
        'yellow':         -1,
        'red':            -3,
        'save_per_3':      0,
        'pen_save':        0,
        'pen_miss':       -2,
        'own_goal':       -2,
        'defcon_eligible': True,
    },
    'FWD': {
        'play_u60':        1,
        'play_60plus':     2,
        'goal':            4,
        'assist':          3,
        'clean_sheet':     0,
        'goals_conceded':  0,
        'yellow':         -1,
        'red':            -3,
        'save_per_3':      0,
        'pen_save':        0,
        'pen_miss':       -2,
        'own_goal':       -2,
        'defcon_eligible': True,
    },
}

# Defensive contribution threshold to earn bonus point (total CBI + recoveries + tackles).
# P(defcon bonus) = P(total > threshold) under a Poisson model.
# GKPs are not eligible.
DEFCON_THRESHOLD = {
    'GKP': None,
    'DEF': 10,
    'MID': 12,
    'FWD': 12,
}

N_PRIOR = 10