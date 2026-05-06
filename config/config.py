from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR     = DATA_DIR / "cache"
OUTPUTS_DIR   = ROOT_DIR / "outputs"
MODELS_DIR    = OUTPUTS_DIR / "models"
PLOTS_DIR     = OUTPUTS_DIR / "plots"
REPORTS_DIR   = OUTPUTS_DIR / "reports"

# Create them if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, OUTPUTS_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Seasons ─────────────────────────────────────────────────────────────────
SEASONS        = list(range(2010, 2026))   # 2018–2025 for training
PREDICT_SEASON = 2026

# ── Regulation eras ─────────────────────────────────────────────────────────
# Constructor ELO decays heavily at these boundaries
REGULATION_ERAS = {
    2014: "hybrid_v6",
    2017: "wider_cars",
    2019: "simplified_aero",
    2022: "ground_effect",
    2026: "active_aero_50_50_pu",
}

# ── Points system ────────────────────────────────────────────────────────────
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8,  7: 6,  8: 4,  9: 2,  10: 1,
}
FASTEST_LAP_POINT = 1  # top-10 finisher only, post-2019

SPRINT_POINTS = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1,
}

# ── Rookies by season ────────────────────────────────────────────────────────
# Used to seed ELO ratings from F2 performance
KNOWN_ROOKIES_BY_SEASON = {
    2019: ["NOR", "RUS"],
    2020: [],
    2021: ["TSU"],
    2022: ["ZHO"],
    2023: ["PIA", "SAR"],
    2024: ["BEA", "COL"],
    2026: ["ANT", "BOR", "DOO"],
}

# F2 result → estimated F1 ELO starting rating
F2_TO_F1_ELO_MAP = {
    "champion":  1847,
    "runner_up": 1823,
    "third":     1800,
    "top_6":     1780,
    "unknown":   1760,
}

# ── 2026 driver-team lineup ──────────────────────────────────────────────────
DRIVER_TEAM_2026 = {
    "VER": "Red Bull Racing",
    "HAD": "Red Bull Racing",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "DOO": "Alpine",
    "ALB": "Williams",
    "SAI": "Williams",
    "HUL": "Kick Sauber",
    "BOR": "Kick Sauber",
    "LAW": "Racing Bulls",
    "LAB": "Racing Bulls",
    "OCO": "Haas",
    "BEA": "Haas",
}

# ── ELO settings ─────────────────────────────────────────────────────────────
ELO_INITIAL_RATING = 1800
ELO_K_FACTOR       = 32
ELO_DECAY_PER_YEAR = 0.1    # 10% regression toward mean each off-season

# ── Model defaults ────────────────────────────────────────────────────────────
XGBOOST_DEFAULTS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state":     42,
    "n_jobs":           -1,
}

LIGHTGBM_DEFAULTS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":          -1,
}