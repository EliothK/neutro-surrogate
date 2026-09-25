"""Central configuration. Every other module reads ranges, sizes and paths from here."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"

N_CELLS = 192
N_ZONES = 16
assert N_CELLS % N_ZONES == 0, "N_CELLS must be a multiple of N_ZONES"
CELLS_PER_ZONE = N_CELLS // N_ZONES
ZONE_OF_CELL = np.repeat(np.arange(N_ZONES), CELLS_PER_ZONE)

# Parameter ranges (low, high)
SLAB_WIDTH_RANGE = (100.0, 400.0) # cm
DIFF_COEFFICIENT_RANGE = (0.7, 1.4) # cm
MACRO_ABSORP_CROSS_SECTION_RANGE = (0.02, 0.15) # 1/cm
FIS_NEU_PROD_RANGE = (0.02, 0.16) # 1/cm

# Sampling
TARGET_EIGENVALUE_RANGE = (0.90, 1.10)
REFLECTOR_FRACTION = 0.35
MIN_TOTAL_FISSION = 1e-6

# Layout of the input vector: slab width, then per-zone D, Sa, nSf
N_INPUTS = 1 + 3 * N_ZONES
IDX_SLAB = 0
SLICE_DIFF_COEFFICIENT = slice(1, 1 + N_ZONES)
SLICE_MACRO_ABSORP_CROSS_SECTION = slice(1 + N_ZONES, 1 + 2 * N_ZONES)
SLICE_FIS_NEU_PROD = slice(1 + 2 * N_ZONES, 1 + 3 * N_ZONES)

DEFAULT_SEED=62922
DEFAULT_N_SAMPLES = 30000
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
# Training defaults (also the base point of the hyperparameter sweep)
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_LAM = 1.0
DEFAULT_MU_MAX = 0.1
DEFAULT_WIDTH = 256
DEFAULT_DEPTH = 3
DEFAULT_RAMP_FRAC = 0.25
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_DROPOUT = 0.0
DEFAULT_ACTIVATION = "silu"
DEFAULT_SCHEDULER = "cosine"
DEFAULT_TRAIN_FRACTION = 1.0
DEFAULT_GRAD_CLIP = 0.0  # 0 = off

# Smooth-profile sampling: knots per material property, linearly interpolated onto the zones (0 = independent zones)
PROFILE_KNOTS = 5
# Accepted range of k in the natural (un-rescaled) dataset
NATURAL_K_RANGE = (0.5, 1.5)

def input_bounds():
    """Lower and upper bounds of the N_INPUTS dimensional sampling box"""
    low = np.array([SLAB_WIDTH_RANGE[0]]
                  + [DIFF_COEFFICIENT_RANGE[0]] * N_ZONES
                  + [MACRO_ABSORP_CROSS_SECTION_RANGE[0]] * N_ZONES
                  + [FIS_NEU_PROD_RANGE[0]] * N_ZONES)
    high = np.array([SLAB_WIDTH_RANGE[1]]
                  + [DIFF_COEFFICIENT_RANGE[1]] * N_ZONES
                  + [MACRO_ABSORP_CROSS_SECTION_RANGE[1]] * N_ZONES
                  + [FIS_NEU_PROD_RANGE[1]] * N_ZONES)
    return low, high

def widened_bounds(factor=0.2):
    """Bounds widened by `factor` of the span"""
    low, high = input_bounds()
    span = high - low
    return np.maximum(low - factor * span, 0.25 * low), high + factor * span

def ensure_dirs():
    for d in (DATA_DIR, MODEL_DIR, RESULTS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)