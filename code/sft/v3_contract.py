"""Frozen public contract shared by SFT v3 builders, audits, and publishing."""

import os

RECIPE = "sft-v3-single-longhist"
PROMPT_VERSION = "sft-v3-single-longhist-20260721-v5"
EXPECTED_ROWS = 12_000
USERS_PER_GRADE = 6_000
PILOT_ROWS = 6_000
PILOT_USERS_PER_GRADE = 3_000
FORMAL_TARGET_ROWS = (PILOT_ROWS, EXPECTED_ROWS)
PHASE_MANIFEST_SCHEMA = 1
PHASE_SELECTION_VERSION = "sha256-grade-balanced-v1"
PHASE_SPLIT_VERSION = "sha256-grade-stagepart-v1"
PHASE_SEED = 5
DEV_FRAC = 0.08
TEACHER_MODEL = "gemini-3.6-flash"
TEACHER_BACKEND = "vertex"
TEACHER_EFFORT = "medium"
TEACHER_TEMPERATURE = None
TEACHER_THINKING_LEVEL = "medium"
MAX_TOKENS = 8192
THINKING_BUDGET = None
FORMAT_RETRIES = 1
K_GRID = [32, 64, 128, 256]
K_WEIGHTS = [0.10, 0.30, 0.30, 0.30]
N_CAND_GRID = [10, 20, 30, 50, 100]
N_CAND_WEIGHTS = [0.10, 0.15, 0.40, 0.20, 0.15]


def is_smoke_path(path):
    """Whether a path is visibly isolated as temporary smoke output."""
    real = os.path.realpath(path)
    return real.startswith(os.path.realpath("/tmp") + os.sep) and "smoke" in real.lower()
