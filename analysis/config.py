"""Shared constants for UniLID analysis."""

import math

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
SCRATCH_DIR = "/capstor/scratch/cscs/cmeister747/unilid_analysis"
DATA_DIR = f"{SCRATCH_DIR}/glotlid_unilid"
CACHE_DIR = SCRATCH_DIR
TEST_FILE = f"{DATA_DIR}/glotlid_correct_test.txt"
TRAIN_COUNTS_FILE = f"{DATA_DIR}/glotlid_train_counts.json"

PRED_FILES = {
    "UniLID": f"{DATA_DIR}/glotlidc_y_pred.txt",
    "UniLID-DeepSeek": f"{DATA_DIR}/deepseek_v3.2_glotlid_y_pred.txt",
    "UniLID-Qwen": f"{DATA_DIR}/qwen3_8b_glotlid_y_pred.txt",
    "UniLID-Marg": f"{DATA_DIR}/marg_y_pred.txt",
    "fastText": f"{DATA_DIR}/fasttext_y_pred.txt",
}

MODEL_NAMES = list(PRED_FILES.keys())
PRIMARY_MODEL = "UniLID"

TOTAL_LINES = 45_627_279
DEFAULT_SAMPLE_SIZE = 500_000
SAMPLE_SEED = 42

# Training data
TRAIN_FILE = f"{SCRATCH_DIR}/train.txt"
TRAIN_LINES = 60_683_151

# ---------------------------------------------------------------------------
# Text length bins (characters)
# ---------------------------------------------------------------------------
LENGTH_BINS = [0, 30, 75, 150, 300, math.inf]
LENGTH_LABELS = ["<30", "30--75", "75--150", "150--300", "300+"]

# ---------------------------------------------------------------------------
# Resource level bins (training document counts)
# Bin 1: <1k (fixed). Bins 2-5: quartiles of remaining languages, rounded.
# ---------------------------------------------------------------------------
RESOURCE_BINS = [0, 500, 1_000, 12_000, 18_000, 35_000, math.inf]
RESOURCE_LABELS = ["<500", "500--1k", "1k--12k", "12k--18k", "18k--35k", "35k+"]

# ---------------------------------------------------------------------------
# Scripts — top 10 by sample count, rest grouped as "Other"
# ---------------------------------------------------------------------------
TOP_SCRIPTS = [
    "Latn", "Cyrl", "Arab", "Grek", "Deva",
    "Hang", "Hebr", "Beng", "Jpan", "Armn",
]

# ---------------------------------------------------------------------------
# Confusion matrix clusters
# ---------------------------------------------------------------------------
CONFUSION_CLUSTERS = {
    "arabic_dialects": {
        "title": "Arabic Dialects",
        "languages": ["arb_Arab", "arz_Arab", "ary_Arab", "ars_Arab", "apc_Arab", "acm_Arab"],
    },
    "chinese_varieties": {
        "title": "Chinese Varieties",
        "languages": ["cmn_Hani", "wuu_Hani", "yue_Hani", "lzh_Hani", "hak_Hani"],
    },
    "hindi_belt": {
        "title": "Hindi Belt (Devanagari)",
        "languages": ["hin_Deva", "anp_Deva", "bho_Deva", "mai_Deva", "mag_Deva", "hne_Deva", "kas_Deva", "doi_Deva"],
    },
    "malay_indonesian": {
        "title": "Malay--Indonesian",
        "languages": ["ind_Latn", "zsm_Latn", "bjn_Latn"],
    },
    "scandinavian": {
        "title": "Scandinavian",
        "languages": ["dan_Latn", "nob_Latn", "nno_Latn", "swe_Latn"],
    },
    "hebrew": {
        "title": "Hebrew",
        "languages": ["heb_Hebr", "hbo_Hebr"],
    },
    "persian_iranian": {
        "title": "Persian--Iranian",
        "languages": ["fas_Arab", "glk_Arab", "mzn_Arab", "sdh_Arab"],
    },
}
