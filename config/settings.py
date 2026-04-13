from pathlib import Path
import os

# DATA SET PATHS
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "output"
CLEANED_DIR = DATA_DIR / 'cleaned'
PDF_DIR  = DATA_DIR / "pdfs"         
PDF_CONFIG_PATH   = ROOT_DIR / "config" / "pdf_config.toml"  # PDF generation config
BM_INDEX_PATH = OUT_DIR / 'BM_index.pkl'
BM_CORPUS_PATH = OUT_DIR / 'BM_corpus.json'



# RAW SOURCE DATA
OPL_PATH = RAW_DIR / "openpowerlifting.csv"
GYM_EX_PATH = RAW_DIR / "programs_detailed_boostcamp_kaggle.csv"
GYM_PROG_PATH = RAW_DIR / "program_summary.csv"

# DERIVED DATA 
OPL_CLEANED_PATH = CLEANED_DIR / 'opl_cleaned.csv'
GYM_COMBINED_PATH = CLEANED_DIR / "combined_600k_dataset.csv"
SESSIONS_PATH = OUT_DIR / "sessions.csv"
BLOCK_SUMMARY_PATH = OUT_DIR / "block_summary.csv"
CHECKPOINT_PATH = DATA_DIR / "checkpoint.json"


# GENERATION PARAMETERS
N_ATHLETES = 3000
CHECKPOINT_THRESHOLD = 50
BLOCK_WEEKS = 12
SESSIONS_PER_WEEK = 4
ACCESSORIES_PER_SESSION = 4


# DOTS THRESHOLDS
DOTS_THRESHOLDS = {
    "elite":        510,
    "advanced":     410,
    "intermediate": 310,
}

# MINIMUM MEETS FOR AMPLITUDE CALCLUATION
MIN_MEETS_FOR_AMPLITUDE = 5

# Maps our training_level to the SET of acceptable level strings in the 600K dataset.
# "novice" maps to both because the dataset uses both labels interchangeably.
LEVEL_MAP = {
    "elite":        {"advanced"},
    "advanced":     {"advanced"},
    "intermediate": {"intermediate"},
    "novice":       {"beginner", "novice"}
}

# 600K GOAL CLASSIFICATION
STRENGTH_GOAL_KEYWORDS = {
    "strength", "muscle", "hypertrophy", "powerlifting",
    "bulk", "mass", "gain", "power",
    "weightlifting",  
    "build",           
}

# SESSION STRUCTURE
# SESSION_FOCUS: dict[int, dict] = {
#     0: {
#         "label": "Lower A",
#         "body_keywords": [
#             "squat", "leg", "lunge", "hamstring", "glute",
#             "hip", "calf", "quad", "bulgarian", "goblet",
#         ],
#     },
#     1: {
#         "label": "Upper A",
#         "body_keywords": [
#             "bench", "press", "chest", "shoulder", "tricep",
#             "push", "dip", "fly", "pec", "incline", "decline",
#         ],
#     },
#     2: {
#         "label": "Lower B",
#         "body_keywords": [
#             "deadlift", "row", "cleans", "back", "lat", "trap",
#             "bicep", "curl", "rdl", "romanian", "rack",
#         ],
#     },
#     3: {
#         "label": "Upper B",
#         "body_keywords": [
#             "overhead", "ohp", "shoulder", "press", "lateral",
#             "raise", "tricep", "dip", "push", "deltoid", "shrug",
#         ],
#     },
# }


SESSION_FOCUS: dict[int, dict] = {
    0: {
        "label": "Lower A",
        "body_keywords": [
            "back squat", "front squat", "hack squat", "leg press", "goblet squat", 
            "split squat", "bulgarian", "leg extension", "quad extension", "walking lunge", 
            "reverse lunge", "step up", "sissy squat", "v-squat", "pendulum squat", 
            "smith machine squat", "belt squat", "zercher", "box squat", "cyclist squat",
            "standing calf", "seated calf", "donkey calf", "tibialis raise", "soleus", 
            "adductor machine", "inner thigh press", "sled push", "wall sit", "jump squat",
            "pistol squat", "landmine squat", "cossack squat", "heel elevated", "quad dominant"
        ],
    },
    1: {
        "label": "Upper A",
        "body_keywords": [
            "bench press", "chest press", "incline bench", "decline bench", "flat bench", 
            "dumbbell bench", "db bench", "floor press", "board press", "pec deck", 
            "chest fly", "cable fly", "weighted pushup", "chest dip", "machine press",
            "barbell row", "dumbbell row", "db row", "seal row", "t-bar row", 
            "chest supported row", "kroc row", "pendlay row", "cable row", "seated row", 
            "meadows row", "inverted row", "one arm row", "tricep pushdown", "tricep extension", 
            "skullcrusher", "french press", "tate press", "close grip bench", "jm press"
        ],
    },
    2: {
        "label": "Lower B",
        "body_keywords": [
            "conventional deadlift", "sumo deadlift", "romanian deadlift", "rdl", "stiff leg deadlift", 
            "sldl", "deficit deadlift", "trap bar deadlift", "hex bar deadlift", "rack pull", 
            "snatch grip deadlift", "hip thrust", "glute bridge", "barbell bridge", "good morning", 
            "nordic curl", "leg curl", "hamstring curl", "lying leg curl", "seated leg curl", 
            "standing leg curl", "hyperextension", "back extension", "glute-ham raise", "ghr", 
            "reverse hyper", "kettlebell swing", "kb swing", "glute kickback", "cable pull through", 
            "landmine rdl", "single leg deadlift", "b-stance deadlift", "hamstring slide", "posterior chain",
            "deadlift"
        ],
    },
    3: {
        "label": "Upper B",
        "body_keywords": [
            "overhead press", "ohp", "military press", "strict press", "push press", 
            "shoulder press", "arnold press", "z press", "landmine press", "lateral raise", 
            "side raise", "front raise", "rear delt fly", "facepull", "upright row", 
            "barbell shrug", "dumbbell shrug", "trap raise", "lat pulldown", "wide grip pull", 
            "neutral grip pull", "pullup", "pull up", "chinup", "chin up", "weighted pullup", 
            "lat prayer", "straight arm pulldown", "bicep curl", "hammer curl", "preacher curl", 
            "incline curl", "concentration curl", "spider curl", "ez bar curl"
        ],
    },
}

PHASE_BOUNDARIES = {
    "accumulation":    range(1, 5),
    "intensification": range(5, 9),
    "realisation":     range(9, 12),
    "deload":          range(12, 13),
}

# Periodization shape
# Source: Prilepin (1974) intensity zones; Issurin (2008) block phase structure.
PERIODIZATION_SHAPE: list[float] = [
    0.00, 0.07, 0.14, 0.21,   # accumulation    weeks 1–4
    0.29, 0.37, 0.46, 0.55,   # intensification weeks 5–8
    0.65, 0.80, 1.00,          # realisation     weeks 9–11
    0.20,                      # deload          week 12
]


# PER LEVEL FLOOR AND CEILING 
LEVEL_SHAPE_PARAMS = {
    "novice":       {"floor": 0.65, "ceiling": 0.90},
    "intermediate": {"floor": 0.68, "ceiling": 0.93},
    "advanced":     {"floor": 0.72, "ceiling": 0.95},
    "elite":        {"floor": 0.75, "ceiling": 0.93},
}

RPE_FLOOR = {
    "novice":       6.5,
    "intermediate": 7.0,
    "advanced":     7.5,
    "elite":        8.0,
}

RPE_CEILING: float = 9.5

OHP_BENCH_RATIO = {
    "novice":       0.61,
    "intermediate": 0.64,
    "advanced":     0.67,
    "elite":        0.70,
}

BARBELL_EQUIPMENT_VALUES: set[str] = {"Full Gym", "Garage Gym"}
OPL_ROW_INDEX_COL    = "opl_row_index"      
ACCESSORY_TITLES_COL = "accessory_titles"
PRIMARY_PROGRAM_COL = "primary_program"
SECONDARY_PROGRAM_COL = "secondary_program"


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")