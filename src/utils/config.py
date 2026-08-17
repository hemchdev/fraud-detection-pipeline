"""
Every important "where is X" or "what number is Y" lives in this one file.

Why bother? Because later, when you have 10 scripts that all need to know
"where's the data" or "what's our random seed", you do NOT want to type
that path 10 times. Change it once here, and every script picks it up.
"""
import os

# BASE_DIR = the top folder of this project (fraud-detection-pipeline/)
# We calculate it automatically so the code works no matter where you
# put the project folder on your computer.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Where things live ---
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

# --- Constants used across scripts ---
RANDOM_SEED = 42          # fixes randomness so results are repeatable
TARGET_COLUMN = "Class"   # 1 = fraud, 0 = normal transaction
