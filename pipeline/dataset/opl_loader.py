from __future__ import annotations
import numpy as np
import pandas as pd

from config.settings import (
    DOTS_THRESHOLDS,
    MIN_MEETS_FOR_AMPLITUDE,
    OPL_CLEANED_PATH
)

def _classify_DOTS(dots: float) -> str:

    DOTS_THRESHOLDS = {
        "elite":        510,
        "advanced":     410,
        "intermediate": 310,
    }

    if dots >= DOTS_THRESHOLDS['elite']:
        return "elite"
    if dots >= DOTS_THRESHOLDS['advanced']:
        return 'advanced'
    if dots >= DOTS_THRESHOLDS['intermediate']:
        return 'intermediate'

    return "novice"


def _derive_opl_amplitude(opl_df: pd.DataFrame) -> dict[str, float]:

    meet_counts = opl_df.groupby("Name")['Date'].count()
    qualified = meet_counts[meet_counts >= MIN_MEETS_FOR_AMPLITUDE].index
    df = opl_df[opl_df['Name'].isin(qualified)].copy()
    print(f"[INFO-OPL-Shape] - Minimum {MIN_MEETS_FOR_AMPLITUDE} Meets Filtered OPL Dataset Shape: {df.shape}")

    peak_dots = df.groupby("Name")['Dots'].max().rename('peak_dots')
    df = df.join(peak_dots, on = 'Name')
    df['training_level'] = df['peak_dots'].apply(_classify_DOTS)

    df['prev_total'] = df.sort_values(['Name', 'Date'])['TotalKg'].shift(1)
    df = df.dropna(subset = ['prev_total', 'TotalKg'])
    df = df[df['prev_total'] > 0]
    df['progression'] = (df['TotalKg'] - df['prev_total']) / df['prev_total']
    df = df[df['progression'] > 0]
    df = df[df["progression"] <= 0.20]

    print(f"[INFO-OPL-Shape] - Increasing Progression Filtered OPL Dataset Shape: {df.shape}")


    amplitude = (
        df.groupby('training_level')['progression']
        .mean()
        .to_dict()
    )

    from config.settings import LEVEL_SHAPE_PARAMS
    AMPLITUDE_FLOOR = 0.005   # 0.5% minimum — ensures block always targets a new PR
    for level in list(amplitude.keys()):
        ceiling   = LEVEL_SHAPE_PARAMS[level]["ceiling"]
        amp_cap   = (1.0 / ceiling) - 1.0   # e.g. 0.95 ceiling → cap = 0.0526
        amplitude[level] = max(AMPLITUDE_FLOOR,
                               min(amplitude[level], amp_cap * 0.95))  # 5% headroom under cap
 
    counts = df.groupby("training_level")["progression"].count().to_dict()
    for level, val in amplitude.items():
        n = counts.get(level, 0)
        print(f"  [OPL amplitude] {level:>12}: {val:.4f}  ({n:,} progressions)")
    
    return amplitude


def get_opl_dataset(path) -> pd.DataFrame:

    cleaned_path = OPL_CLEANED_PATH

    if cleaned_path.is_file():
        print(f"[INFO-OPL] - Found Cleaned OPL dataset - Loading directly...")
        df = pd.read_csv(cleaned_path, low_memory = False)
        print(f"[INFO-OPL-Shape] Cleaned OPL Dataset Shape: {df.shape}")
        return df 


    print("[INFO-OPL] - Reading original OPL, Cleaning dataset and Saving")
    cols = ['Name', 'Sex', 'Age', 'BodyweightKg', 'WeightClassKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'TotalKg', 'Dots', 'Date'] 
    df = pd.read_csv(path, usecols = lambda c : c in cols, low_memory = False)
    print(f"[INFO-OPL-Shape] Original OPL Dataset Shape: {df.shape}")

    # Keep Valid Data only
    df = df.dropna(subset = ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Dots', 'Age'])
    for lift in ['Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg', 'Dots', 'Age']:
        df = df[df[lift] > 0]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.reset_index(drop = True)

    print(f"[INFO-OPL-Shape] Cleaned OPL Dataset Shape: {df.shape}")
    df.to_csv(cleaned_path, index = False)
    print(f"[INFO-OPL] Cleaned OPL Dataset Saved to: {cleaned_path}")

    return df
