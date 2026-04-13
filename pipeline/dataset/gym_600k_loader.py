from __future__ import annotations
import ast
import numpy as np
import pandas as pd
import random
import math

from config.settings import (
    GYM_COMBINED_PATH,
    STRENGTH_GOAL_KEYWORDS,
    BARBELL_EQUIPMENT_VALUES,
    LEVEL_MAP,
    ACCESSORIES_PER_SESSION,
    SESSION_FOCUS
)

from pipeline.dataset.custom_dataclasses import Exercise, GymData

AccessoryPools = dict

def _parse_list_field(val) -> list[str]:
    """
    Convert list-literal strings to actual Python lists, normalised to lowercase.
    "['strength', 'hypertrophy']" → ["strength", "hypertrophy"]
    "strength"                    → ["strength"]
    "[]" / NaN / None             → []
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s == "[]":
        return []
    if s.startswith("["):
        try:
            result = ast.literal_eval(s)
            return [str(v).strip().lower() for v in result if str(v).strip()]
        except (ValueError, SyntaxError):
            return []
    return [s.lower()]


def _derive_strength(goals: pd.Series) -> set:

    all_goals = {g for goal in goals for g in goal}
    strength = {g for g in all_goals if any(kw in g for kw in STRENGTH_GOAL_KEYWORDS)}
    print(f"  [goals]     {len(all_goals):,} unique goal strings found in dataset")
    print(f"  [goals]     {len(strength):,} classified as strength-relevant: {sorted(strength)}")
    return strength




def get_gym_dataset(ex_path, prog_path) -> pd.DataFrame:

    combined_path = GYM_COMBINED_PATH
    if combined_path.is_file():

        print("[INFO-600K-Combined] - Found Combined 600k Dataset")
        ex_df = pd.read_csv(combined_path, low_memory = False)
        ex_df["goal"]  = ex_df["goal"].apply(_parse_list_field)
        ex_df["level"] = ex_df["level"].apply(_parse_list_field)
        print(f"[INFO-600K-Combined] Combined Cleaned 600k Dataset Shape: {ex_df.shape}")
        # return ex_df

    else:
        print("[INFO-600K-Combined] - Reading Both datasets, Cleaning datasets and Saving")

        check_600k_EX_exist = ex_path
        check_600k_PROG_exist = prog_path

        # ----------------------------------------------------------------------------------------------------------
        # Read 600K Original Datasets 

        ex_df = pd.read_csv(check_600k_EX_exist, low_memory = False)
        prog_df = pd.read_csv(check_600k_PROG_exist, low_memory = False)

        print("[INFO-600k-EX] - Reading original 600K Fitness Exercises data")
        print(f"[INFO-600k-EX] Original 600K Fitness Exercises Dataset Shape: {ex_df.shape}")

        print("[INFO-600k-PROG] - Reading original 600K Program Summary data")
        print(f"[INFO-600k-PROG] Original 600K Program Summary Dataset Shape: {prog_df.shape}")

        # ----------------------------------------------------------------------------------------------------------
        # Drop Goal less Programs and corresponding exercises 

        prog_df = prog_df[prog_df['goal'] != '[]'].reset_index(drop = True)
        prog_df = prog_df[prog_df['equipment'] != '[]'].reset_index(drop = True)
        prog_df = prog_df[prog_df['equipment'].notna()].reset_index(drop = True)
        prog_df = prog_df[prog_df['description'].notna()].reset_index(drop = True)
        prog_df = prog_df[prog_df['last_edit'].notna()].reset_index(drop = True)


        valid_titles = set(prog_df['title'])
        ex_df = ex_df[ex_df['title'].isin(valid_titles)].reset_index(drop = True)

        print(f"[INFO-600k_EX] - Dropped Goalless Exercises - Shape: {ex_df.shape}")
        print(f"[INFO-600k_PROG] - Dropped Goalless program - Shape: {prog_df.shape}")

        # ----------------------------------------------------------------------------------------------------------
        # Join Program and Exercises dataset onto exercises rows

        prog_slim = (
            prog_df[["title", "goal", "equipment"]]
            .drop_duplicates("title")
            .rename(columns={"goal": "goal_prog", "equipment": "equipment_prog"})
        )

        ex_df = ex_df.merge(prog_slim, on="title", how="left")

        # ----------------------------------------------------------------------------------------------------------
        # Fill Program level goals and equipment onto exercises goals 

        for col, prog_col in [("goal", "goal_prog"), ("equipment", "equipment_prog")]:
            if prog_col in ex_df.columns:
                ex_df[col] = ex_df[prog_col].fillna(ex_df[col])
                ex_df.drop(columns=[prog_col], inplace=True)

        # ----------------------------------------------------------------------------------------------------------
        # Convert list literal string to Python lists

        ex_df['goal'] = ex_df['goal'].apply(_parse_list_field)
        ex_df['level'] = ex_df['level'].apply(_parse_list_field)
        ex_df = ex_df[ex_df["equipment"].isin(BARBELL_EQUIPMENT_VALUES)].reset_index(drop=True)


        # ----------------------------------------------------------------------------------------------------------
        # Convert Sets and Reps to int

        # ex_df["sets"] = pd.to_numeric(ex_df["sets"], errors="coerce").astype(int)
        ex_df["reps"] = pd.to_numeric(ex_df["reps"], errors="coerce")
        ex_df['intensity'] = pd.to_numeric(ex_df['intensity'], errors = "coerce").fillna(0.0)
        # ex_df = ex_df[ex_df['level'] != '[]'].reset_index(drop = True)

        print(f"[INFO-600K-Combined] Final 600k dataset Shape]: {ex_df.shape}")
        ex_df.to_csv(combined_path, index = False)

    strength_goals = _derive_strength(ex_df["goal"])
    barbell_equip  = BARBELL_EQUIPMENT_VALUES

    return GymData(df = ex_df, strength_goals = strength_goals, barbell_equip = barbell_equip)


def build_program_catalog(
        gym_df: pd.DataFrame, 
        strength_goals: set,
) -> dict[str, list[str]]:
    
    catalog: dict[str, list[str]] = {}
    goal_mask = gym_df['goal'].apply(lambda x: bool(set(x) & strength_goals))
    strength_df = gym_df[goal_mask]

    for our_level, dataset_levels in LEVEL_MAP.items():
        level_mask = strength_df['level'].apply(lambda x: bool(set(x) & dataset_levels))
        titles = strength_df[level_mask]['title'].dropna().unique().tolist()
        # catalog[our_level] = sorted(titles)
        catalog[our_level] = sorted(titles)

    return catalog

def select_program(
        training_level: str,
        program_catalog: dict[str, list[str]],
        seed: int,
        exclude: str = ""
) -> str:
    
    candidates = [t for t in program_catalog.get(training_level, []) if t != exclude]

    if not candidates:
        all_titles = [t for titles in program_catalog.values() 
                      for t in titles if t != exclude]
        candidates = all_titles
    
    if not candidates: return ""

    return random.Random(seed).choice(candidates)



def precompute_accessory_pools(
    gym_df: pd.DataFrame,
    strength_goals: set,
    barbell_equip: set,
) -> AccessoryPools:

    pools: AccessoryPools = {}
 
    for level in ["novice", "intermediate", "advanced", "elite"]:
        for day_idx in range(4):
            level_targets  = LEVEL_MAP[level]
            body_keywords  = SESSION_FOCUS[day_idx]["body_keywords"]
 
            level_mask = gym_df["level"].apply(lambda x: bool(set(x) & level_targets))
            goal_mask  = gym_df["goal"].apply(lambda x: bool(set(x) & strength_goals))
            equip_mask = gym_df["equipment"].str.strip().isin(barbell_equip)
            body_mask  = gym_df["exercise_name"].str.lower().apply(
                lambda name: any(kw in name for kw in body_keywords)
            )
 
            pool = pd.DataFrame()
            for mask in [
                body_mask & level_mask & goal_mask & equip_mask,
                body_mask & level_mask & goal_mask,
                body_mask & goal_mask,
                body_mask,
                # level_mask & goal_mask & equip_mask,
                # level_mask & goal_mask,
                # goal_mask,
                # pd.Series(True, index=gym_df.index),
            ]:
                pool = gym_df[mask].dropna(subset=["exercise_name", "reps"])
                if not pool.empty:
                    break
 
            # Sort by intensity descending (0 = missing = lowest priority)
            pool = pool.copy()
            pool["_intensity_sort"] = pool["intensity"].where(pool["intensity"] > 0, -1)
            pool = pool.sort_values("_intensity_sort", ascending = False).reset_index(drop = True)
            pool.drop(columns=["_intensity_sort"], inplace = True)

            pool = pool.drop_duplicates(subset = ['exercise_name']).reset_index(drop = True)
            pools[(level, day_idx)] = pool
            print(f"  [pool] ({level:>12}, day {day_idx}): {len(pool):>6,} exercises")
 
    return pools




def query_accessories(
        pools: AccessoryPools,
        training_level: str,
        day_index: int, 
        n: int = ACCESSORIES_PER_SESSION,
        seed: int = 0,
        primary_program: str = "",
        secondary_program: str = ""

) -> list[Exercise]:
    
    pool = pools[(training_level, day_index)]
    level_targets = LEVEL_MAP[training_level]
    level_label = sorted(level_targets)[0]
    used_idx: set = set()
    used_names: set = set()
    parts: list = []


    def _draw(sub_df: pd.DataFrame, count: int, seed_offset: int = 0) -> pd.DataFrame:
        available = sub_df[
            ~sub_df.index.isin(used_idx) &
            ~sub_df['exercise_name'].isin(used_names)
            ]
        if available.empty or count <= 0:
            return pd.DataFrame()
        available = available.drop_duplicates(subset = ['exercise_name'])
        take = min(count, len(available))
        return available.sample(n = take, random_state = seed + seed_offset, replace = False)

    if primary_program:
        # ── Option 3: primary → secondary → fallback ─────────────────────────
        needed = n
 
        drawn = _draw(pool[pool["title"] == primary_program], needed, 0)
        if not drawn.empty:
            parts.append(drawn)
            used_idx.update(drawn.index)
            used_names.update(drawn['exercise_name'])
            needed -= len(drawn)
 
        if needed > 0 and secondary_program:
            drawn = _draw(pool[pool["title"] == secondary_program], needed, 1)
            if not drawn.empty:
                parts.append(drawn)
                used_idx.update(drawn.index)
                used_names.update(drawn['exercise_name'])
                needed -= len(drawn)
 
        if needed > 0:
            drawn = _draw(pool, needed, 2)
            if not drawn.empty:
                parts.append(drawn)
                used_idx.update(drawn.index)
                used_names.update(drawn['exercise_name'])


    else:

        max_per_prog = max(1, math.ceil(n * 0.4))
        pool_ranked = pool.copy()
        pool_ranked["_rank"] = pool_ranked.groupby("title").cumcount()
        diverse = pool_ranked[pool_ranked["_rank"] < max_per_prog].drop(columns = ["_rank"])
        diverse = diverse.sample(frac=1, random_state=seed)
        if len(diverse) >= n:
            parts.append(diverse.head(n))
        else:
            parts.append(diverse)
            remainder = pool[~pool.index.isin(diverse.index)]
            if not remainder.empty:
                fill_n = min(n - len(diverse), len(remainder))
                remainder = remainder[
                    ~remainder['exercise_name'].isin(diverse['exercise_name'])
                ]
                if not remainder.empty:
                    fill_n = min(fill_n, len(remainder))
                    parts.append(remainder.sample(n = fill_n, random_state  =seed, replace = False))

    
    if not parts: return []
    sampled = pd.concat(parts).reset_index(drop = True)
    
    exercises = []
    for _, row in sampled.iterrows():
        raw_reps = float(row["reps"])
        exercises.append(Exercise(
            name       = str(row["exercise_name"]),
            goal       = ", ".join(row["goal"]) if row["goal"] else "",
            equipment  = str(row.get("equipment", "")),
            intensity  = float(row.get("intensity", 0.0)),
            sets       = int(row.get("sets", 3)),
            reps_value = abs(raw_reps),
            reps_unit  = "seconds" if raw_reps < 0 else "reps",
            level      = level_label,   
            program_title = str(row.get('title', ""))
        ))
    return exercises
