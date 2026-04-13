from __future__ import annotations
import random
import pandas as pd

from config.settings import (
    BLOCK_WEEKS,
    SESSIONS_PER_WEEK,
    SESSION_FOCUS,
    PHASE_BOUNDARIES,
    ACCESSORIES_PER_SESSION,
    OHP_BENCH_RATIO
)

from pipeline.dataset.custom_dataclasses import (
    AthletePersona,
    AthleteRecord,
    PeriodizationTemplate,
    SessionLog
)

from pipeline.dataset.opl_loader import _classify_DOTS
from pipeline.dataset.gym_600k_loader import query_accessories, AccessoryPools




def _week_to_phase(week: int) -> str:
    for phase, weeks in PHASE_BOUNDARIES.items():
        if week in weeks:
            return phase
    return "deload"



def _main_lift_for_day(day_index: int, persona: AthletePersona) -> tuple[str, float]:
    ohp_peak = persona.bench_peak_kg * OHP_BENCH_RATIO[persona.training_level]
    return {
        0: ("Squat",    persona.squat_peak_kg),
        1: ("Bench",    persona.bench_peak_kg),
        2: ("Deadlift", persona.deadlift_peak_kg),
        3: ("OHP",      ohp_peak),
    }[day_index % 4]


def sample_athlete_persona(opl_df: pd.DataFrame, 
                           athlete_id: str, 
                           used_indicies: set) -> AthletePersona:
    
    persona_df = opl_df.drop(
        columns = [c for c in ['Name', 'Date'] if c in opl_df.columns]
    )
    n = len(persona_df)
    seed = int.from_bytes(athlete_id.encode(), "big") % (2 ** 31)
    row = persona_df.iloc[random.Random(seed).randint(0, len(persona_df) - 1)]
    index = random.Random(seed).randint(0, n - 1)
    dots = float(row['Dots'])

    attempts = 0
    while index in used_indicies:
        index = (index + 1) % n
        attempts += 1 
        if attempts >= n:
            raise RuntimeError(
                f"OPL dataset Exhausted - all rows are assigned"
                f"Reduce -n-athletes "
            )

    row = persona_df.iloc[index]
    dots = float(row['Dots'])

    persona = AthletePersona(
        athlete_id = athlete_id,
        sex = str(row['Sex']),
        age = float(row['Age']),
        bodyweight_kg = float(row['BodyweightKg']),
        weight_class_kg = float(row['WeightClassKg']),
        squat_peak_kg = float(row['Best3SquatKg']),
        bench_peak_kg = float(row['Best3BenchKg']),
        deadlift_peak_kg = float(row['Best3DeadliftKg']),
        total_kg = float(row['TotalKg']),
        dots = float(row['Dots']),
        training_level = _classify_DOTS(dots)
    )

    return persona, index



def build_training_block(
        persona: AthletePersona,
        pools: AccessoryPools,
        template: PeriodizationTemplate,
        n_weeks: int = BLOCK_WEEKS,
        sessions_per_week: int = SESSIONS_PER_WEEK,
        seed: int = 0
) -> list[SessionLog]:

    sessions = []
    for week in range(1, n_weeks + 1):
        vol_pct = template.week_pcts[week - 1]
        rpe = template.rpe_curve[week - 1]
        block_phase = _week_to_phase(week)


        for day_index in range(sessions_per_week):
            lift_name, competition_1rm = _main_lift_for_day(day_index, persona)
            block_target_kg = competition_1rm * (1.0 + template.amplitude)
            working_kg = round(block_target_kg * vol_pct / 2.5) * 2.5
            working_kg = min(working_kg, competition_1rm - 2.5)

            sessions.append(SessionLog(
                week = week,
                day_index = day_index,
                day_label = SESSION_FOCUS[day_index]['label'],
                main_lift = lift_name,
                main_lift_kg = working_kg,
                main_lift_rpe = rpe,
                volume_pct = vol_pct,
                block_phase = block_phase,
                accessories = query_accessories(
                    pools, 
                    training_level = persona.training_level,
                    day_index = day_index,
                    n = ACCESSORIES_PER_SESSION,
                    seed = seed + week * 100 + day_index,
                    primary_program = persona.primary_program,
                    secondary_program = persona.secondary_program
                )
            ))

    return sessions



def generate_one_athlete(
        athlete_id: str,
        opl_df: pd.DataFrame,
        pools: AccessoryPools,
        templates: dict[str, PeriodizationTemplate],
        used_indices: set
) -> AthleteRecord:
    
    persona, opl_index = sample_athlete_persona(opl_df, athlete_id, used_indices)
    template = templates[persona.training_level]
    sessions = build_training_block(
        persona,
        pools,
        template,
        seed = int.from_bytes(athlete_id.encode(), "big") % (2 ** 31)
    )

    return AthleteRecord(persona = persona,
                         sessions = sessions), opl_index