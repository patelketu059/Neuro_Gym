from __future__ import annotations
import numpy as np
import pandas as pd

from config.settings import OHP_BENCH_RATIO, OPL_ROW_INDEX_COL, ACCESSORY_TITLES_COL, PRIMARY_PROGRAM_COL, SECONDARY_PROGRAM_COL
from pipeline.dataset.custom_dataclasses import AthleteRecord, SessionLog

def records_to_session_df(records: list[AthleteRecord]) -> pd.DataFrame:
    rows = []

    for record in records:
        persona = record.persona
        ohp_peak = persona.bench_peak_kg * OHP_BENCH_RATIO[persona.training_level]
        lift_peaks = {
            "Squat": persona.squat_peak_kg,
            "Bench": persona.bench_peak_kg,
            "Deadlift": persona.deadlift_peak_kg,
            "OHP": ohp_peak
        }

        base = {
            "athlete_id": persona.athlete_id,
            OPL_ROW_INDEX_COL: record.opl_row_index,
            PRIMARY_PROGRAM_COL: persona.primary_program,
            SECONDARY_PROGRAM_COL: persona.secondary_program,
            "sex": persona.sex,
            "age": persona.age if (persona.age is not None and persona.age == persona.age) else 25.0,
            "bodyweight_kg":    persona.bodyweight_kg,
            "weight_class_kg":  persona.weight_class_kg if (persona.weight_class_kg is not None) else persona.bodyweight_kg + 3.0,
            "squat_peak_kg":    persona.squat_peak_kg,
            "bench_peak_kg":    persona.bench_peak_kg,
            "deadlift_peak_kg": persona.deadlift_peak_kg,
            "total_kg":         persona.total_kg,
            "dots":             persona.dots,
            "training_level":   persona.training_level,
        }


        for s in record.sessions:
            peak_for_lift = lift_peaks.get(s.main_lift, s.main_lift_kg)
            rows.append({
                **base,
                "week":                  s.week,
                "day_index":             s.day_index,
                "day_label":             s.day_label,
                "block_phase":           s.block_phase,
                "main_lift":             s.main_lift,
                "main_lift_kg":          s.main_lift_kg,
                "main_lift_pct_of_peak": round(s.main_lift_kg / peak_for_lift, 3)
                                         if peak_for_lift > 0 else None,
                "main_lift_rpe":         s.main_lift_rpe,
                "volume_pct":            s.volume_pct,
                "accessories":           " | ".join(e.name            for e in s.accessories),
                ACCESSORY_TITLES_COL:    " | ".join(e.program_title   for e in s.accessories),
                "accessory_sets":        " | ".join(str(e.sets)       for e in s.accessories),
                "accessory_reps":        " | ".join(str(e.reps_value) for e in s.accessories),
                "accessory_reps_unit":   " | ".join(e.reps_unit       for e in s.accessories),
                "accessory_intensity":   " | ".join(str(e.intensity)  for e in s.accessories),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(['athlete_id', 'main_lift', 'week', 'day_index'])
    df['main_lift_delta_kg'] = (
        df.groupby(['athlete_id', 'main_lift'])['main_lift_kg']
        .diff()
        .round(2)
    )

    return df.sort_values(['athlete_id', 'week', 'day_index']).reset_index(drop = True)



def records_to_block_summary_df(records: list[AthleteRecord]) -> pd.DataFrame:
    rows = []
 
    for record in records:
        p = record.persona
        ohp_peak = p.bench_peak_kg * OHP_BENCH_RATIO[p.training_level]
        lift_peaks = {
            "Squat":    p.squat_peak_kg,
            "Bench":    p.bench_peak_kg,
            "Deadlift": p.deadlift_peak_kg,
            "OHP":      ohp_peak,
        }
 
        lift_sessions: dict[str, list[SessionLog]] = {}
        for s in record.sessions:
            lift_sessions.setdefault(s.main_lift, []).append(s)
 
        for lift, sessions in lift_sessions.items():
            representative_day = sessions[0].day_index
            weekly = sorted(
                [s for s in sessions if s.day_index == representative_day],
                key=lambda s: s.week,
            )
            if not weekly:
                continue
 
            kgs      = [s.main_lift_kg for s in weekly]
            weeks    = [s.week         for s in weekly]
            phases   = [s.block_phase  for s in weekly]
            peak_1rm = lift_peaks.get(lift, kgs[-1])
 
            peak_idx  = int(np.argmax(kgs))
            floor_idx = int(np.argmin(kgs))
 
            rows.append({
                "athlete_id":          p.athlete_id,
                OPL_ROW_INDEX_COL:     record.opl_row_index,
                PRIMARY_PROGRAM_COL:   p.primary_program,
                SECONDARY_PROGRAM_COL: p.secondary_program,
                "sex":                 p.sex,
                "training_level":      p.training_level,
                "dots":                p.dots,
                "lift":                lift,
                "competition_1rm_kg":  round(peak_1rm, 2),
                "week_1_kg":           kgs[0],
                "week_peak_kg":        kgs[peak_idx],
                "week_floor_kg":       kgs[floor_idx],
                "peak_week":           weeks[peak_idx],
                "floor_week":          weeks[floor_idx],
                "total_gain_kg":       round(kgs[peak_idx] - kgs[0], 2),
                "peak_pct_of_1rm":     round(kgs[peak_idx] / peak_1rm, 4)
                                       if peak_1rm > 0 else None,
                "block_phase_at_peak": phases[peak_idx],
                "weekly_kg_series":    " | ".join(str(k) for k in kgs),
            })
 
    return pd.DataFrame(rows).sort_values(["athlete_id", "lift"]).reset_index(drop=True)