from __future__ import annotations
import pandas as pd
from tqdm.auto import tqdm



def extract_coaching_texts(sessions_df: pd.DataFrame) -> dict[str, str]:
    texts: dict[str, str] = {}

    for aid, ath in tqdm(sessions_df.groupby('athlete_id'), desc = "Extracting Coaching texts"):
        r = ath.iloc[0]
        level               = str(r.get('training_level', ''))
        dots                = float(r.get('dots', ''))
        sex                 = str(r.get('sex', ''))
        bodyweight          = float(r.get('bodyweight_kg', 0))
        primary_program     = str(r.get('primary_program', ''))

        squat_peak          = float(r.get('squat_peak_kg', 0))
        bench_peak          = float(r.get('bench_peak_kg', 0))
        deadlift_peak       = float(r.get('deadlift_peak_kg', 0))

        squat_rows          = ath[ath['main_lift'] == 'Squat'].sort_values('week')
        week1_main          = float(squat_rows.iloc[0]['main_lift_kg']) if not squat_rows.empty else 0
        peak_main           = float(squat_rows['main_lift_kg'].max()) if not squat_rows.empty else 0
        rpe_min             = float(ath['main_lift_rpe'].min())
        rpe_max             = float(ath['main_lift_rpe'].max())

        text = (
            f"Athlete {aid} · {level.capitalize()} powerlifter · {sex} · "
            f"{bodyweight:.1f}kg bodyweight · Dots {dots:.1f}\n"
            f"Competition lifts: Squat {squat_peak:.1f}kg  Bench {bench_peak:.1f}kg  "
            f"Deadlift {deadlift_peak:.1f}kg\n"
            f"12-week block: opens at {week1_main:.1f}kg squat, peaks at {peak_main:.1f}kg "
            f"in week 11.\n"
            f"RPE progression: {rpe_min:.1f} → {rpe_max:.1f} across the block.\n"
            f"Primary program: {primary_program}."
        )

        texts[aid] = text
    
    return texts



def chunk_text(
        text : str,
        chunk_size : int = 1000,
        chunk_overlap : int = 100
) -> list[str]:
    
    text = text.strip()
    if not text: return []


    chunks : list[str] = []
    start = 0

    while start < len(text):
        end  = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            last_space = chunk.rfind(" ")
            if last_space > chunk_overlap:
                end = start + last_space
                chunk = text[start:end]

        chunks.append(chunk)
        start = end - chunk_overlap

        if end >= len(text):
            break

    return [c for c in chunks if c]


_PHASE_LABEL = {
    "accumulation":    "Accumulation",
    "intensification": "Intensification",
    "realisation":     "Realisation",
    "deload":          "Deload",
}


def session_to_nl(
        df: pd.DataFrame,
        athlete_id : str,
        week : int,
):
    wk_rows = df[
        (df['athlete_id'] == athlete_id) &
        (df['week'] == week)
        ]
    
    if wk_rows.empty: return ""

    phase_raw = str(wk_rows['block_phase'].iloc[0])
    phase_label = _PHASE_LABEL.get(phase_raw, phase_raw.capitalize())

    lines : list[str] = [
        f"Athlete {athlete_id}  |   Week {week}  |  {phase_label}"
    ]

    lift_order = ['Squat', 'Bench', 'Deadlift', 'OHP']
    for lift in lift_order:
        # Get lift row
        row = wk_rows[wk_rows['main_lift'] == lift]
        # Check for empty row
        if row.empty: continue
        # get first row, extract kg, rpe, delta, pct
        r = row.iloc[0]
        kg = r['main_lift_kg']
        rpe = r['main_lift_rpe']
        delta = r.get('main_lift_delta_kg', None)
        pct = r.get('main_lift_pct_of_peak', None)

        delta_str = ""

        # Check if delta exists and not 0
        if pd.notna(delta) and delta != 0:
            sign = "+" if delta > 0 else ""
            delta_str = f" ({sign}{delta:.1f}kg vs prev week)"

        pct_str = f"  [{pct*100:.1f}% of 1RM]" if pd.notna(pct) else ""
        lines.append(
            f"{lift}: {kg:.1f}kg{delta_str}  |  RPE {rpe:.1f}{pct_str}"
        )
    # print(lines)
    # print(wk_rows['day_label'])
    day_order = ['Lower A', 'Upper A', 'Lower B', 'Upper B']
    for day_label in day_order:
        # get label row
        row = wk_rows[wk_rows['day_label'] == day_label]
        if row.empty: continue
        r = row.iloc[0]


        names = [x.strip() for x in str(r.get("accessories",            "")).split("|")]
        sets_ = [x.strip() for x in str(r.get("accessory_sets",         "")).split("|")]
        reps_ = [x.strip() for x in str(r.get("accessory_reps",         "")).split("|")]
        units_ = [x.strip() for x in str(r.get("accessory_reps_unit",   "")).split("|")]


        acc_parts : list[str] = []
        for i, name in enumerate(names):
            s = sets_[i]    if i < len(sets_) else "?"
            rep = reps_[i]  if i < len(reps_) else "?"
            u = units_[i]  if i < len(units_) else "?"

            try: 
                rv = float(rep)
                r_fmt = f"{int(rv)}" if rv == int(rv) else f"{rv:.1f}"
            except:
                r_fmt = rep
            sfx = "s" if u == 'seconds' else ""
            acc_parts.append(f"{name} {s} x {r_fmt}{sfx}")


        if acc_parts:
            lines.append(f"Accessories ({day_label}): {' | '.join(acc_parts)}")
    # print(lines)
    return "\n".join(lines)


#######################################################################################################



def optimized_session_to_nl(
        df: pd.DataFrame,
        athlete_id : str,
        week : int,
):
    wk_rows = df[
        (df['athlete_id'] == athlete_id) &
        (df['week'] == week)
        ]
    
    if wk_rows.empty: return ""


    records = wk_rows.to_dict('records')
    first_record = records[0]

    phase_raw = str(first_record.get('block_phase', ""))
    phase_label = _PHASE_LABEL.get(phase_raw, phase_raw.capitalize())

    lines : list[str] = [
        f"Athlete {athlete_id}  |   Week {week}  |  {phase_label}"
    ]

    lift_order = ['Squat', 'Bench', 'Deadlift', 'OHP']

    lift_lookup = {r['main_lift']: r for r in records if r.get('main_lift')}
    day_lookup = {r['day_label']: r for r in records if r.get('day_label')}


    for lift in lift_order:
        r = lift_lookup.get(lift)
        # Get lift row
        # row = wk_rows[wk_rows['main_lift'] == lift]
        # Check for empty row
        if not r: continue
        # get first row, extract kg, rpe, delta, pct
        # r = row.iloc[0]
        kg = r['main_lift_kg']
        rpe = r['main_lift_rpe']
        delta = r.get('main_lift_delta_kg', None)
        pct = r.get('main_lift_pct_of_peak', None)

        delta_str = ""

        # Check if delta exists and not 0
        if delta and delta != 0:
            sign = "+" if delta > 0 else ""
            delta_str = f" ({sign}{delta:.1f}kg vs prev week)"

        pct_str = f"  [{pct*100:.1f}% of 1RM]" if pd.notna(pct) else ""
        lines.append(
            f"{lift}: {kg:.1f}kg{delta_str}  |  RPE {rpe:.1f}{pct_str}"
        )
    # print(lines)
    # print(wk_rows['day_label'])
    day_order = ['Lower A', 'Upper A', 'Lower B', 'Upper B']
    for day_label in day_order:
        # get label row
        row = wk_rows[wk_rows['day_label'] == day_label]
        if row.empty: continue
        r = row.iloc[0]


        names = str(r["accessories"]).split("|")
        sets_ = str(r.get("accessory_sets", "")).split("|")
        reps_ = str(r.get("accessory_reps", "")).split("|")
        units_ = str(r.get("accessory_reps_unit", "")).split("|")


        acc_parts : list[str] = []
        for i, name in enumerate(names):
            name = name.strip()
            if not name: continue
            
            s = sets_[i]    if i < len(sets_) else "?"
            rep = reps_[i]  if i < len(reps_) else "?"
            u = units_[i]  if i < len(units_) else "?"

            try: 
                rv = float(rep)
                r_fmt = f"{int(rv)}" if rv == int(rv) else f"{rv:.1f}"
            except:
                r_fmt = rep
            sfx = "s" if u == 'seconds' else ""
            acc_parts.append(f"{name} {s} x {r_fmt}{sfx}")


        if acc_parts:
            lines.append(f"Accessories ({day_label}): {' | '.join(acc_parts)}")
    # print(lines)
    return "\n".join(lines)







def build_all_nl_strings(
        session_df: pd.DataFrame,
) -> list[dict]:
    
    records: list[dict] = []

    for athlete_id in tqdm(session_df['athlete_id'].unique()):
    
        athlete = session_df[session_df['athlete_id'] == athlete_id]

        meta = {
            'athlete_id': athlete_id,
            'training_level': session_df['training_level'].iloc[0],
            'dots': float(session_df['dots'].iloc[0]),
            'opl_row_index': int(athlete['opl_row_index'].iloc[0]) 
                        if 'opl_row_index' in athlete.columns else -1,
            'primary_program': str(athlete['primary_program'].iloc[0])
                        if 'primary_program' in athlete.columns else ""
        }
        # print(len(sorted(athlete['week'].unique())))
        for week in sorted(athlete['week'].unique()):
            phase = str(athlete[athlete['week'] == week]['block_phase'].iloc[0])
            text = session_to_nl(session_df, athlete_id, int(week))
            if not text: continue
            records.append({
                **meta,
                'week': int(week),
                'block_phase': phase,
                'text': text
            })

    return records


def optimized_build_all_nl_strings(
        session_df: pd.DataFrame
) -> list[dict]:
    records: list[dict] = []

    for athlete_id, athlete_group in tqdm(session_df.groupby('athlete_id')):
        first_row = athlete_group.iloc[0]
        meta = {
            'athlete_id': athlete_id,
            'training_level': first_row['training_level'],
            'dots': float(first_row['dots']),
            'opl_row_index': int(first_row['opl_row_index']) 
                        if 'opl_row_index' in athlete_group.columns else -1,
            'primary_program': str(first_row['primary_program'])
                        if 'primary_program' in athlete_group.columns else ""
        }

        for week, week_group in athlete_group.groupby('week'):
            phase = str(week_group['block_phase'].iloc[0])
            text = optimized_session_to_nl(session_df, athlete_id, int(week))
            if not text: continue
            records.append({
                **meta,
                'week': int(week),
                'block_phase': phase,
                'text': text
            })

    return records