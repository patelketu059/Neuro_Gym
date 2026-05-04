from __future__ import annotations
import argparse
import json
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import(
    N_ATHLETES,
    CHECKPOINT_THRESHOLD,
    OPL_PATH,
    GYM_EX_PATH,
    GYM_PROG_PATH,
    SESSIONS_PATH,
    BLOCK_SUMMARY_PATH,
    CHECKPOINT_PATH, 
    OUT_DIR
)

N_WORKERS = min(os.cpu_count() or 1, 8)

from pipeline.dataset.opl_loader import get_opl_dataset, _classify_DOTS
from pipeline.dataset.gym_600k_loader import get_gym_dataset, precompute_accessory_pools, build_program_catalog, select_program
from pipeline.dataset.periodization import build_periodization_templates
from pipeline.dataset.athlete_generator import generate_one_athlete
from pipeline.dataset.export import records_to_block_summary_df, records_to_session_df
from pipeline.dataset.custom_dataclasses import AthleteRecord


def load_last_checkpoint(path: Path) -> set[str]:
    if not path.is_file():
        return set(), set()
    
    with open(path) as f:
        data = json.load(f)

    
    completed = set(data.get('completed', []))
    used_indices = set(data.get("used_opl_indices", []))
    print(f"[INFO-CHECKPOINT] Resuming - {len(completed)} : athletes already generated."
          f"{len(used_indices)} OPL rows reserved.")
    return completed, used_indices
    

def save_checkpoint(path: Path, completed: set[str], used_indices: set) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({
            'completed': sorted(completed),
            'used_opl_indices': sorted(used_indices)},
            f)

    tmp.replace(path)

def clear_checkpoint(path: Path) -> None:
    if path.is_file():
        path.unlink()
        print("[INFO CHECKPOINT] Cleared existing checkpoint.")







def _pre_assign_opl_indices(
    athlete_ids: list,
    n_opl_rows: int,
    existing_used: set,
) -> dict:

    import random
    assignments: dict = {}
    used = set(existing_used)
    for aid in athlete_ids:
        seed = int.from_bytes(aid.encode(), "big") % (2 ** 31)
        idx  = random.Random(seed).randint(0, n_opl_rows - 1)
        attempts = 0
        while idx in used:
            idx = (idx + 1) % n_opl_rows
            attempts += 1
            if attempts >= n_opl_rows:
                raise RuntimeError(
                    f"OPL dataset exhausted after {len(used):,} assignments. "
                    f"Reduce --n-athletes or use a larger OPL dataset."
                )
        used.add(idx)
        assignments[aid] = idx
    return assignments
 
 
def _generate_worker(
        athlete_id: str, 
        opl_row: dict,
        opl_row_index: int, 
        pools,
        template_data: dict,
        program_catalog: dict
        ):

    from pipeline.dataset.custom_dataclasses import AthletePersona, AthleteRecord, PeriodizationTemplate
    from pipeline.dataset.athlete_generator import build_training_block
    from pipeline.dataset.opl_loader import _classify_DOTS
    from pipeline.dataset.gym_600k_loader import select_program

    try:
        dots    = float(opl_row.get("Dots", 0) or 0)
        seed = int.from_bytes(athlete_id.encode(), "big") % (2 ** 31)
        primary_program = select_program(
                _classify_DOTS(dots), program_catalog, seed
            )
        persona = AthletePersona(
            athlete_id       = athlete_id,
            sex              = str(opl_row.get("Sex", "M")),
            age              = float(opl_row.get("Age", 25) or 25),
            bodyweight_kg    = float(opl_row.get("BodyweightKg", 80) or 80),
            weight_class_kg  = str(opl_row.get("WeightClassKg", "83")),
            squat_peak_kg    = float(opl_row["Best3SquatKg"]),
            bench_peak_kg    = float(opl_row["Best3BenchKg"]),
            deadlift_peak_kg = float(opl_row["Best3DeadliftKg"]),
            total_kg         = float(opl_row.get("TotalKg", 0) or 0),
            dots             = dots,
            training_level   = _classify_DOTS(dots),
            primary_program = primary_program,
            secondary_program = select_program(
                _classify_DOTS(dots), program_catalog, seed + 1,
                exclude = primary_program
                )
            )

        tmpl    = PeriodizationTemplate(**template_data[persona.training_level])
        sessions = build_training_block(persona, pools, tmpl, seed = seed)
        record = AthleteRecord(
            persona = persona,
            sessions = sessions,
            opl_row_index = opl_row_index
        )
        return athlete_id, record
    except Exception as exc:
        return athlete_id, None
    


def run(
        n_athletes: int,
        resume: bool,
        opl_path: Path,
        gym_ex_path: Path,
        gym_prog_path: Path,
        sessions_out: Path,
        summary_out: Path,
        checkpoint_path: Path
) -> None:
    
    OUT_DIR.mkdir(parents = True, exist_ok = True)


    print("-" * 80)
    print("\n[INFO-OPL] Loading OpenPowerLifting...")
    opl_df = get_opl_dataset(opl_path)


    print("\n[INFO-GYM] - Loading 600k Gym Dataset... ")
    gym_data = get_gym_dataset(gym_ex_path, gym_prog_path)
    gym_df = gym_data.df 
    strength_goals = gym_data.strength_goals
    barbell_equip = gym_data.barbell_equip

    print("\n[INFO-PERIODIZATION] Building Periodization template...")
    templates = build_periodization_templates(opl_df)

    if not resume:
        clear_checkpoint(checkpoint_path)

    if resume:
        completed, used_indices = load_last_checkpoint(checkpoint_path)
    else:
        completed, used_indices = set(), set()


    all_athlete_ids = [f"athlete_{i:05d}" for i in range(n_athletes)]
    remaining = [aid for aid in all_athlete_ids if aid not in completed]

    if not remaining:
        print(f"[checkpoint] All {n_athletes} athletes already generated")
        print("Run with --no-resume to regenerate from scratch.")
        return
    
    opl_capacity = len(opl_df)
    print(f"\n[OPL-Capacity] MAX OPL Capacity - {opl_capacity} | Keep n-athletes under this")


    print(f"\n[OPTIMIZATION] - Pre-Computing Accessory Pools...")
    pools = precompute_accessory_pools(gym_df, strength_goals, barbell_equip)

    program_catalog = build_program_catalog(gym_df, strength_goals)
    total_programs = sum(len(v) for v in program_catalog.values())
    print(f"  {total_programs} program-level slots across 4 training levels")

    print(f"\n Pre-assigning OPL Row indices - Parallelization")
    assignments = _pre_assign_opl_indices(remaining, len(opl_df), used_indices)
    used_indices.update(assignments.values())
    print(f"  {len(assignments):,} assignments resolved, "
          f"{len(used_indices):,} OPL rows reserved total\n")



    template_data = {
        level: {
            "training_level": t.training_level,
            "week_pcts":      t.week_pcts,
            "rpe_curve":      t.rpe_curve,
            "amplitude":      t.amplitude,
            "amp_source":     t.amp_source,
        }
        for level, t in templates.items()
    }

    persona_cols  = ["Sex", "Age", "BodyweightKg", "WeightClassKg",
                     "Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg", "TotalKg", "Dots"]
    opl_records   = opl_df[[c for c in persona_cols if c in opl_df.columns]]

    sessions_write_header = not (resume and sessions_out.is_file())
    summary_write_header  = not (resume and summary_out.is_file())
 
    batch: list[AthleteRecord] = []
    total_sessions_written = 0
    total_summary_written  = 0


    def _flush_batch(batch):
        nonlocal sessions_write_header, summary_write_header
        if not batch:
            return 0, 0
        bs = records_to_session_df(batch)
        bb = records_to_block_summary_df(batch)
        bs.to_csv(sessions_out, mode="a", header=sessions_write_header, index=False)
        bb.to_csv(summary_out,  mode="a", header=summary_write_header,  index=False)
        sessions_write_header = False
        summary_write_header  = False
        return len(bs), len(bb)
    


    print(f"\nGenerating with {N_WORKERS} parallel workers ...")
    with ProcessPoolExecutor(max_workers = N_WORKERS) as executor:
        futures = {
            executor.submit(
                _generate_worker,
                aid,
                opl_records.iloc[assignments[aid]].to_dict(),
                assignments[aid],
                pools,
                template_data,
                program_catalog
            ): aid
            for aid in remaining
        }
 
        i = 0
        for future in as_completed(futures):
            i += 1  
            aid = futures[future]
            try:
                _, record = future.result()
                if record is not None:
                    batch.append(record)
                    completed.add(aid)
                else:
                    print(f"  [SKIP] {aid}: worker returned None.")
            except Exception as exc:
                print(f"  [ERROR] {aid}: {exc}")
 
            if i % CHECKPOINT_THRESHOLD == 0 or i == len(remaining):
                print(f"  [{i:>6}/{len(remaining)}]  batch={len(batch)}  "
                      f"unique OPL rows={len(used_indices):,}  checkpointing ...")
                ns, nb = _flush_batch(batch)
                total_sessions_written += ns
                total_summary_written  += nb
                save_checkpoint(checkpoint_path, completed, used_indices)
                batch = []
 
    print(f"\n{'=' * 60}")
    print(f"Done. {len(completed):,} / {n_athletes} athletes generated.")
    print(f"  Unique OPL rows used: {len(used_indices):,}")
    print(f"\nSessions:      {sessions_out}  (~{total_sessions_written:,} rows this run)")
    print(f"Block summary: {summary_out}  (~{total_summary_written:,} rows this run)")
    print(f"Checkpoint:    {checkpoint_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate master powerlifting dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-athletes",  type = int,  default = N_ATHLETES)
    parser.add_argument("--n-workers",   type = int,  default = N_WORKERS,
                        help="Parallel workers for generation (default: CPU count, max 8)")
    parser.add_argument("--resume",      action="store_true",  default=True,
                        help="Resume from checkpoint if it exists.")
    parser.add_argument("--no-resume",   action="store_true",  default=False,
                        help="Start fresh — clears existing checkpoint.")
    parser.add_argument("--out-sessions", type = str, default = str(SESSIONS_PATH))
    parser.add_argument("--out-summary",  type = str, default = str(BLOCK_SUMMARY_PATH))
    parser.add_argument("--opl",          type = str, default = str(OPL_PATH))
    parser.add_argument("--gym-ex",       type = str, default = str(GYM_EX_PATH))
    parser.add_argument("--gym-prog",     type = str, default = str(GYM_PROG_PATH))
    parser.add_argument("--checkpoint",   type = str, default = str(CHECKPOINT_PATH))
    return parser.parse_args()
 
if __name__ == "__main__":
    args   = parse_args()
    resume = args.resume and not args.no_resume
    N_WORKERS = args.n_workers

 
    run(
        n_athletes     = args.n_athletes,
        resume         = resume,
        opl_path       = Path(args.opl),
        gym_ex_path    = Path(args.gym_ex),
        gym_prog_path  = Path(args.gym_prog),
        sessions_out   = Path(args.out_sessions),
        summary_out    = Path(args.out_summary),
        checkpoint_path = Path(args.checkpoint),
    )
 