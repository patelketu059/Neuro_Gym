from __future__ import annotations
from dataclasses import dataclass, field
from typing import NamedTuple
import pandas as pd



@dataclass
class PeriodizationTemplate:
    training_level: str
    week_pcts:      list[float]
    rpe_curve:      list[float]
    amplitude:      float
    amp_source:     int           # number of OPL athletes contributing



@dataclass
class AthletePersona:
    athlete_id: str
    sex: str
    age: float
    bodyweight_kg: float
    weight_class_kg: float
    squat_peak_kg: float
    bench_peak_kg: float
    deadlift_peak_kg: float
    total_kg: float
    dots: float
    training_level: str
    primary_program: str = ""
    secondary_program: str = ""



@dataclass
class Exercise:
    name: str
    goal: str
    equipment: str
    intensity: float    
    sets: int
    reps_value: float   
    reps_unit: str      # "reps" or "seconds"
    level: str
    program_title: str = ""




@dataclass
class SessionLog:
    """One training session within a 12-week block."""  
    week:           int
    day_index:      int
    day_label:      str
    main_lift:      str
    main_lift_kg:   float
    main_lift_rpe:  float
    volume_pct:     float
    block_phase:    str    # accumulation / intensification / realisation / deload
    accessories:    list[Exercise]
 


@dataclass
class AthleteRecord:
    """Complete generated record for one athlete: persona + full 12-week block."""
    persona:  AthletePersona
    sessions: list[SessionLog]
    opl_row_index: int = -1
 
 
class GymData(NamedTuple):

    df:             pd.DataFrame
    strength_goals: set
    barbell_equip:  set