from __future__ import annotations
import pandas as pd

from config.settings import (
    BLOCK_WEEKS,
    PERIODIZATION_SHAPE,
    LEVEL_SHAPE_PARAMS,
    RPE_FLOOR,
    RPE_CEILING
)


from pipeline.dataset.custom_dataclasses import PeriodizationTemplate
from pipeline.dataset.opl_loader import _classify_DOTS, _derive_opl_amplitude


def build_periodization_templates(
        opl_df: pd.DataFrame,
        n_weeks: int = BLOCK_WEEKS
) -> dict[str, PeriodizationTemplate]:
    
    amplitude = _derive_opl_amplitude(opl_df)
    opl_copy = opl_df.copy()
    opl_copy['training_level'] = opl_copy['Dots'].apply(_classify_DOTS)

    athletes_per_level = (
        opl_copy.groupby("training_level")['Name']
        .nunique().
        to_dict()
    )

    templates: dict[str, PeriodizationTemplate] = {}

    for level in ['novice', 'intermediate', 'advanced', 'elite']:
        amplitude_level = amplitude[level]
        floor = LEVEL_SHAPE_PARAMS[level]['floor']
        ceiling = LEVEL_SHAPE_PARAMS[level]['ceiling']
        

        week_pcts = [round(float(floor + s * (ceiling - floor)), 4)
                    for s in PERIODIZATION_SHAPE]

        rpe_curve = [
            round(float(RPE_FLOOR[level] + s * (RPE_CEILING - RPE_FLOOR[level])), 3)
            for s in PERIODIZATION_SHAPE
        ]
        templates[level] = PeriodizationTemplate(
            training_level = level,
            week_pcts = week_pcts,
            rpe_curve = rpe_curve,
            amplitude = amplitude_level,
            amp_source = athletes_per_level[level]
        )
    return templates
        