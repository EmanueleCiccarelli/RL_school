"""
Play2Learn Hill Climb Racing: Laboratorio Didattico di Reinforcement Learning.
Costruito sopra l'engine fisico Gymnasium di alexzh3/hillclimbracing.
"""

from __future__ import annotations

from play2learn_hillclimb.config import ENV_ID, MISSIONS, TARGET_DISTANCE_SUCCESS
from play2learn_hillclimb.lab import (
    run_lab,
    LabResult,
    collauda_missione,
    confronta_run,
    mostra_mappa_ambiente,
    mostra_preferiti,
    carica_preferito,
    mostra_cronologia,
    mostra_cronologia_run,
    genera_report_da_timestamp,
    comprimi_archivio,
    comprimi_runs,
)
from play2learn_hillclimb.favorites import salva_nei_preferiti, LoadedFavorite
from play2learn_hillclimb.environment import (
    HillClimbRewardWrapper,
    classify_hill_climb_outcome,
)

__version__ = "0.2.0"

__all__ = [
    "ENV_ID",
    "MISSIONS",
    "TARGET_DISTANCE_SUCCESS",
    "run_lab",
    "LabResult",
    "collauda_missione",
    "confronta_run",
    "mostra_mappa_ambiente",
    "mostra_preferiti",
    "carica_preferito",
    "mostra_cronologia",
    "mostra_cronologia_run",
    "genera_report_da_timestamp",
    "comprimi_archivio",
    "comprimi_runs",
    "salva_nei_preferiti",
    "LoadedFavorite",
    "HillClimbRewardWrapper",
    "classify_hill_climb_outcome",
]
