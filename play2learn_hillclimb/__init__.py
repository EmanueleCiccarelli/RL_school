"""
Play2Learn - Hill Climb Racing.
Laboratorio di Reinforcement Learning didattico per Hill Climb Racing.
"""

from __future__ import annotations

import play2learn_core  # Assicura configurazione threading a monte

# Registrazione automatica dell'ambiente Gymnasium
import play2learn_hillclimb.engine
from play2learn_hillclimb.config import ENV_ID, MISSIONS
from play2learn_hillclimb.environment import (
    HillClimbRewardWrapper,
    classify_hill_climb_outcome,
)
from play2learn_hillclimb.favorites import (
    carica_preferito,
    mostra_preferiti,
    salva_nei_preferiti,
    salva_preferito,
)
from play2learn_hillclimb.interactive import (
    gioca,
    gioca_single_player,
    gioca_vs_agente,
)
from play2learn_hillclimb.lab import (
    LabResult,
    dimostrazione_live,
    run_lab,
)
from play2learn_hillclimb.reporting import (
    confronta_run,
    genera_report_da_timestamp,
    genera_report_esperimento,
)
from play2learn_hillclimb.storage import comprimi_runs
from play2learn_hillclimb.visualizer import mostra_mappa_ambiente

__all__ = [
    "run_lab",
    "LabResult",
    "dimostrazione_live",
    "carica_preferito",
    "salva_preferito",
    "salva_nei_preferiti",
    "mostra_preferiti",
    "confronta_run",
    "genera_report_da_timestamp",
    "genera_report_esperimento",
    "comprimi_runs",
    "gioca_single_player",
    "gioca_vs_agente",
    "gioca",
    "mostra_mappa_ambiente",
    "classify_hill_climb_outcome",
    "HillClimbRewardWrapper",
    "MISSIONS",
    "ENV_ID",
]
