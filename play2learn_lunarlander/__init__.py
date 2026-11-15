"""
Play2Learn - Lunar Lander
Motore di Reinforcement Learning didattico per studenti di scuola secondaria.
Ottimizzato per Google Colab ed esecuzione locale, con tracciamento automatico delle run,
sezione Preferiti e reportistica tecnica dettagliata.
"""

from __future__ import annotations

from play2learn_lunarlander.config import MISSIONS, ENV_ID
from play2learn_lunarlander.visualizer import mostra_mappa_ambiente, plot_comparison
from play2learn_lunarlander.lab import run_lab, dimostrazione_live, LabResult
from play2learn_lunarlander.favorites import (
    salva_nei_preferiti,
    salva_video_preferito,
    mostra_preferiti,
    elenca_preferiti,
    carica_preferito,
)
from play2learn_lunarlander.reporting import (
    genera_report_esperimento,
    genera_report_maturita,
    confronta_run,
)
from play2learn_lunarlander.storage import list_all_runs, mostra_successi, elenca_successi
from play2learn_lunarlander.reset import reset_laboratorio

__version__ = "2.0.0"

__all__ = [
    "run_lab",
    "mostra_mappa_ambiente",
    "salva_nei_preferiti",
    "salva_video_preferito",
    "mostra_preferiti",
    "elenca_preferiti",
    "carica_preferito",
    "mostra_successi",
    "elenca_successi",
    "genera_report_esperimento",
    "genera_report_maturita",
    "confronta_run",
    "dimostrazione_live",
    "list_all_runs",
    "reset_laboratorio",
    "MISSIONS",
    "ENV_ID",
    "LabResult",
]
