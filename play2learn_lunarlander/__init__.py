"""
Play2Learn - Lunar Lander
Motore di Reinforcement Learning didattico per studenti di scuola secondaria.
Ottimizzato per Google Colab ed esecuzione locale, con tracciamento automatico delle run,
sezione Preferiti e reportistica tecnica dettagliata.
"""

from __future__ import annotations

import os

# Imposta variabili d'ambiente di threading PRIMA di caricare numpy, torch o matplotlib
# per evitare conflitti e sovraccaricamento del pool di thread OpenMP/MKL su CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
    genera_report_da_timestamp,
    genera_report_maturita,
    confronta_run,
)
from play2learn_lunarlander.storage import (
    list_all_runs,
    mostra_successi,
    elenca_successi,
    find_run_by_timestamp,
    elenca_cronologia_run,
    mostra_cronologia_run,
    comprimi_runs,
)
from play2learn_lunarlander.reset import reset_laboratorio
from play2learn_lunarlander.interactive import (
    gioca_single_player,
    gioca_vs_agente,
    gioca,
)

# Alias pubblico allineato a LabResult.salva_preferito
salva_preferito = salva_nei_preferiti

__version__ = "2.0.0"

__all__ = [
    "run_lab",
    "mostra_mappa_ambiente",
    "salva_preferito",
    "salva_nei_preferiti",
    "salva_video_preferito",
    "mostra_preferiti",
    "elenca_preferiti",
    "carica_preferito",
    "mostra_successi",
    "elenca_successi",
    "find_run_by_timestamp",
    "elenca_cronologia_run",
    "mostra_cronologia_run",
    "genera_report_esperimento",
    "genera_report_da_timestamp",
    "genera_report_maturita",
    "confronta_run",
    "dimostrazione_live",
    "list_all_runs",
    "reset_laboratorio",
    "gioca_single_player",
    "gioca_vs_agente",
    "gioca",
    "comprimi_runs",
    "MISSIONS",
    "ENV_ID",
    "LabResult",
]
