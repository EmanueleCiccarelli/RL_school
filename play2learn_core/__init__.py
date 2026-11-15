"""
Play2Learn Core Architecture.
Motore e orchestratore unificato di Reinforcement Learning per ambienti educativi
(LunarLander, HillClimbRacing, ecc.).
"""

import os

# Configurazione threading all'avvio per evitare contesa CPU e oversubscription
for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(var, "1")

__version__ = "1.0.0"
