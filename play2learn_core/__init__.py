"""
Play2Learn Core Architecture.
Motore e orchestratore unificato di Reinforcement Learning per ambienti educativi
(LunarLander, HillClimbRacing, ecc.).
"""

# Fix preventivo e definitivo per PyTorch / Sympy in Python 3.13 (Google Colab)
import sys
try:
    import sympy
    import sympy.logic
    import sympy.logic.boolalg
    sympy.logic = sympy.logic
    sympy.logic.boolalg = sympy.logic.boolalg
    sys.modules["sympy"].logic = sympy.logic
    sys.modules["sympy.logic"] = sympy.logic
    sys.modules["sympy.logic.boolalg"] = sympy.logic.boolalg
    import torch.utils._sympy.numbers
except Exception:
    pass

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
