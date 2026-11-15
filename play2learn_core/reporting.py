"""
Modulo Core di Reporting e Comparazione Esperimenti (Play2Learn).
Generazione schede tecniche in PDF ad alta risoluzione, HTML standalone e Markdown.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


def serialize_eval_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Rimuove in modo ricorsivo oggetti `Path` e strutture non serializzabili
    da inserire nei metadati JSON della run.
    """
    clean: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, Path):
            continue
        if isinstance(v, (int, float, str, bool)):
            clean[k] = v
        elif isinstance(v, list):
            clean_list = [x for x in v if isinstance(x, (int, float, str, bool))]
            if len(clean_list) == len(v):
                clean[k] = clean_list
        elif isinstance(v, dict):
            sub = serialize_eval_metrics(v)
            if sub:
                clean[k] = sub
    return clean


def load_reward_fn_from_source(source_code: str) -> Callable | None:
    """
    Carica ed esegue dinamicamente il codice sorgente di una funzione di reward,
    iniettando automaticamente `np` e `math` nel namespace locale per prevenire NameError.
    """
    if not source_code or source_code.startswith("#"):
        return None

    scope: dict[str, Any] = {
        "np": np,
        "math": math,
        "numpy": np,
    }
    try:
        exec(source_code, scope)
        for key in ("custom_reward", "reward_function", "reward_fn"):
            if key in scope and callable(scope[key]):
                return scope[key]

        # Cerca la prima funzione definita
        for k, v in scope.items():
            if callable(v) and not k.startswith("_") and k not in ("np", "math", "numpy"):
                return v
    except Exception as exc:
        print(f"[Play2Learn] Impossibile ricostruire la reward function: {exc}")

    return None


def sanitize_report_text(text: str) -> str:
    """Esegue l'escape sicuro di caratteri HTML per evitare corruzioni di rendering."""
    return html.escape(str(text))
