"""
Modulo Gestione Esperimenti Preferiti per Hill Climb Racing (Play2Learn).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from play2learn_core.favorites import BaseFavoritesManager
from play2learn_hillclimb.storage import get_default_runs_dir


def _get_manager() -> BaseFavoritesManager:
    fav_dir = get_default_runs_dir() / "favorites"
    return BaseFavoritesManager(fav_dir)


def salva_nei_preferiti(
    run_or_dir: Any,
    nome: str,
    note: str = "",
    tag: str = "didattico",
) -> dict[str, Any]:
    """Salva la run nei preferiti personali."""
    source_dir = None
    if isinstance(run_or_dir, dict):
        source_dir = Path(run_or_dir.get("checkpoint_dir", run_or_dir.get("run_dir", ".")))
    elif isinstance(run_or_dir, (str, Path)):
        source_dir = Path(run_or_dir)

    if not source_dir or not source_dir.exists():
        raise ValueError(f"[Play2Learn] Directory di run non trovata: {run_or_dir}")

    mgr = _get_manager()
    entry = mgr.salva_preferito(source_dir=source_dir, nome=nome, note=note, tag=tag)
    print(f"⭐ Esperimento salvato nei preferiti: #{entry['id']} - {entry['nome']}")
    return entry


salva_preferito = salva_nei_preferiti


def carica_preferito(fav_id_or_name: int | str) -> dict[str, Any]:
    """Carica un esperimento preferito dal catalogo."""
    mgr = _get_manager()
    return mgr.carica_preferito(fav_id_or_name)


def mostra_preferiti() -> list[dict[str, Any]]:
    """Stampa e restituisce l'elenco di tutti gli esperimenti preferiti salvati."""
    mgr = _get_manager()
    entries = mgr.read_index()
    if not entries:
        print("[play2learn] Nessun esperimento preferito salvato.")
        return []

    print("=" * 70)
    print(" ⭐ GALLERIA PREFERITI - HILL CLIMB RACING")
    print("=" * 70)
    for e in entries:
        print(f"#{e['id']:02d} | {e['nome'][:24]:24} | Distanza: {e.get('success_rate', 0.0):.0f}% | Data: {e['timestamp']}")
        if e.get("note"):
            print(f"     Note: {e['note']}")
    print("=" * 70)
    return entries
