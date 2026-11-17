from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    from stable_baselines3 import PPO
    _SB3_AVAILABLE = True
except ImportError:
    PPO = Any  # type: ignore
    _SB3_AVAILABLE = False

from play2learn_core.favorites import BaseFavoritesManager
from play2learn_hillclimb.storage import BASE_RUNS_DIR

FAVORITES_DIR = BASE_RUNS_DIR / "favorites"
_mgr = BaseFavoritesManager(FAVORITES_DIR)


class LoadedFavorite:
    def __init__(self, model: Any, metadata: dict[str, Any], dir_path: Path, entry: dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.dir_path = dir_path
        self.entry = entry

    def __iter__(self):
        return iter((self.model, self.metadata))

    def __getitem__(self, item: str) -> Any:
        if item == "model":
            return self.model
        if item == "metadata":
            return self.metadata
        if item in self.metadata:
            return self.metadata[item]
        if item in self.entry:
            return self.entry[item]
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            return self[item]
        except KeyError:
            return default


def salva_nei_preferiti(
    run_dir: Path | str,
    nome: str,
    note: str = "",
    tag: str = "didattico",
) -> dict[str, Any]:
    return _mgr.salva_preferito(Path(run_dir), nome=nome, note=note, tag=tag)


salva_preferito = salva_nei_preferiti  # Alias consistente con Lunar Lander


def mostra_preferiti() -> None:
    entries = _mgr.read_index()
    if not entries:
        print("[play2learn] Nessun esperimento salvato nei preferiti.")
        return
    print("=" * 82)
    print(f"{'ID':<4} | {'NOME':<26} | {'MISSIONE':<20} | {'SUCC %':<8} | {'DIST MAX'}")
    print("-" * 82)
    for e in entries:
        print(
            f"{str(e.get('id', '')):<4} | "
            f"{str(e.get('nome', ''))[:25]:<26} | "
            f"{str(e.get('mission', ''))[:19]:<20} | "
            f"{float(e.get('success_rate', 0.0)):>6.1f}%  | "
            f"{float(e.get('max_distance', 0.0)):>6.1f}m"
        )
    print("=" * 82)


def carica_preferito(identificatore: int | str) -> Optional[LoadedFavorite]:
    try:
        loaded = _mgr.carica_preferito(identificatore)
    except ValueError as exc:
        print(str(exc))
        return None

    dir_path: Path = loaded["dir_path"]
    model_path = dir_path / "best_model.zip"
    if not model_path.exists():
        model_path = dir_path / "final_model.zip"

    model = None
    if model_path.exists() and _SB3_AVAILABLE:
        try:
            model = PPO.load(str(model_path), device="cpu")
        except Exception as exc:
            print(f"[play2learn] Avviso caricamento modello: {exc}")

    return LoadedFavorite(
        model=model,
        metadata=loaded["metadata"],
        dir_path=dir_path,
        entry=loaded["entry"],
    )
