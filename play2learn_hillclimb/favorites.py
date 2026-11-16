"""
Modulo Gestione Preferiti per Play2Learn Hill Climb Racing.
"""

from __future__ import annotations

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
    """Oggetto ibrido restituito da carica_preferito: supporta tupla, attributi e dizionario."""

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
        return self.entry.get(item)

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
    """Salva una run completata nella cartella dei preferiti."""
    return _mgr.salva_preferito(Path(run_dir), nome=nome, note=note, tag=tag)


def mostra_preferiti() -> None:
    """Visualizza l'elenco formattato dei modelli salvati nei preferiti."""
    _mgr.mostra_preferiti()


def carica_preferito(identificatore: int | str) -> Optional[LoadedFavorite]:
    """Carica il modello e i metadati di un preferito tramite ID numerico o nome/timestamp."""
    entry = _mgr.trova_per_id_o_nome(identificatore)
    if not entry:
        print(f"[play2learn] Nessun preferito trovato con identificatore: {identificatore}")
        return None

    dir_path = FAVORITES_DIR / entry["dir_name"]
    model_path = dir_path / "best_model.zip"
    if not model_path.exists():
        model_path = dir_path / "final_model.zip"

    model = None
    if model_path.exists() and _SB3_AVAILABLE:
        try:
            model = PPO.load(str(model_path))
        except Exception as e:
            print(f"[play2learn] Avviso caricamento modello: {e}")

    meta = {}
    meta_file = dir_path / "run_metadata.json"
    if meta_file.exists():
        import json
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return LoadedFavorite(model=model, metadata=meta, dir_path=dir_path, entry=entry)
