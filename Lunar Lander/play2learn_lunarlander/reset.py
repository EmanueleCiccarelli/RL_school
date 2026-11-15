"""
Script di Reset per il Laboratorio di Reinforcement Learning (Lunar Lander).
Posizionato all'interno di play2learn_lunarlander/ per mantenere la root pulita.

Ripristina l'ambiente allo stato originale e pulito:
1. Svuota la cartella 'runs/' (eliminando tutte le prove, video, modelli e report di test).
2. Ricrea la struttura vergine 'runs/favorites/index.json'.
3. Azzera tutti gli output e contatori di esecuzione nel Notebook (.ipynb).
4. Rimuove la cache Python (__pycache__ e file .pyc).
"""

from __future__ import annotations

import os
import sys
import json
import shutil
from pathlib import Path


def get_root_dir() -> Path:
    """Restituisce la root directory del laboratorio (cartella genitore del package)."""
    current_dir = Path(__file__).resolve().parent
    # Se il file si trova dentro play2learn_lunarlander/, la root è il genitore
    if current_dir.name == "play2learn_lunarlander":
        return current_dir.parent
    return current_dir


def pulisci_notebook(nb_path: Path) -> int:
    """Azzera tutti gli output e execution_count delle celle del notebook."""
    if not nb_path.exists():
        print(f"⚠️  Notebook non trovato in: {nb_path.name}")
        return 0

    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cleared_count = 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    cell["outputs"] = []
                    cell["execution_count"] = None
                    cleared_count += 1

        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        size_kb = nb_path.stat().st_size / 1024
        print(f"✅ Notebook ripulito: {nb_path.name} ({size_kb:.1f} KB, azzerate {cleared_count} celle di output)")
        return cleared_count
    except Exception as e:
        print(f"❌ Errore durante la pulizia del notebook: {e}")
        return 0


def pulisci_runs(runs_dir: Path) -> None:
    """Elimina tutte le run precedenti e ricrea l'alberatura runs/favorites vergine."""
    if runs_dir.exists():
        for item in runs_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                item.unlink(missing_ok=True)
    else:
        runs_dir.mkdir(parents=True, exist_ok=True)

    # Ricrea la struttura minima pulita
    fav_dir = runs_dir / "favorites"
    fav_dir.mkdir(parents=True, exist_ok=True)
    (fav_dir / "index.json").write_text("[]\n", encoding="utf-8")
    print(f"✅ Cartella runs/ ripristinata allo stato iniziale (svuotata e pronta per nuove run)")


def pulisci_cache(root_dir: Path) -> None:
    """Rimuove tutte le cartelle __pycache__ e i file .pyc."""
    rimossi = 0
    for p in root_dir.rglob("__pycache__"):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            rimossi += 1
    for p in root_dir.rglob("*.pyc"):
        if p.is_file():
            p.unlink(missing_ok=True)
            rimossi += 1
    if rimossi > 0:
        print(f"✅ Cache Python rimossa ({rimossi} elementi eliminati)")


def reset_laboratorio(chiedi_conferma: bool = False) -> None:
    """Esegue il reset completo del laboratorio."""
    root_dir = get_root_dir()

    print("\n" + "=" * 70)
    print("  🧹 RESET DEL LABORATORIO: LUNAR LANDER")
    print("=" * 70)

    if chiedi_conferma:
        risposta = input("Sei sicuro di voler cancellare tutte le run e azzerare il notebook? [s/N]: ").strip().lower()
        if risposta not in ("s", "si", "y", "yes"):
            print("❌ Operazione annullata.")
            return

    # 1. Pulizia cartella runs
    pulisci_runs(root_dir / "runs")

    # 2. Pulizia notebook (sia versione Colab che versione Locale)
    pulisci_notebook(root_dir / "laboratorio_lunar_lander.ipynb")
    pulisci_notebook(root_dir / "play2learn_lunarlander" / "laboratorio_lunar_lander_locale.ipynb")

    # 3. Pulizia cache Python
    pulisci_cache(root_dir)

    print("=" * 70)
    print("  ✨ AMBIENTE COMPLETAMENTE RIPRISTINATO E PULITO!")
    print("  Il laboratorio è pronto per essere distribuito agli studenti.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    force = ("--force" in sys.argv) or ("-y" in sys.argv)
    reset_laboratorio(chiedi_conferma=not force)
