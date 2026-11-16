"""
Script di Ripristino e Verifica Integrità per Hill Climb Racing (Play2Learn).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def pulisci_runs() -> None:
    """Elimina tutte le run precedenti di Hill Climb e ricrea la struttura vergine."""
    pkg_dir = Path(__file__).resolve().parent
    runs_dir = pkg_dir / "runs"
    if runs_dir.exists():
        for item in runs_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                item.unlink(missing_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    fav_dir = runs_dir / "favorites"
    fav_dir.mkdir(parents=True, exist_ok=True)
    (fav_dir / "index.json").write_text("[]\n", encoding="utf-8")
    print("✅ Cartella play2learn_hillclimb/runs/ ripristinata allo stato iniziale.")


def check_integrity() -> bool:
    """Verifica la presenza e l'integrità dei moduli del pacchetto."""
    pkg_dir = Path(__file__).parent
    required = [
        "__init__.py",
        "config.py",
        "environment.py",
        "agent.py",
        "evaluator.py",
        "diagnostics.py",
        "visualizer.py",
        "storage.py",
        "favorites.py",
        "reporting.py",
        "lab.py",
    ]
    missing = [f for f in required if not (pkg_dir / f).exists()]
    if missing:
        print(f"❌ File mancanti in {pkg_dir.name}: {missing}")
        return False
    print(f"🛡️  Verifica di sicurezza superata: tutti i moduli di {pkg_dir.name} sono intatti al 100%!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica e reset ambiente didattico Hill Climb Racing")
    parser.add_argument("--check", action="store_true", help="Esegue solo il controllo di integrità")
    parser.add_argument("--reset", action="store_true", help="Ripristina la cartella runs allo stato iniziale")
    args = parser.parse_args()

    if args.reset:
        pulisci_runs()

    ok = check_integrity()
    sys.exit(0 if ok else 1)
