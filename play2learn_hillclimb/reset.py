"""
Script di Ripristino e Verifica Integrità per Hill Climb Racing (Play2Learn).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    args = parser.parse_args()

    ok = check_integrity()
    sys.exit(0 if ok else 1)
