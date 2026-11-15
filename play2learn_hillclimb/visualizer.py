"""
Modulo di Visualizzazione Grafica e Telemetria per Hill Climb Racing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def mostra_mappa_ambiente() -> None:
    """Mostra un diagramma concettuale didattico della vettura, del terreno e delle osservazioni."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0f172a")

    # Disegna profilo terreno collinare
    x = np.linspace(0, 30, 200)
    y = 2.0 + 1.2 * np.sin(x * 0.4) + 0.5 * np.cos(x * 0.9)
    ax.fill_between(x, -2, y, color="#582300", alpha=0.9, label="Terreno (Dirt)")
    ax.plot(x, y, color="#22c55e", lw=3.5, label="Erba (Grass - Trazione)")

    # Disegna sagoma auto stilizzata sul terreno
    car_x = 10.0
    car_y = 2.0 + 1.2 * np.sin(car_x * 0.4) + 0.5 * np.cos(car_x * 0.9) + 0.8
    ax.plot([car_x - 1.2, car_x + 1.2], [car_y, car_y], color="#ef4444", lw=7, label="Telaio (Chassis)")
    # Ruote
    ax.scatter([car_x - 1.0, car_x + 1.0], [car_y - 0.4, car_y - 0.4], s=180, color="#1e293b", edgecolors="#38bdf8", lw=2, zorder=5, label="Ruote (Wheels)")
    # Pilota
    ax.scatter([car_x - 0.3], [car_y + 0.7], s=140, color="#fde047", edgecolors="#eab308", lw=2, zorder=5, label="Testa Pilota (Impatto = Schianto!)")

    # Annotazioni didattiche
    ax.annotate("obs[0]: Distanza X\nobs[1]: Quota Y", xy=(car_x, car_y), xytext=(car_x - 4.5, car_y + 2.5),
                arrowprops=dict(facecolor="#38bdf8", shrink=0.08, width=1.5, headwidth=6),
                color="#38bdf8", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#0284c7"))

    ax.annotate("obs[4]: Inclinazione\nobs[2]: Velocità VX", xy=(car_x + 0.8, car_y), xytext=(car_x + 2.5, car_y + 2.2),
                arrowprops=dict(facecolor="#f59e0b", shrink=0.08, width=1.5, headwidth=6),
                color="#f59e0b", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#d97706"))

    ax.annotate("obs[6, 7]: Contatto Ruote", xy=(car_x + 1.0, car_y - 0.4), xytext=(car_x + 2.8, car_y - 1.5),
                arrowprops=dict(facecolor="#10b981", shrink=0.08, width=1.5, headwidth=6),
                color="#10b981", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#059669"))

    ax.set_xlim(-1, 25)
    ax.set_ylim(-1, 6)
    ax.set_title("🚗 Mappa dell'Ambiente e Telemetria: Hill Climb Racing", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Distanza di Avanzamento (metri)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Quota Terreno", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#64748b")
    ax.grid(True, linestyle="--", alpha=0.2, color="#475569")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_hill_telemetry(eval_history: list[dict[str, Any]], output_path: Path | str | None = None) -> Path | None:
    """Genera il grafico multi-pannello ad alta risoluzione delle prove di valutazione."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor("#ffffff")

    distances = [ep.get("distance", ep.get("score", 0.0)) for ep in eval_history]
    speeds = [ep.get("max_speed", 0.0) for ep in eval_history]
    airtimes = [ep.get("airtime_pct", 0.0) for ep in eval_history]
    ep_indices = list(range(1, len(eval_history) + 1))

    # 1. Distanza percorsa
    ax1 = axes[0, 0]
    ax1.bar(ep_indices, distances, color="#3b82f6", alpha=0.85, edgecolor="#1d4ed8")
    ax1.axhline(800.0, color="#10b981", linestyle="--", lw=1.8, label="Obiettivo (800 m)")
    ax1.set_title("Distanza Raggiunta per Episodio", fontweight="bold", fontsize=11, color="#1e293b")
    ax1.set_xlabel("Episodio di Verifica", fontsize=9)
    ax1.set_ylabel("Distanza (metri)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.5)

    # 2. Velocità di punta
    ax2 = axes[0, 1]
    ax2.plot(ep_indices, speeds, marker="o", color="#f59e0b", lw=2.0)
    ax2.set_title("Velocità Massima Registrata (m/s)", fontweight="bold", fontsize=11, color="#1e293b")
    ax2.set_xlabel("Episodio di Verifica", fontsize=9)
    ax2.set_ylabel("Velocità (m/s)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Percentuale di Airtime
    ax3 = axes[1, 0]
    ax3.plot(ep_indices, airtimes, marker="s", color="#8b5cf6", lw=2.0)
    ax3.set_title("Percentuale Tempo in Volo (Airtime %)", fontweight="bold", fontsize=11, color="#1e293b")
    ax3.set_xlabel("Episodio di Verifica", fontsize=9)
    ax3.set_ylabel("Airtime (%)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Distribuzione esiti
    ax4 = axes[1, 1]
    successes = sum(1 for ep in eval_history if ep.get("is_success", False))
    crashes = sum(1 for ep in eval_history if ep.get("is_crashed", False))
    timeouts = sum(1 for ep in eval_history if ep.get("is_timeout", False))
    labels = ["Successo (>=800m)", "Schianto/Ribaltamento", "Blocco/Timeout"]
    counts = [successes, crashes, timeouts]
    colors = ["#10b981", "#ef4444", "#f59e0b"]
    ax4.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140,
            textprops={"fontsize": 9, "color": "#1e293b"})
    ax4.set_title("Distribuzione Esiti su 20 Verifiche", fontweight="bold", fontsize=11, color="#1e293b")

    plt.tight_layout()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(p), dpi=160, bbox_inches="tight")
        plt.close(fig)
        return p

    plt.show()
    plt.close(fig)
    return None
