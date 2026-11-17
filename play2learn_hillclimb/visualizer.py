"""
Modulo di Visualizzazione Grafica e Telemetria per Play2Learn Hill Climb Racing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def mostra_mappa_ambiente() -> None:
    """Mostra una guida visiva interattiva dell'ambiente fisico e dei sensori a bordo della vettura."""
    print("""
════════════════════════════════════════════════════════════════════════════════════
 🏎️  MAPPA FISICA E SENSORI DI BORDO: HILL CLIMB RACING (Box2D Engine)
════════════════════════════════════════════════════════════════════════════════════

                     Inclinazione Telaio (angle) [rad]
                                ⤹    ⤸
                       [Pilota: Testa + Torso]
                               |     |
                       ┌───────┴─────┴──────┐
       Alettone/Retro  │   TELAIO AUTO (2D) │  Muso / Cofano
             O─────────┤                    ├─────────O
            /          └──────┬──────┬──────┘          \\
       Sospensione Posteriore │      │ Sospensione Anteriore
           (Molla + Pistone)  │      │  (Molla + Pistone)
                           [Ruota]  [Ruota]
                           Post.    Ant.
                              ↓        ↓
                 ═════════════════════════════ Terreno Collinare (Perlin Noise)

 ──────────────────────────────────────────────────────────────────────────────────
 📡 VETTORE DI OSSERVAZIONE PER LO STUDENTE (obs - 7 Grandezze Fisiche Originali):
 ──────────────────────────────────────────────────────────────────────────────────
   [0] dist_x           : Distanza orizzontale percorsa (metri, Spawn ≈ 6.67m, Traguardo >= 800m)
   [1] quota_y          : Quota altimetrica verticale del telaio in metri
   [2] angle            : Inclinazione del telaio in radianti (-pi a +pi, 0 = orizzontale)
   [3] rear_wheel_speed : Velocità di rotazione ruota posteriore (rad/s, > 0 spinge avanti)
   [4] front_wheel_speed: Velocità di rotazione ruota anteriore (rad/s, > 0 spinge avanti)
   [5] rear_wheel       : Contatto ruota posteriore col suolo (1.0 = aderenza, 0.0 = in volo)
   [6] front_wheel      : Contatto ruota anteriore col suolo (1.0 = aderenza, 0.0 = in volo)

 ──────────────────────────────────────────────────────────────────────────────────
 🏁 STATI SPECIALI DI FINE EPISODIO:
 ──────────────────────────────────────────────────────────────────────────────────
   • is_success : Scalata riuscita! Veicolo oltre il traguardo degli 800 metri
   • is_crashed : Schianto mortale! La testa del pilota ha toccato terra
   • is_timeout : Tempo massimo scaduto o veicolo bloccato/stuck in salita

 ──────────────────────────────────────────────────────────────────────────────────
 🕹️ SPAZIO DELLE AZIONI DISCRETE (action in {0, 1, 2}):
 ──────────────────────────────────────────────────────────────────────────────────
   • 0: Folle / Motore spento (il veicolo prosegue per inerzia)
   • 1: Acceleratore (Gas avanti: coppia motrice su entrambe le ruote)
   • 2: Freno / Retromarcia (frena e applica coppia inversa per correggere l'assetto)
════════════════════════════════════════════════════════════════════════════════════
""")


def plot_hill_telemetry(
    history: list[dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Genera i grafici diagnostici di telemetria sui seed di test con protezione zero-division."""
    if not history or not _MPL_AVAILABLE:
        return None

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [h.get("seed", i) for i, h in enumerate(history)]
    distances = [float(h.get("distance", h.get("max_distance", 0.0))) for h in history]
    speeds = [float(h.get("max_speed", 0.0)) for h in history]
    airtimes = [float(h.get("airtime_pct", 0.0)) for h in history]
    rewards = [float(h.get("reward", 0.0)) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Telemetria e Prestazioni di Guida - Hill Climb Racing", fontsize=14, fontweight="bold")

    # 1. Distanza massima percorsa
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(seeds)), distances, color=["#2ca02c" if d >= 800.0 else "#1f77b4" for d in distances])
    ax1.axhline(800.0, color="red", linestyle="--", linewidth=1.5, label="Traguardo Successo (800m)")
    ax1.set_title("Distanza Raggiunta per Tracciato (m)")
    ax1.set_xlabel("Episodio di Test")
    ax1.set_ylabel("Metri (m)")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # 2. Velocità di punta
    ax2 = axes[0, 1]
    ax2.plot(range(len(seeds)), speeds, marker="o", color="#ff7f0e", linewidth=2)
    ax2.set_title("Velocità Massima (m/s)")
    ax2.set_xlabel("Episodio di Test")
    ax2.set_ylabel("m/s")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Percentuale di volo (airtime)
    ax3 = axes[1, 0]
    ax3.bar(range(len(seeds)), airtimes, color="#9467bd", alpha=0.8)
    ax3.set_title("Tempo Trascorso in Volo (%)")
    ax3.set_xlabel("Episodio di Test")
    ax3.set_ylabel("Airtime %")
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    # 4. Reward per episodio
    ax4 = axes[1, 1]
    ax4.plot(range(len(seeds)), rewards, marker="s", color="#d62728", linewidth=2)
    ax4.set_title("Reward Totale Ottenuto")
    ax4.set_xlabel("Episodio di Test")
    ax4.set_ylabel("Reward Cumulativo")
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path is not None:
        fig.savefig(str(output_path), dpi=150)
        plt.close(fig)
        return output_path
    else:
        plt.show()
        plt.close(fig)
        return None
