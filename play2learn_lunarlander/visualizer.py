"""
Modulo di visualizzazione grafica: mappa dell'ambiente, dashboard telemetrica 2x2,
visualizzazione video inline su Google Colab e grafici di confronto tra run.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

from play2learn_lunarlander.agent import RewardTrackerCallback
from play2learn_lunarlander.diagnostics import diagnose_agent_behavior


def mostra_mappa_ambiente() -> None:
    """
    Genera e visualizza nel Notebook una scheda grafica annotata dell'ambiente
    LunarLander-v3 con sistema di riferimento, coordinate piazzola, limiti di rottura e motori.
    """
    try:
        import gymnasium as gym
    except ImportError as e:
        raise ImportError(
            "❌ Gymnasium non trovato! Assicurati di aver eseguito la Cella di installazione in cima al notebook:\n"
            "!pip install -q 'gymnasium[box2d]' stable-baselines3"
        ) from e

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env.reset(seed=42)
    # Esegue qualche step per accendere i motori e visualizzare il modulo
    for _ in range(12):
        env.step(2)
    frame = env.render()
    env.close()

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=130)
    # Mappiamo l'immagine nelle coordinate normalizzate approssimative di Gymnasium:
    # x va da -1.0 a +1.0, y va da -0.33 (fondo) a +1.50 (cima), con y=0.0 sulla piazzola
    ax.imshow(frame, extent=[-1.0, 1.0, -0.33, 1.50], aspect="auto")

    # Linee di riferimento coordinate
    ax.axhline(0.0, color="#00e676", linestyle="--", linewidth=1.6, alpha=0.9, label="Quota Piazzola (y = 0.0)")
    ax.axvline(0.0, color="#ffeb3b", linestyle=":", linewidth=1.5, alpha=0.9, label="Centro Bersaglio (x = 0.0)")
    ax.axvline(-0.2, color="#ff5252", linestyle="--", linewidth=1.3, alpha=0.75, label="Confini Bandiere (x = ±0.20)")
    ax.axvline(0.2, color="#ff5252", linestyle="--", linewidth=1.3, alpha=0.75)

    # Annotazione Origine (0,0) e Piazzola
    ax.annotate(
        "ORIGINE (x = 0.0, y = 0.0)\nZona sicura di allunaggio: x ∈ [-0.20, +0.20]",
        xy=(0.0, 0.0), xytext=(0.26, 0.14),
        color="white", fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1b5e20", edgecolor="#00e676", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#00e676", lw=1.8)
    )

    # Annotazione Punto di Partenza e Azioni
    ax.annotate(
        "PUNTO DI PARTENZA (x ≈ 0.0, y ≈ 1.4)\n"
        "• Azione 0: Motori spenti (fuel = 0.0)\n"
        "• Azione 1: Razzo Destro (spinge verso SX: vx < 0, rotazione oraria)\n"
        "• Azione 2: Motore Principale (frena la caduta: vy sale, spinta verticale)\n"
        "• Azione 3: Razzo Sinistro (spinge verso DX: vx > 0, rotazione antioraria)",
        xy=(0.0, 1.25), xytext=(-0.95, 1.06),
        color="white", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d47a1", edgecolor="#64b5f6", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#64b5f6", lw=1.8)
    )

    # Box Regole Fisiche e Criteri di Rottura
    regole_box = (
        "REGOLE DI INTEGRITÀ STRUTTURALE (QUANDO SI DISTRUGGE?):\n"
        "1. Fusoliera: se il corpo centrale tocca terra -> ESPLOSIONE IMMEDIATA\n"
        "2. Fuori bordo: se esce a sinistra (x ≤ -1.0) o destra (x ≥ +1.0) -> DISPERSO NELLO SPAZIO\n"
        "3. Velocità d'impatto: al tocco (y ≈ 0) serve |vy| < 0.20 e |vx| < 0.15 (se vy < -0.50 le zampe cedono!)\n"
        "4. Assetto: all'atterraggio serve |angle| < 0.20 rad (circa 11°)\n"
        "5. Fine missione: quando entrambe le zampe toccano terra, l'agente DEVE spegnere i motori (action = 0)!"
    )
    ax.text(
        0.0, -0.29, regole_box,
        color="#fffde7", fontsize=8.6, ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#212121", edgecolor="#ffb300", alpha=0.93)
    )

    ax.set_title("Mappa Fisica e Sistema di Riferimento - LunarLander-v3 (50 step = 1 secondo)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Posizione Orizzontale x (da -1.0 estrema sinistra a +1.0 estrema destra)", fontsize=10)
    ax.set_ylabel("Altitudine y (0.0 = quota piazzola, ~1.4 = spawn iniziale)", fontsize=10)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.33, 1.50)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_training_results(
    tracker: RewardTrackerCallback,
    mission_title: str,
    save_path: Path | str
) -> dict[str, Any]:
    """
    Genera e salva una dashboard 2x2 con:
    1. Custom Reward per episodio (curva istantanea e media mobile)
    2. Tasso di Successo / Schianto / Timeout (%) nel tempo
    3. Durata del volo (Step per episodio con fascia ideale 150-350)
    4. Consumo di Carburante (unità spese)
    Include in calce il box della Diagnosi Flight Coach.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_episodes = len(tracker.episode_rewards)
    diag = diagnose_agent_behavior(tracker)
    diag_text = diag["formatted_text"]

    if n_episodes == 0:
        print("[play2learn] Nessun episodio completato per generare il grafico.")
        return diag

    episodes = np.arange(1, n_episodes + 1)
    window = min(40, max(5, n_episodes // 12))

    fig = Figure(figsize=(14, 8.8), dpi=300)
    FigureCanvasAgg(fig)
    fig.suptitle(f"Telemetria di Addestramento RL — {mission_title}", fontsize=14, fontweight="bold")
    axes = fig.subplots(2, 2)
    (ax1, ax2), (ax3, ax4) = axes

    # 1. Custom Reward
    ax1.plot(episodes, tracker.episode_rewards, color="#90caf9", alpha=0.45, label="Reward Episodio", rasterized=True)
    if n_episodes >= window:
        mov_rew = np.convolve(tracker.episode_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window, n_episodes + 1), mov_rew, color="#0d47a1", lw=2.2, label=f"Media Mobile ({window} ep)")
    ax1.set_title("1. Ricompensa Cumulata (La tua Custom Reward)", fontweight="bold")
    ax1.set_ylabel("Custom Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5)

    # 2. Percentuali Esiti nel Tempo
    succ_arr = np.array([100.0 if o == "success" else 0.0 for o in tracker.outcomes])
    crash_arr = np.array([100.0 if o == "crash" else 0.0 for o in tracker.outcomes])
    time_arr = np.array([100.0 if o == "timeout" else 0.0 for o in tracker.outcomes])

    if n_episodes >= window:
        x_roll = range(window, n_episodes + 1)
        ax2.plot(x_roll, np.convolve(succ_arr, np.ones(window) / window, mode="valid"), color="#2e7d32", lw=2.2, label="% Allunaggi Riusciti")
        ax2.plot(x_roll, np.convolve(crash_arr, np.ones(window) / window, mode="valid"), color="#c62828", lw=1.8, linestyle="--", label="% Schianti")
        ax2.plot(x_roll, np.convolve(time_arr, np.ones(window) / window, mode="valid"), color="#ef6c00", lw=1.8, linestyle=":", label="% Timeout (Hovering)")
    else:
        ax2.plot(episodes, succ_arr, color="#2e7d32", label="% Successo")
    ax2.set_title("2. Esito delle Missioni nel Tempo (%)", fontweight="bold")
    ax2.set_ylabel("Percentuale (%)")
    ax2.set_ylim(-5, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="center left", fontsize=8.5)

    # 3. Durata del Volo
    ax3.plot(episodes, tracker.episode_lengths, color="#a5d6a7", alpha=0.45, label="Step per Episodio", rasterized=True)
    if n_episodes >= window:
        mov_len = np.convolve(tracker.episode_lengths, np.ones(window) / window, mode="valid")
        ax3.plot(range(window, n_episodes + 1), mov_len, color="#1b5e20", lw=2.0, label=f"Media Mobile ({window} ep)")
    ax3.axhspan(150, 350, color="#00e676", alpha=0.12, label="Fascia durata ideale (150-350 step)")
    ax3.axhline(1000, color="#ef6c00", linestyle=":", lw=1.2, label="Limite Timeout (1000 step)")
    ax3.set_title("3. Durata del Volo (Step: <90 Kamikaze, 1000 Timeout)", fontweight="bold")
    ax3.set_xlabel("Numero di Episodio")
    ax3.set_ylabel("Numero di Step (50 step = 1s)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=8.5)

    # 4. Consumo di Carburante
    ax4.plot(episodes, tracker.fuel_used_list, color="#ffcc80", alpha=0.45, label="Carburante Speso", rasterized=True)
    if n_episodes >= window:
        mov_fuel = np.convolve(tracker.fuel_used_list, np.ones(window) / window, mode="valid")
        ax4.plot(range(window, n_episodes + 1), mov_fuel, color="#e65100", lw=2.0, label=f"Media Mobile ({window} ep)")
    ax4.set_title("4. Carburante Consumato (1.0 princ., 0.3 lat.)", fontweight="bold")
    ax4.set_xlabel("Numero di Episodio")
    ax4.set_ylabel("Unità di Carburante")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout(rect=[0, 0.15, 1, 0.95])

    # Box Diagnosi Flight Coach posizionato in calce
    # Sostituisce eventuali emoji non presenti nei font standard per evitare warning grafici
    clean_diag = (
        diag_text.replace("🏎️", "")
        .replace("📍", "")
        .replace("📐", "")
        .replace("⚙️", "")
        .replace("💡", "-> ")
        .replace("💥", "*")
        .replace("🎯", "[OK]")
    )
    fig.text(
        0.5, 0.015, clean_diag, ha="center", va="bottom", fontsize=9.2, family="sans-serif",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#fffde7", edgecolor="#fbc02d", lw=1.6)
    )

    fig.savefig(str(save_path), dpi=300)
    fig.clear()
    return diag


def display_notebook_summary(
    plot_path: Path | str,
    videos: list[Path] | None = None,
    success_videos: list[Path] | None = None,
    failure_videos: list[Path] | None = None,
    max_videos: int = 4,
    gif_path: Path | str | None = None,
) -> None:
    """
    Mostra il grafico e i video di test direttamente nell'output della cella del Notebook,
    separando chiaramente i voli di successo dai fallimenti per un'analisi didattica mirata,
    e includendo l'eventuale timelapse GIF dell'evoluzione dell'agente.
    """
    try:
        from IPython.display import Image, HTML, display

        p_plot = Path(plot_path) if plot_path else None
        if p_plot and p_plot.exists():
            display(Image(filename=str(p_plot)))

        # Se sono passati esplicitamente success_videos e failure_videos
        has_split = (success_videos is not None) or (failure_videos is not None)
        succ_list = success_videos or []
        fail_list = failure_videos or []

        if not has_split and videos:
            # Suddivisione automatica dal nome file
            succ_list = [v for v in videos if "successo" in v.name.lower()]
            fail_list = [v for v in videos if "fallimento" in v.name.lower() or "crash" in v.name.lower()]
            if not succ_list and not fail_list:
                succ_list = videos[:max_videos]

        sections = []

        # Card timelapse GIF dell'evoluzione dell'apprendimento
        if gif_path:
            p_gif = Path(gif_path)
            if p_gif.exists():
                b64_gif = base64.b64encode(p_gif.read_bytes()).decode("ascii")
                sections.append(
                    f'<div style="margin-top: 18px; text-align: center; background: #f3e5f5; padding: 12px; border-radius: 8px; border: 1.5px solid #ab47bc;">'
                    f'<h4 style="color:#6a1b9a; margin-top:2px; margin-bottom:6px;">🎞️ Evoluzione dell\'Apprendimento nel Tempo (Timelapse delle Fasi di Training)</h4>'
                    f'<p style="color:#4a148c; font-size:12px; margin-bottom:10px;">Confronto della traiettoria sullo stesso terreno attraverso le diverse fasi di addestramento:</p>'
                    f'<img src="data:image/gif;base64,{b64_gif}" style="max-width: 460px; border-radius: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);" />'
                    f'</div>'
                )

        if succ_list:
            cards = []
            for idx, v_path in enumerate(succ_list[:max_videos]):
                if not v_path.exists():
                    continue
                b64 = base64.b64encode(v_path.read_bytes()).decode("ascii")
                cards.append(
                    f'<div style="margin: 8px; text-align: center; background: #e8f5e9; padding: 10px; border-radius: 8px; border: 1.5px solid #4caf50;">'
                    f'<div style="font-weight:bold; margin-bottom:6px; color:#1b5e20;">🟢 Allunaggio Riuscito #{idx+1}</div>'
                    f'<video width="320" controls autoplay loop muted style="border-radius:6px;">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
                    f'</video></div>'
                )
            if cards:
                sections.append(
                    f'<div style="margin-top: 14px;">'
                    f'<h4 style="color:#2e7d32; margin-bottom:4px;">🎯 Esempi di Successo (Atterraggio Controllato):</h4>'
                    f'<div style="display:flex; flex-wrap:wrap; justify-content:center;">{"".join(cards)}</div>'
                    f'</div>'
                )

        if fail_list:
            cards = []
            for idx, v_path in enumerate(fail_list[:max_videos]):
                if not v_path.exists():
                    continue
                b64 = base64.b64encode(v_path.read_bytes()).decode("ascii")
                cards.append(
                    f'<div style="margin: 8px; text-align: center; background: #ffebee; padding: 10px; border-radius: 8px; border: 1.5px solid #e53935;">'
                    f'<div style="font-weight:bold; margin-bottom:6px; color:#b71c1c;">🔴 Fallimento / Schianto #{idx+1}</div>'
                    f'<video width="320" controls autoplay loop muted style="border-radius:6px;">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
                    f'</video></div>'
                )
            if cards:
                sections.append(
                    f'<div style="margin-top: 14px;">'
                    f'<h4 style="color:#c62828; margin-bottom:4px;">💥 Esempi di Fallimento (Cosa correggere nella reward):</h4>'
                    f'<div style="display:flex; flex-wrap:wrap; justify-content:center;">{"".join(cards)}</div>'
                    f'</div>'
                )
        else:
            sections.append(
                f'<div style="margin-top: 14px; background: #f1f8e9; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #4caf50;">'
                f'<strong style="color: #2e7d32;">🛡️ Nessun fallimento registrato nelle prove:</strong> '
                f'<span style="color: #333;">Tutti i voli di test sono stati completati con successo senza schianti o timeout!</span>'
                f'</div>'
            )

        if sections:
            display(HTML("".join(sections)))
    except Exception as e:
        print(f"[play2learn] Impossibile mostrare i media inline: {e}")


def plot_comparison(runs_data: list[dict[str, Any]], save_path: Path | str) -> None:
    """
    Genera un grafico comparativo tra più run (utile per la reportistica tecnica:
    es. confronto tra approccio iniziale ingenuo vs approccio ingegnerizzato con reward shaping).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not runs_data:
        return

    n_runs = len(runs_data)
    labels = []
    for i, r in enumerate(runs_data):
        base_name = r.get("name", f"Run {i+1}")
        ts = r.get("timestamp")
        if ts and ts not in base_name:
            labels.append(f"{base_name}\n({ts})")
        else:
            labels.append(base_name)
    success_rates = [float(r.get("eval_success_rate", 0.0)) for r in runs_data]
    fuel_means = [float(r.get("eval_mean_fuel", 0.0)) for r in runs_data]
    lengths = [float(r.get("eval_mean_length", 0.0)) for r in runs_data]

    x = np.arange(n_runs)
    width = 0.28

    fig = Figure(figsize=(13, 5), dpi=300)
    FigureCanvasAgg(fig)
    fig.suptitle("Confronto Esperimenti di Reinforcement Learning — Report Tecnico", fontsize=13, fontweight="bold")
    (ax1, ax2) = fig.subplots(1, 2)

    # Grafico 1: Tasso di Successo
    bars1 = ax1.bar(x, success_rates, width, color="#2e7d32", alpha=0.85, edgecolor="black")
    ax1.set_title("Tasso di Allunaggio Riuscito (%)", fontweight="bold")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.0f}%", ha="center", va="bottom", fontweight="bold")

    # Grafico 2: Consumo di Carburante e Durata
    bars2 = ax2.bar(x - width/2, fuel_means, width, label="Carburante Medio", color="#ef6c00", alpha=0.85, edgecolor="black")
    bars3 = ax2.bar(x + width/2, [l/5 for l in lengths], width, label="Durata (step / 5)", color="#1565c0", alpha=0.85, edgecolor="black")
    ax2.set_title("Efficienza del Volo (Consumo e Durata)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300)
    fig.clear()
