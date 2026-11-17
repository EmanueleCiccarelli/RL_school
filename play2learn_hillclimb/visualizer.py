"""
Modulo di Visualizzazione Grafica, Telemetria e Dashboard per Play2Learn Hill Climb Racing.
Include la mappa grafica annotata dell'ambiente, la telemetria 2x2 e la riproduzione video inline HTML5.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def mostra_mappa_ambiente() -> None:
    """
    Genera e visualizza nel Notebook la mappa grafica annotata di Hill Climb Racing
    con frame Box2D renderizzato, callout sui sensori, meccanica delle sospensioni e regole fisiche.
    """
    frame = None
    if _MPL_AVAILABLE:
        try:
            import gymnasium as gym
            import hill_racing_env
            env = gym.make("hill_racing_env/HillRacing-v0", render_mode="rgb_array")
            env.reset(seed=42)
            frame = env.render()
            env.close()
        except Exception:
            frame = None

    if frame is not None and _MPL_AVAILABLE:
        fig, ax = plt.subplots(figsize=(11, 6.2), dpi=130)
        ax.imshow(frame)

        # 1. Annotazione Telaio e Testa Pilota (Spawn attorno a x=200, y=520)
        ax.annotate(
            "TELAIO E PILOTA\n"
            "• Inclinazione telaio: angle (rad, 0 = orizzontale)\n"
            "• Quota altimetrica: quota_y (metri)\n"
            "• TESTA DEL PILOTA: se tocca il suolo -> SCHIANTO MORTALE (is_crashed = True)!",
            xy=(200, 520), xytext=(240, 350),
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#b71c1c", edgecolor="#ff5252", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#ff5252", lw=2)
        )

        # 2. Ruota Posteriore
        ax.annotate(
            "RUOTA POSTERIORE\n"
            "• vel_ruota_posteriore (rad/s)\n"
            "• contatto_posteriore (1 = suolo, 0 = volo)",
            xy=(150, 600), xytext=(40, 460),
            color="white", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d47a1", edgecolor="#64b5f6", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#64b5f6", lw=1.8)
        )

        # 3. Ruota Anteriore
        ax.annotate(
            "RUOTA ANTERIORE\n"
            "• vel_ruota_anteriore (rad/s)\n"
            "• contatto_anteriore (1 = suolo, 0 = volo)",
            xy=(240, 600), xytext=(320, 460),
            color="white", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0d47a1", edgecolor="#64b5f6", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#64b5f6", lw=1.8)
        )

        # 4. Traguardo e Avanzamento
        ax.annotate(
            "AVANZAMENTO & TRAGUARDO\n"
            "• Partenza: dist_x ≈ 6.67 m\n"
            "• Obiettivo: dist_x >= 800 m -> SCALATA RIUSCITA (is_success = True)!",
            xy=(600, 560), xytext=(550, 410),
            color="white", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1b5e20", edgecolor="#00e676", alpha=0.92),
            arrowprops=dict(arrowstyle="->", color="#00e676", lw=1.8)
        )

        # 5. Box Meccanica Sospensioni Box2D e chiarimento didattico
        info_box = (
            "⚙️ MECCANICA DELLE SOSPENSIONI (Box2D Engine):\n"
            "• In Box2D le sospensioni collegano le ruote al telaio con un giunto prismatico (guida lineare verticale) "
            "e una molla/distance joint elastica.\n"
            "• Le sospensioni sono totalmente PASSIVE: l'agente non deve azionarle né compaiono nei sensori di bordo.\n"
            "• Il pilota controlla solo la coppia motrice o frenante tramite action: "
            "0 (Folle / Inerzia), 1 (Gas avanti), 2 (Freno / Retro per bilanciare il beccheggio in volo)."
        )
        ax.text(
            0.5, 0.03, info_box, transform=ax.transAxes,
            color="#fffde7", fontsize=8.2, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#212121", edgecolor="#ffb300", alpha=0.93)
        )

        ax.set_title("Mappa Fisica e Sensori di Bordo - Hill Climb Racing (Box2D Physics)", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    print("""
════════════════════════════════════════════════════════════════════════════════════
 🏎️  SCHEDA TECNICA DEI SENSORI: HILL CLIMB RACING (Box2D Engine)
════════════════════════════════════════════════════════════════════════════════════

  📡 VETTORE DI OSSERVAZIONE PER LO STUDENTE (obs - 7 Grandezze Fisiche Originali):
  ──────────────────────────────────────────────────────────────────────────────────
    [0] dist_x           : Distanza orizzontale percorsa (metri, Spawn ≈ 6.67m, Traguardo >= 800m)
    [1] quota_y          : Quota altimetrica verticale del telaio in metri
    [2] angle            : Inclinazione del telaio in radianti (-pi a +pi, 0 = orizzontale)
    [3] rear_wheel_speed : Velocità di rotazione ruota posteriore (rad/s, > 0 spinge avanti)
    [4] front_wheel_speed: Velocità di rotazione ruota anteriore (rad/s, > 0 spinge avanti)
    [5] rear_wheel       : Contatto ruota posteriore col suolo (1.0 = aderenza, 0.0 = in volo)
    [6] front_wheel      : Contatto ruota anteriore col suolo (1.0 = aderenza, 0.0 = in volo)

  🏁 STATI SPECIALI DI FINE EPISODIO:
  ──────────────────────────────────────────────────────────────────────────────────
    • is_success : Scalata riuscita! Veicolo oltre il traguardo degli 800 metri
    • is_crashed : Schianto mortale! La testa del pilota ha toccato terra
    • is_timeout : Tempo massimo scaduto o veicolo bloccato/stuck in salita

  🕹️ SPAZIO DELLE AZIONI DISCRETE (action in {0, 1, 2}):
  ──────────────────────────────────────────────────────────────────────────────────
    • 0: Folle / Motore spento (il veicolo prosegue per inerzia a coppia zero)
    • 1: Acceleratore (Gas avanti: coppia motrice su entrambe le ruote)
    • 2: Freno / Retromarcia (frena a terra e applica coppia inversa per raddrizzare l'auto in volo)
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


def display_notebook_summary(
    plot_path: Path | str | None = None,
    videos: list[Path] | None = None,
    success_videos: list[Path] | None = None,
    failure_videos: list[Path] | None = None,
    max_videos: int = 3,
    gif_path: Path | str | None = None,
) -> None:
    """
    Mostra il grafico di telemetria e i video di prova direttamente nell'output della cella del Notebook,
    separando le scalate riuscite dai ribaltamenti, con lettori video HTML5 inline (senza dipendenze esterne).
    """
    try:
        from IPython.display import Image, HTML, display

        p_plot = Path(plot_path) if plot_path else None
        if p_plot and p_plot.exists():
            display(Image(filename=str(p_plot)))

        # Se sono passati esplicitamente success_videos e failure_videos
        has_split = (success_videos is not None) or (failure_videos is not None)
        succ_list = list(success_videos) if success_videos is not None else []
        fail_list = list(failure_videos) if failure_videos is not None else []

        if not has_split and videos:
            succ_list = [v for v in videos if "successo" in v.name.lower()]
            fail_list = [v for v in videos if "fallimento" in v.name.lower() or "crash" in v.name.lower()]
            if not succ_list and not fail_list:
                succ_list = list(videos)[:max_videos]

        sections: list[str] = []

        # Card timelapse GIF o MP4 dell'evoluzione dell'apprendimento
        if gif_path:
            p_media = Path(gif_path)
            if p_media.exists():
                b64_media = base64.b64encode(p_media.read_bytes()).decode("ascii")
                if p_media.suffix.lower() == ".mp4":
                    media_tag = (
                        f'<video width="460" controls autoplay loop muted style="border-radius:6px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">'
                        f'<source src="data:video/mp4;base64,{b64_media}" type="video/mp4">'
                        f'</video>'
                    )
                else:
                    media_tag = f'<img src="data:image/gif;base64,{b64_media}" style="max-width: 480px; border-radius: 6px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);" />'

                sections.append(
                    f'<div style="margin-top: 18px; text-align: center; background: #f3e5f5; padding: 12px; border-radius: 8px; border: 1.5px solid #ab47bc;">'
                    f'<h4 style="color:#6a1b9a; margin-top:2px; margin-bottom:6px;">🎞️ Evoluzione dell\'Apprendimento nel Tempo (Timelapse delle Fasi di Training)</h4>'
                    f'<p style="color:#4a148c; font-size:12px; margin-bottom:10px;">Progressione della guida sullo stesso tracciato attraverso i vari checkpoint:</p>'
                    f'{media_tag}'
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
                    f'<div style="font-weight:bold; margin-bottom:6px; color:#1b5e20;">🟢 Scalata Riuscita #{idx+1} (&ge; 800m)</div>'
                    f'<video width="320" controls autoplay loop muted playsinline style="border-radius:6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
                    f'</video></div>'
                )
            if cards:
                sections.append(
                    f'<div style="margin-top: 14px;">'
                    f'<h4 style="color:#2e7d32; margin-bottom:4px;">🎯 Esempi di Successo (Traguardo Raggiunto):</h4>'
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
                    f'<div style="font-weight:bold; margin-bottom:6px; color:#b71c1c;">🔴 Fallimento / Ribaltamento #{idx+1}</div>'
                    f'<video width="320" controls autoplay loop muted playsinline style="border-radius:6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">'
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
        elif succ_list:
            sections.append(
                f'<div style="margin-top: 14px; background: #f1f8e9; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #4caf50;">'
                f'<strong style="color: #2e7d32;">🛡️ Nessun fallimento registrato nelle prove:</strong> '
                f'<span style="color: #333;">Tutte le corse di test hanno raggiunto e superato il traguardo degli 800 metri!</span>'
                f'</div>'
            )

        if sections:
            display(HTML("".join(sections)))
    except Exception as e:
        print(f"[play2learn] Impossibile mostrare i media inline: {e}")


def plot_comparison(runs_data: list[dict[str, Any]], save_path: Path | str | None = None) -> Optional[Path]:
    """Genera un grafico comparativo tra più run (confronto tra reward shaping e prestazioni)."""
    if not runs_data or not _MPL_AVAILABLE:
        return None

    labels = []
    successes = []
    distances = []
    speeds = []
    airtimes = []

    for idx, r in enumerate(runs_data):
        meta = r.get("eval_metrics", r)
        nome = r.get("nome_run") or r.get("tag") or f"Run {idx+1}"
        labels.append(str(nome))
        successes.append(float(meta.get("success_rate", meta.get("eval_success_rate", 0.0))))
        distances.append(float(meta.get("mean_distance", 0.0)))
        speeds.append(float(meta.get("mean_speed", 0.0)))
        airtimes.append(float(meta.get("mean_airtime_pct", 0.0)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Confronto Sperimentale tra Run - Hill Climb Racing", fontsize=13, fontweight="bold")

    axes[0, 0].bar(labels, successes, color="#2e7d32")
    axes[0, 0].set_title("Tasso di Successo (% >= 800m)")
    axes[0, 0].set_ylim(0, 105)

    axes[0, 1].bar(labels, distances, color="#1565c0")
    axes[0, 1].set_title("Distanza Media Percorsa (m)")

    axes[1, 0].bar(labels, speeds, color="#e65100")
    axes[1, 0].set_title("Velocità Media di Punta (m/s)")

    axes[1, 1].bar(labels, airtimes, color="#6a1b9a")
    axes[1, 1].set_title("Tempo Trascorso in Volo (%)")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        return save_path
    else:
        plt.show()
        plt.close(fig)
        return None
