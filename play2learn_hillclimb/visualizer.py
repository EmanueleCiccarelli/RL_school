"""
Modulo di Visualizzazione Grafica e Telemetria per Hill Climb Racing.
Implementazione Object-Oriented con Figure/FigureCanvasAgg (zero memory leak in pyplot)
e gestione sicura contro ZeroDivisionError sui conteggi degli esiti.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


def mostra_mappa_ambiente() -> None:
    """Mostra un diagramma concettuale didattico della vettura, del terreno e delle osservazioni."""
    fig = Figure(figsize=(10, 5), dpi=120)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#0b0f19")
    ax = fig.subplots()
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

    ax.annotate("obs[4]: Inclinazione\nobs[2]: Velocita VX", xy=(car_x + 0.8, car_y), xytext=(car_x + 2.5, car_y + 2.2),
                arrowprops=dict(facecolor="#f59e0b", shrink=0.08, width=1.5, headwidth=6),
                color="#f59e0b", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#d97706"))

    ax.annotate("obs[6, 7]: Contatto Ruote", xy=(car_x + 1.0, car_y - 0.4), xytext=(car_x + 2.8, car_y - 1.5),
                arrowprops=dict(facecolor="#10b981", shrink=0.08, width=1.5, headwidth=6),
                color="#10b981", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#059669"))

    ax.set_xlim(-1, 25)
    ax.set_ylim(-1, 6)
    ax.set_title("Mappa dell'Ambiente e Telemetria: Hill Climb Racing", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Distanza di Avanzamento (metri)", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Quota Terreno", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#64748b")
    ax.grid(True, linestyle="--", alpha=0.2, color="#475569")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0", loc="upper left", fontsize=8.5)

    fig.tight_layout()
    try:
        from IPython.display import Image, display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        fig.clear()
        display(Image(data=buf.getvalue()))
    except Exception:
        pass


def plot_hill_telemetry(eval_history: list[dict[str, Any]], output_path: Path | str | None = None) -> Path | None:
    """Genera il grafico 2x2 usando Figure/FigureCanvasAgg (zero leak pyplot, safe su conteggi 0)."""
    fig = Figure(figsize=(12, 7), dpi=110)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#ffffff")
    axes = fig.subplots(2, 2)

    distances = [float(ep.get("distance", ep.get("score", 0.0))) for ep in eval_history]
    speeds = [float(ep.get("max_speed", 0.0)) for ep in eval_history]
    airtimes = [float(ep.get("airtime_pct", 0.0)) for ep in eval_history]
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

    # 2. Velocita massima
    ax2 = axes[0, 1]
    ax2.plot(ep_indices, speeds, marker="o", color="#f59e0b", lw=2.0)
    ax2.set_title("Velocita Massima Registrata (m/s)", fontweight="bold", fontsize=11, color="#1e293b")
    ax2.set_xlabel("Episodio di Verifica", fontsize=9)
    ax2.set_ylabel("Velocita (m/s)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Percentuale tempo in volo
    ax3 = axes[1, 0]
    ax3.plot(ep_indices, airtimes, marker="s", color="#8b5cf6", lw=2.0)
    ax3.set_title("Percentuale Tempo in Volo (Airtime %)", fontweight="bold", fontsize=11, color="#1e293b")
    ax3.set_xlabel("Episodio di Verifica", fontsize=9)
    ax3.set_ylabel("Airtime (%)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # 4. Distribuzione esiti (protetta da ZeroDivisionError)
    ax4 = axes[1, 1]
    successes = sum(1 for ep in eval_history if ep.get("is_success", False))
    crashes = sum(1 for ep in eval_history if ep.get("is_crashed", False))
    timeouts = sum(1 for ep in eval_history if ep.get("is_timeout", False))
    raw_items = [
        ("Successo (>=800m)", successes, "#10b981"),
        ("Schianto/Ribaltamento", crashes, "#ef4444"),
        ("Blocco/Timeout", timeouts, "#f59e0b"),
    ]
    non_zero = [(lbl, cnt, col) for lbl, cnt, col in raw_items if cnt > 0]
    if non_zero:
        ax4.pie(
            [c for _, c, _ in non_zero],
            labels=[l for l, _, _ in non_zero],
            colors=[col for _, _, col in non_zero],
            autopct="%1.1f%%",
            startangle=140,
            textprops={"fontsize": 9, "color": "#1e293b"},
        )
    else:
        ax4.text(0.5, 0.5, "Nessun esito classificato", ha="center", va="center", color="#64748b")
        ax4.axis("off")
    ax4.set_title(f"Distribuzione Esiti su {len(eval_history)} Verifiche", fontweight="bold", fontsize=11, color="#1e293b")

    fig.tight_layout()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=160, bbox_inches="tight")
        fig.clear()
        return p

    try:
        from IPython.display import Image, display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        fig.clear()
        display(Image(data=buf.getvalue()))
    except Exception:
        pass
    return None


def display_notebook_summary(
    plot_path: Path | str | None = None,
    videos: list[Path | str] | None = None,
    success_videos: list[Path | str] | None = None,
    failure_videos: list[Path | str] | None = None,
    max_videos: int = 3,
    gif_path: Path | str | None = None,
) -> None:
    """
    Mostra il grafico e i video di test direttamente nell'output della cella del Notebook,
    separando chiaramente le salite di successo dai fallimenti per un'analisi didattica mirata,
    e includendo l'eventuale timelapse GIF o MP4 dell'evoluzione dell'agente.
    """
    try:
        import base64
        from IPython.display import HTML, Image, display

        p_plot = Path(plot_path) if plot_path else None
        if p_plot and p_plot.exists():
            display(Image(filename=str(p_plot)))

        # Se sono passati esplicitamente success_videos e failure_videos
        has_split = (success_videos is not None) or (failure_videos is not None)
        succ_list = [Path(v) for v in (success_videos or [])]
        fail_list = [Path(v) for v in (failure_videos or [])]

        if not has_split and videos:
            all_v = [Path(v) for v in videos]
            succ_list = [v for v in all_v if "successo" in v.name.lower()]
            fail_list = [v for v in all_v if "fallimento" in v.name.lower() or "crash" in v.name.lower()]
            if not succ_list and not fail_list:
                succ_list = all_v[:max_videos]

        sections = []

        # Card timelapse (GIF animata o MP4 looping) dell'evoluzione dell'apprendimento
        if gif_path:
            p_media = Path(gif_path)
            # Preferisci il formato GIF se presente per massima compatibilità e riproduzione continua
            if p_media.suffix.lower() == ".mp4" and p_media.with_suffix(".gif").exists():
                p_media = p_media.with_suffix(".gif")
            elif p_media.suffix.lower() == ".gif" and not p_media.exists() and p_media.with_suffix(".mp4").exists():
                p_media = p_media.with_suffix(".mp4")

            if p_media.exists():
                b64_media = base64.b64encode(p_media.read_bytes()).decode("ascii")
                if p_media.suffix.lower() == ".gif":
                    media_tag = (
                        f'<img src="data:image/gif;base64,{b64_media}" '
                        f'style="max-width: 480px; width: 100%; border-radius: 6px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);" />'
                    )
                else:
                    media_tag = (
                        f'<video width="480" controls autoplay loop muted playsinline style="border-radius:6px; box-shadow: 0 4px 8px rgba(0,0,0,0.15);">'
                        f'<source src="data:video/mp4;base64,{b64_media}" type="video/mp4">'
                        f'</video>'
                    )

                sections.append(
                    f'<div style="margin-top: 18px; text-align: center; background: #f3e5f5; padding: 14px; border-radius: 8px; border: 1.5px solid #ab47bc;">'
                    f'<h4 style="color:#6a1b9a; margin-top:2px; margin-bottom:6px;">🎞️ Evoluzione dell\'Apprendimento nel Tempo (Timelapse delle Fasi di Training)</h4>'
                    f'<p style="color:#4a148c; font-size:12px; margin-bottom:10px;">Confronto della guida sullo stesso tracciato attraverso le diverse fasi di addestramento:</p>'
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
                    f'<div style="font-weight:bold; margin-bottom:6px; color:#1b5e20;">🟢 Salita Riuscita #{idx+1}</div>'
                    f'<video width="300" controls autoplay loop muted playsinline style="border-radius:6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
                    f'</video></div>'
                )
            if cards:
                sections.append(
                    f'<div style="margin-top: 14px;">'
                    f'<h4 style="color:#2e7d32; margin-bottom:4px;">🎯 Esempi di Successo (Traguardo Raggiunto &gt;= 800m):</h4>'
                    f'<div style="display:flex; flex-wrap:wrap; justify-content:center;">{"".join(cards)}</div>'
                    f'</div>'
                )
        elif fail_list:
            sections.append(
                f'<div style="margin-top: 14px; background: #fff8e1; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #ffb300;">'
                f'<strong style="color: #f57f17;">🧗 Nessuna salita completata al traguardo (>=800m) in questa run:</strong> '
                f'<span style="color: #333;">L\'auto non ha ancora superato la vetta negli scenari di test. Analizza i video di fallimento qui sotto per capire dove perde aderenza o si ribalta!</span>'
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
                    f'<video width="300" controls autoplay loop muted playsinline style="border-radius:6px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">'
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
                f'<span style="color: #333;">Tutte le salite di test sono state completate con successo senza schianti o blocchi!</span>'
                f'</div>'
            )

        if sections:
            display(HTML("".join(sections)))
    except Exception as e:
        print(f"[play2learn] Impossibile mostrare i media inline: {e}")

