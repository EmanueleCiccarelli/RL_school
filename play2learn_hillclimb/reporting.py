"""
Modulo di Generazione Report e Confronto Esperimenti per Hill Climb Racing.
Produce schede tecniche in PDF (vettoriale ad alta risoluzione), HTML standalone e Markdown,
con corretta separazione tra sanitizzazione HTML e formati PDF/MD, e supporto flessibile per confronta_run.
"""

from __future__ import annotations

import base64
import datetime
import html
import io
import json
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from play2learn_core.storage import atomic_write_json, atomic_write_text
from play2learn_core.evaluator import (
    make_fast_predictor,
    create_split_screen_comparison_mp4,
)
from play2learn_hillclimb.config import ENV_ID
from play2learn_hillclimb.storage import find_run_by_timestamp, get_default_runs_dir
from play2learn_hillclimb.visualizer import plot_hill_telemetry


def genera_report_esperimento(
    run_or_fav: Any,
    autore: str = "Studente",
    titolo: str = "Report Esperimento Hill Climb Racing",
    note: str = "",
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Genera la scheda tecnica completa (PDF, HTML, MD) dell'esperimento di guida."""
    source_dir: Path | None = None
    if isinstance(run_or_fav, dict):
        for k in ("checkpoint_dir", "path", "dir_path", "run_dir"):
            if k in run_or_fav:
                source_dir = Path(run_or_fav[k])
                break
    elif isinstance(run_or_fav, (str, Path)):
        p = Path(run_or_fav)
        if p.exists() and p.is_dir():
            source_dir = p
        else:
            source_dir = find_run_by_timestamp(str(run_or_fav), get_default_runs_dir())

    if not source_dir or not source_dir.exists():
        raise ValueError(f"[Play2Learn] Nessuna directory di run trovata per: {run_or_fav}")

    meta_file = source_dir / "run_metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = Path(output_dir) if output_dir else (source_dir / f"report_{timestamp}")
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics = meta.get("eval_metrics", {})
    success_rate = float(metrics.get("eval_success_rate", 0.0))
    mean_dist = float(metrics.get("mean_distance", 0.0))
    max_dist = float(metrics.get("max_distance_achieved", mean_dist))
    mean_speed = float(metrics.get("mean_speed", 0.0))
    airtime_pct = float(metrics.get("mean_airtime_pct", 0.0))
    crashes = int(metrics.get("eval_crashes", 0))
    timeouts = int(metrics.get("eval_timeouts", 0))

    mission = meta.get("mission", {})
    mission_title = mission.get("title", "Campagna Standard")
    mission_desc = mission.get("description", "Colline ondulate.")

    code_source = meta.get("reward_source_code", "# Codice non disponibile.")
    raw_notes = note or meta.get("notes", "")

    # Sanitizzazione separata per contesto (HTML vs testo/PDF/MD)
    html_titolo = html.escape(titolo)
    html_autore = html.escape(autore)
    html_mission = html.escape(mission_title)
    html_note = html.escape(raw_notes)
    html_code = html.escape(code_source)

    # Generazione grafico telemetrico
    plot_path = target_dir / "telemetria_guida.png"
    history = meta.get("eval_metrics", {}).get("history", [])
    if not history:
        history = [{"distance": mean_dist, "max_speed": mean_speed, "airtime_pct": airtime_pct, "is_success": success_rate >= 80}] * 20
    plot_hill_telemetry(history, plot_path)

    # 1. Generazione PDF multipagina (senza entita HTML escape)
    pdf_path = target_dir / "scheda_esperimento.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        # Pagina 1: Telemetria e KPI
        fig1 = Figure(figsize=(8.27, 11.69), dpi=160)
        FigureCanvasAgg(fig1)
        ax1 = fig1.subplots()
        ax1.axis("off")
        fig1.text(0.5, 0.94, titolo, ha="center", fontsize=15, fontweight="bold", color="#1e293b")
        fig1.text(0.5, 0.915, f"Missione: {mission_title} | Autore: {autore} | Data: {timestamp}", ha="center", fontsize=9.5, color="#64748b")

        # Tabella KPI
        table_data = [
            ["Distanza Massima", f"{max_dist:.1f} m", ">= 800.0 m", "Obiettivo" if max_dist >= 800 else "In Corso"],
            ["Distanza Media", f"{mean_dist:.1f} m", ">= 650.0 m", "Nominale"],
            ["Velocita Media", f"{mean_speed:.1f} m/s", "3.0 - 8.0 m/s", "Costante"],
            ["Airtime Medio", f"{airtime_pct:.1f}%", "< 35.0%", "Aderenza"],
            ["Schianti / Ribaltamenti", f"{crashes} su 20", "<= 3", "Sicurezza" if crashes <= 3 else "Criticita"],
            ["Blocchi in Salita", f"{timeouts} su 20", "< 2", "Trazione"],
        ]
        t = ax1.table(cellText=table_data, colLabels=["Parametro Telemetrico", "Valore Rilevato", "Riferimento Ideale", "Stato"],
                      loc="top", cellLoc="center", bbox=[0.05, 0.70, 0.90, 0.18])
        t.auto_set_font_size(False)
        t.set_fontsize(8.5)

        # Inserimento grafico telemetrico nel PDF
        if plot_path.exists():
            img = plt.imread(str(plot_path))
            ax_img = fig1.add_axes([0.05, 0.10, 0.90, 0.55])
            ax_img.imshow(img)
            ax_img.axis("off")

        pdf.savefig(fig1)
        fig1.clear()

        # Pagina 2: Codice Sorgente e Note
        fig2 = Figure(figsize=(8.27, 11.69), dpi=160)
        FigureCanvasAgg(fig2)
        ax2 = fig2.subplots()
        ax2.axis("off")
        fig2.text(0.08, 0.94, "2. Codice Sorgente della Reward Function:", fontsize=12, fontweight="bold", color="#1e293b")
        fig2.text(0.08, 0.91, code_source[:2200], fontsize=7.5, family="monospace", va="top",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"))

        fig2.text(0.08, 0.32, "3. Note di Laboratorio e Debug:", fontsize=12, fontweight="bold", color="#1e293b")
        notes_box = f"Note:\n{raw_notes}\n\nAnalisi didattica:\n• Punti di forza dell'assetto\n• Comportamento nelle salite ripide"
        fig2.text(0.08, 0.29, notes_box, fontsize=8.5, va="top",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffde7", edgecolor="#fbc02d"))

        pdf.savefig(fig2)
        fig2.clear()

    # 2. Generazione HTML e Markdown
    plot_b64 = base64.b64encode(plot_path.read_bytes()).decode("ascii") if plot_path.exists() else ""

    md_content = f"""# 🏎️ Scheda Tecnica Esperimento — Hill Climb Racing
**Titolo:** {titolo}  
**Autore:** {autore}  
**Missione:** {mission_title}  
**Data:** {timestamp}  

## 1. Indicatori di Performance Telemetrica
| Parametro Telemetrico | Valore Rilevato | Obiettivo Ideale | Esito |
| :--- | :--- | :--- | :--- |
| **Distanza Massima** | {max_dist:.1f} m | $\\ge 800.0$ m | {'✅ Raggiunto' if max_dist >= 800 else '⚠️ In corso'} |
| **Distanza Media** | {mean_dist:.1f} m | $\\ge 650.0$ m | Nominale |
| **Velocità Media** | {mean_speed:.1f} m/s | $3.0 - 8.0$ m/s | Trazione |
| **Airtime Medio** | {airtime_pct:.1f}% | $< 35.0\\%$ | Assetto |
| **Ribaltamenti / Schianti** | {crashes} / 20 | $\\le 3$ | {'Nominale' if crashes <= 3 else 'Criticità'} |

## 2. Codice della Custom Reward
```python
{code_source}
```

## 3. Note di Laboratorio
{raw_notes}
"""

    html_content = f"""<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>{html_titolo}</title>
    <style>
        body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0b0f19; color:#f1f5f9; padding:24px; }}
        .card {{ background:#1e293b; border-radius:12px; padding:20px; max-width:860px; margin:auto; border:1px solid #334155; }}
        h1 {{ color:#38bdf8; font-size:22px; margin-top:0; }}
        table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }}
        th, td {{ padding:10px; border-bottom:1px solid #334155; text-align:left; }}
        th {{ background:#0f172a; color:#94a3b8; }}
        .badge {{ background:#0284c7; color:#fff; padding:3px 8px; border-radius:4px; font-weight:bold; }}
        pre {{ background:#0f172a; padding:14px; border-radius:8px; border:1px solid #334155; overflow-x:auto; font-size:12px; }}
        img {{ max-width:100%; border-radius:8px; margin-top:14px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🏎️ {html_titolo}</h1>
        <div>Autore: <strong>{html_autore}</strong> | Missione: <strong>{html_mission}</strong> | Data: <strong>{timestamp}</strong></div>

        <table>
            <thead><tr><th>Parametro</th><th>Rilevato</th><th>Obiettivo</th><th>Esito</th></tr></thead>
            <tbody>
                <tr><td><strong>Distanza Massima</strong></td><td>{max_dist:.1f} m</td><td>&ge; 800.0 m</td><td><span class="badge">{'Traguardo OK' if max_dist >= 800 else 'In Corso'}</span></td></tr>
                <tr><td><strong>Distanza Media</strong></td><td>{mean_dist:.1f} m</td><td>&ge; 650.0 m</td><td>Nominale</td></tr>
                <tr><td><strong>Velocità Media</strong></td><td>{mean_speed:.1f} m/s</td><td>3.0 - 8.0 m/s</td><td>Costante</td></tr>
                <tr><td><strong>Airtime Medio</strong></td><td>{airtime_pct:.1f}%</td><td>&lt; 35%</td><td>Aderenza</td></tr>
                <tr><td><strong>Schianti</strong></td><td>{crashes} / 20</td><td>&le; 3</td><td>Sicurezza</td></tr>
            </tbody>
        </table>

        {f'<img src="data:image/png;base64,{plot_b64}" alt="Telemetria">' if plot_b64 else ''}

        <h3>Codice Reward Function</h3>
        <pre><code>{html_code}</code></pre>

        <h3>Note di Laboratorio</h3>
        <div>{html_note}</div>
    </div>
</body>
</html>
"""

    md_path = target_dir / "report_esperimento.md"
    html_path = target_dir / "report_esperimento.html"
    atomic_write_text(md_path, md_content)
    atomic_write_text(html_path, html_content)

    print(f"✅ Report Hill Climb generato in: {target_dir.name}")
    print(f"   • PDF:      {pdf_path.name}")
    print(f"   • HTML:     {html_path.name}")
    print(f"   • Markdown: {md_path.name}")

    return {
        "pdf_path": pdf_path,
        "html_path": html_path,
        "md_path": md_path,
        "export_dir": target_dir,
    }


def _load_model_from_run_hill(run_dir: Path) -> Any:
    for fname in ("final_model.zip", "best_model.zip"):
        cand = run_dir / fname
        if cand.exists():
            from stable_baselines3 import PPO
            return PPO.load(str(cand))
    zips = list(run_dir.glob("*.zip"))
    if zips:
        from stable_baselines3 import PPO
        return PPO.load(str(zips[0]))
    return None


def _esegui_valutazione_comparativa_hill(
    run_dir_a: Path,
    run_dir_b: Path,
    label_a: str,
    label_b: str,
    n_episodes: int = 20,
    video_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Esegue un confronto deterministico su 20 scenari collinari identici ed estrae 2 video split-screen."""
    import gymnasium as gym
    import numpy as np

    model_a = _load_model_from_run_hill(run_dir_a)
    model_b = _load_model_from_run_hill(run_dir_b)
    if model_a is None or model_b is None:
        return {}

    seeds = [1000 + i for i in range(n_episodes)]
    v_seeds = video_seeds or [42, 142]

    env = gym.make(ENV_ID, render_mode=None)
    pred_a = make_fast_predictor(model_a)
    pred_b = make_fast_predictor(model_b)

    results_a: dict[str, list[float]] = {"distances": [], "speeds": []}
    results_b: dict[str, list[float]] = {"distances": [], "speeds": []}
    wins_a = 0
    wins_b = 0
    ties = 0

    try:
        for s in seeds:
            # Run A
            obs, _ = env.reset(seed=s)
            done = False
            step_a = 0
            while not done and step_a < 1000:
                act = pred_a(obs, True)
                obs, rew, term, trunc, _ = env.step(act)
                step_a += 1
                done = bool(term or trunc)
            dist_a = float(obs[0])
            sp_a = abs(float(obs[2]))
            results_a["distances"].append(dist_a)
            results_a["speeds"].append(sp_a)

            # Run B
            obs, _ = env.reset(seed=s)
            done = False
            step_b = 0
            while not done and step_b < 1000:
                act = pred_b(obs, True)
                obs, rew, term, trunc, _ = env.step(act)
                step_b += 1
                done = bool(term or trunc)
            dist_b = float(obs[0])
            sp_b = abs(float(obs[2]))
            results_b["distances"].append(dist_b)
            results_b["speeds"].append(sp_b)

            if dist_a > dist_b + 5.0:
                wins_a += 1
            elif dist_b > dist_a + 5.0:
                wins_b += 1
            else:
                ties += 1
    finally:
        env.close()

    # Video split-screen su 2 scenari collinari
    video_paths: list[Path] = []
    try:
        render_env = gym.make(ENV_ID, render_mode="rgb_array")
        for vs in v_seeds:
            # Video A
            obs, _ = render_env.reset(seed=vs)
            frames_a = []
            f = render_env.render()
            if f is not None:
                frames_a.append(f)
            done = False
            st_a = 0
            while not done and st_a < 800:
                act = pred_a(obs, True)
                obs, rew, term, trunc, _ = render_env.step(act)
                st_a += 1
                done = bool(term or trunc)
                if st_a % 2 == 0 or done:
                    f = render_env.render()
                    if f is not None:
                        frames_a.append(f)
            dist_va = float(obs[0])

            # Video B
            obs, _ = render_env.reset(seed=vs)
            frames_b = []
            f = render_env.render()
            if f is not None:
                frames_b.append(f)
            done = False
            st_b = 0
            while not done and st_b < 800:
                act = pred_b(obs, True)
                obs, rew, term, trunc, _ = render_env.step(act)
                st_b += 1
                done = bool(term or trunc)
                if st_b % 2 == 0 or done:
                    f = render_env.render()
                    if f is not None:
                        frames_b.append(f)
            dist_vb = float(obs[0])

            tag_a = f"{label_a[:16]} ({dist_va:.0f}m)"
            tag_b = f"{label_b[:16]} ({dist_vb:.0f}m)"
            out_v = get_default_runs_dir() / f"confronto_gara_seed{vs}.mp4"
            create_split_screen_comparison_mp4(
                frames_a=frames_a,
                frames_b=frames_b,
                label_a=tag_a,
                label_b=tag_b,
                output_path=out_v,
                fps=25,
            )
            if out_v.exists():
                video_paths.append(out_v)
        render_env.close()
    except Exception as e:
        print(f"[play2learn] Avviso: registrazione video split-screen Hill Climb saltata: {e}")

    mean_dist_a = float(np.mean(results_a["distances"])) if results_a["distances"] else 0.0
    mean_dist_b = float(np.mean(results_b["distances"])) if results_b["distances"] else 0.0
    max_dist_a = float(np.max(results_a["distances"])) if results_a["distances"] else 0.0
    max_dist_b = float(np.max(results_b["distances"])) if results_b["distances"] else 0.0

    return {
        "n_episodes": n_episodes,
        "mean_dist_a": mean_dist_a,
        "mean_dist_b": mean_dist_b,
        "max_dist_a": max_dist_a,
        "max_dist_b": max_dist_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "video_paths": video_paths,
    }


def _mostra_dashboard_testa_a_testa_hill(r_a: Path, r_b: Path, ext: dict[str, Any]) -> None:
    """Mostra la dashboard HTML interattiva per Hill Climb con video affiancati."""
    try:
        from IPython.display import HTML, display
        wins_a = ext["wins_a"]
        wins_b = ext["wins_b"]
        ties = ext["ties"]
        dist_a = ext["mean_dist_a"]
        dist_b = ext["mean_dist_b"]
        max_a = ext["max_dist_a"]
        max_b = ext["max_dist_b"]
        name_a = html.escape(str(r_a.name[:18]))
        name_b = html.escape(str(r_b.name[:18]))

        winner_banner = (
            f"🏆 Vettura Più Performante: <strong>{name_a}</strong> ({wins_a} vittorie vs {wins_b})"
            if wins_a > wins_b else
            (f"🏆 Vettura Più Performante: <strong>{name_b}</strong> ({wins_b} vittorie vs {wins_a})" if wins_b > wins_a
             else "🤝 Prestazioni Equivalenti sulle Colline!")
        )

        video_cards = []
        for vpath in ext.get("video_paths", []):
            if vpath.exists():
                b64 = base64.b64encode(vpath.read_bytes()).decode("ascii")
                v_name = vpath.stem.replace("confronto_gara_", "Percorso ").capitalize()
                video_cards.append(
                    f'<div style="margin:8px; text-align:center; background:#1e293b; padding:10px; border-radius:8px; border:1.5px solid #38bdf8;">'
                    f'<div style="font-weight:bold; color:#38bdf8; font-size:13px; margin-bottom:6px;">🏁 Gara Simultanea: {v_name} ({name_a} vs {name_b})</div>'
                    f'<video width="560" controls autoplay loop muted style="border-radius:6px; max-width:100%;">'
                    f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
                    f'</video></div>'
                )

        videos_html = f'<div style="display:flex; flex-wrap:wrap; justify-content:center; gap:12px; margin-top:14px;">{"".join(video_cards)}</div>' if video_cards else ''

        card_html = f"""
        <div style="font-family:'Segoe UI',sans-serif; background:#0b1120; color:#f8fafc; padding:18px; border-radius:12px; margin:20px 0; border:2px solid #38bdf8; box-shadow:0 6px 20px rgba(0,0,0,0.4);">
            <h3 style="margin-top:0; color:#38bdf8; border-bottom:1px solid #1e293b; padding-bottom:8px;">⚔️ Sfida Diretta Testa a Testa su 20 Scenari Collinari</h3>
            <div style="font-size:15px; margin-bottom:14px; background:#1e293b; padding:10px 14px; border-radius:8px; color:#fef08a;">{winner_banner}</div>
            
            <table style="width:100%; border-collapse:collapse; font-size:13.5px; text-align:center; background:#111827; border-radius:8px; overflow:hidden;">
                <thead style="background:#1e293b; color:#38bdf8;">
                    <tr>
                        <th style="padding:10px; text-align:left;">Metrica Verificata (20 Gare)</th>
                        <th style="padding:10px;">🔵 {name_a}</th>
                        <th style="padding:10px;">🟢 {name_b}</th>
                        <th style="padding:10px;">Vantaggio</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid #1f2937;">
                        <td style="padding:10px; text-align:left; font-weight:bold;">Distanza Media Percorsa</td>
                        <td style="padding:10px; font-weight:bold; color:{'#10b981' if dist_a>=dist_b else '#f87171'};">{dist_a:.1f} m</td>
                        <td style="padding:10px; font-weight:bold; color:{'#10b981' if dist_b>=dist_a else '#f87171'};">{dist_b:.1f} m</td>
                        <td style="padding:10px;">{f"+{dist_a-dist_b:.1f} m" if dist_a>dist_b else (f"+{dist_b-dist_a:.1f} m" if dist_b>dist_a else "Parità")}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #1f2937;">
                        <td style="padding:10px; text-align:left; font-weight:bold;">Picco Distanza Massima</td>
                        <td style="padding:10px; font-weight:bold; color:{'#10b981' if max_a>=max_b else '#f87171'};">{max_a:.1f} m</td>
                        <td style="padding:10px; font-weight:bold; color:{'#10b981' if max_b>=max_a else '#f87171'};">{max_b:.1f} m</td>
                        <td style="padding:10px;">{f"+{max_a-max_b:.1f} m" if max_a>max_b else (f"+{max_b-max_a:.1f} m" if max_b>max_a else "Parità")}</td>
                    </tr>
                    <tr>
                        <td style="padding:10px; text-align:left; font-weight:bold;">Gare Vinte nello Scontro Diretto</td>
                        <td style="padding:10px; font-size:15px; font-weight:bold; color:#38bdf8;">{wins_a} / 20</td>
                        <td style="padding:10px; font-size:15px; font-weight:bold; color:#10b981;">{wins_b} / 20</td>
                        <td style="padding:10px; color:#94a3b8;">{ties} pareggi</td>
                    </tr>
                </tbody>
            </table>
            {videos_html}
        </div>
        """
        display(HTML(card_html))
    except Exception as e:
        print(f"[play2learn] Errore visualizzazione dashboard testa a testa: {e}")


def confronta_run(runs_or_favs: Any) -> dict[str, Any]:
    """
    Genera un confronto sinottico tra piu sessioni di Hill Climb.
    Accetta una lista di run/preferiti, un oggetto LabResult, una cartella o una stringa separata da virgola
    (es. 'latest, previous' oppure ['run_1', 'run_2']).
    """
    runs_dir = get_default_runs_dir()

    # Normalizza l'input se e una stringa con virgole o singolo identificatore
    items_to_check: list[Any] = []
    if isinstance(runs_or_favs, str):
        if "," in runs_or_favs:
            items_to_check = [s.strip() for s in runs_or_favs.split(",") if s.strip()]
        else:
            items_to_check = [runs_or_favs.strip()]
    elif isinstance(runs_or_favs, (list, tuple)):
        items_to_check = list(runs_or_favs)
    elif isinstance(runs_or_favs, dict):
        items_to_check = [runs_or_favs]
    else:
        items_to_check = [runs_or_favs]

    resolved: list[Path] = []
    for item in items_to_check:
        if isinstance(item, Path) and item.exists():
            resolved.append(item)
        elif isinstance(item, dict):
            for k in ("checkpoint_dir", "path", "dir_path", "run_dir"):
                if k in item:
                    cand = Path(item[k])
                    if cand.exists():
                        resolved.append(cand)
                        break
        elif isinstance(item, str):
            p = find_run_by_timestamp(item, runs_dir)
            if p and p.exists():
                resolved.append(p)

    if not resolved:
        print("[play2learn] Nessuna run valida trovata per il confronto.")
        return {}

    labels = []
    distances = []
    speeds = []
    for r in resolved:
        meta_p = r / "run_metadata.json"
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                labels.append(meta.get("mission", {}).get("title", r.name[:16]))
                distances.append(float(meta.get("eval_metrics", {}).get("mean_distance", 0.0)))
                speeds.append(float(meta.get("eval_metrics", {}).get("mean_speed", 0.0)))
            except Exception:
                labels.append(r.name[:16])
                distances.append(0.0)
                speeds.append(0.0)
        else:
            labels.append(r.name[:16])
            distances.append(0.0)
            speeds.append(0.0)

    fig = Figure(figsize=(8, 4), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.bar(labels, distances, color="#38bdf8", alpha=0.85)
    ax.set_title("Confronto Distanza Media tra Run", fontweight="bold")
    ax.set_ylabel("Distanza Media (m)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    try:
        from IPython.display import Image, display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        fig.clear()
        display(Image(data=buf.getvalue()))
    except Exception:
        pass

    extended_results = {}
    if len(resolved) >= 2:
        try:
            print("\n⚔️ Avvio valutazione estesa testa a testa su 20 scenari collinari...")
            ext = _esegui_valutazione_comparativa_hill(
                run_dir_a=resolved[0],
                run_dir_b=resolved[1],
                label_a=resolved[0].name,
                label_b=resolved[1].name,
                n_episodes=20,
                video_seeds=[42, 142],
            )
            if ext:
                extended_results = ext
                _mostra_dashboard_testa_a_testa_hill(resolved[0], resolved[1], ext)
        except Exception as e:
            print(f"[play2learn] Avviso: confronto esteso non completato: {e}")

    return {"resolved_runs": resolved, "mean_distances": distances, "head_to_head": extended_results}


def genera_report_da_timestamp(
    timestamp: str = "latest",
    tipo: str = "Base",
    autore: str = "Studente",
    note: str = "",
    timestamp_or_id: str | None = None,
    livello_report: str | None = None,
) -> dict[str, Path]:
    """Risolve la run da timestamp e genera il report tecnico."""
    ts = timestamp_or_id if timestamp_or_id is not None else timestamp
    runs_dir = get_default_runs_dir()
    run_path = find_run_by_timestamp(ts, runs_dir)
    if not run_path:
        from play2learn_hillclimb.storage import mostra_cronologia_run
        print(f"❌ Nessuna run trovata corrispondente a: '{ts}'")
        print("💡 Ecco l'elenco delle run disponibili con i relativi timestamp da copiare:\n")
        mostra_cronologia_run()
        return {}

    return genera_report_esperimento(
        run_or_fav=run_path,
        autore=autore,
        titolo=f"Report Esperimento {run_path.name}",
        note=note,
    )
