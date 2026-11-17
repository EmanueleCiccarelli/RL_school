"""
Modulo Orchestratore Principale per Play2Learn Hill Climb Racing.
Fornisce l'interfaccia didattica unificata `run_lab()` e le funzioni di supporto
per gli esperimenti, la visualizzazione e il confronto nei notebook degli studenti.
"""

from __future__ import annotations

# Fix preventivo e definitivo per PyTorch / Sympy in Python 3.13 (Google Colab)
import sys
try:
    import sympy
    import sympy.logic
    import sympy.logic.boolalg
    sympy.logic = sympy.logic
    sympy.logic.boolalg = sympy.logic.boolalg
    sys.modules["sympy"].logic = sympy.logic
    sys.modules["sympy.logic"] = sympy.logic
    sys.modules["sympy.logic.boolalg"] = sympy.logic.boolalg
    import torch.utils._sympy.numbers
except Exception:
    pass

import copy
import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import numpy as np

import gymnasium as gym

try:
    from stable_baselines3 import PPO
    _SB3_AVAILABLE = True
except ImportError:
    PPO = Any  # type: ignore
    _SB3_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from play2learn_hillclimb.config import (
    ENV_ID,
    MISSIONS,
    DEFAULT_TIMESTEPS,
    TARGET_DISTANCE_SUCCESS,
)
from play2learn_hillclimb.environment import (
    make_parallel_envs,
    _make_single_env,
    HillClimbRewardWrapper,
    classify_hill_climb_outcome,
)
from play2learn_hillclimb.agent import (
    create_ppo_agent,
    HillClimbRewardTrackerCallback,
)
from play2learn_hillclimb.evaluator import (
    evaluate_and_record,
    crea_gif_evoluzione_apprendimento,
)
from play2learn_hillclimb.diagnostics import diagnose_hill_climb_behavior
from play2learn_hillclimb.visualizer import (
    mostra_mappa_ambiente,
    plot_hill_telemetry,
    display_notebook_summary,
    plot_comparison,
)
from play2learn_hillclimb.storage import (
    create_run_directory,
    save_run_metadata,
    comprimi_archivio,
)
from play2learn_hillclimb.favorites import (
    salva_nei_preferiti,
    mostra_preferiti,
    carica_preferito,
    LoadedFavorite,
)
from play2learn_hillclimb.reporting import (
    genera_report_esperimento,
    genera_report_da_timestamp,
    mostra_cronologia,
)


def _display_report_popup(run_dir: Path, rep_files: dict[str, Path], succ_rate: float) -> None:
    """Mostra un box notifica interattivo nel Notebook per il report e i file generati."""
    try:
        from IPython.display import HTML, display
        import html
        pdf_f = rep_files.get("pdf") or rep_files.get("pdf_path") or (run_dir / "scheda_esperimento.pdf")
        html_f = rep_files.get("html") or rep_files.get("html_path") or (run_dir / "report_esperimento.html")
        md_f = rep_files.get("md") or rep_files.get("md_path") or (run_dir / "report_esperimento.md")

        pdf_name = html.escape(Path(pdf_f).name)
        html_name = html.escape(Path(html_f).name)
        md_name = html.escape(Path(md_f).name)
        run_name = html.escape(run_dir.name)

        is_winner = succ_rate >= 80.0
        header_color = "#1b5e20" if is_winner else "#0d47a1"
        border_color = "#2e7d32" if is_winner else "#1565c0"
        bg_gradient = (
            "linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)"
            if is_winner
            else "linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)"
        )
        title_text = "🏆 OBIETTIVO SCALATA RAGGIUNTO (&ge; 80%)!" if is_winner else "📊 REPORT TECNICO DELL'ESPERIMENTO PRONTO"
        sub_text = (
            "Traguardo degli 800m superato con successo nella maggioranza delle prove."
            if is_winner
            else "Scheda tecnica completa generata e archiviata per l'analisi didattica."
        )

        box_html = f"""
        <div style="background: {bg_gradient};
                    border: 2px solid {border_color}; border-radius: 12px; padding: 20px;
                    margin: 18px 0; font-family: sans-serif; box-shadow: 0 4px 15px rgba(0,0,0,0.12);">
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <span style="font-size:32px; margin-right:12px;">{'🏆' if is_winner else '🏎️'}</span>
                <div>
                    <h3 style="color:{header_color}; margin:0; font-size:18px;">
                        {title_text}
                    </h3>
                    <p style="color:{'#2e7d32' if is_winner else '#1976d2'}; margin:2px 0 0 0; font-size:13.5px;">
                        {sub_text}
                    </p>
                </div>
            </div>
            <div style="background:#ffffff; border-radius:8px; padding:12px 16px; margin-top:10px; font-size:13px; color:#333;">
                <div>📁 <strong>Cartella Esperimento:</strong> <code>runs/{run_name}</code></div>
                <div style="margin-top:5px;">📄 <strong>Scheda Tecnica PDF:</strong> <code>{pdf_name}</code></div>
                <div style="margin-top:5px;">🌐 <strong>Report HTML Interattivo:</strong> <code>{html_name}</code></div>
                <div style="margin-top:5px;">📝 <strong>Riepilogo Markdown:</strong> <code>{md_name}</code></div>
                <div style="margin-top:5px;">🎬 <strong>Video di Valutazione:</strong> archiviati in <code>runs/{run_name}/</code></div>
            </div>
            <div style="margin-top:12px; padding:10px 14px; background:#fff9c4; border-radius:8px; border-left:4px solid #fbc02d; font-size:12.8px; color:#5d4037;">
                ⭐ <strong>I Tuoi Preferiti:</strong> Puoi salvare questa run tra i tuoi preferiti con:<br>
                <code>risultati.salva_preferito("Mio_Assetto_Top", note="Ottima stabilità e scalata pulita")</code>
            </div>
        </div>
        """
        display(HTML(box_html))
    except Exception:
        pass


class LabResult(dict):
    """
    Oggetto risultato didattico restituito da `run_lab`.
    Permette l'accesso sia come dizionario che come attributi (es. res.eval_metrics).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'LabResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def mostra_evoluzione(self) -> None:
        """Visualizza la GIF animata dell'evoluzione dell'apprendimento su Colab o Jupyter."""
        gif_path = self.get("gif_path")
        if gif_path is not None and Path(gif_path).exists():
            try:
                from IPython.display import display, Image  # type: ignore
                display(Image(filename=str(gif_path)))
            except Exception:
                print(f"[play2learn] GIF salvata in: {gif_path}")
        else:
            print("[play2learn] Nessuna animazione GIF registrata per questa run.")

    def mostra_corsa(self, indice: int = 0) -> None:
        """Visualizza uno dei video di prova registrati."""
        videos = self.get("recorded_videos", []) or self.get("video_paths", []) or self.get("videos", [])
        if videos and 0 <= indice < len(videos):
            display_notebook_summary(plot_path=self.get("plot_path"), videos=[videos[indice]])
        else:
            print(f"[play2learn] Nessun video trovato all'indice {indice}.")

    def mostra_video(self, indice: int = 0) -> None:
        """Alias per mostra_corsa()."""
        self.mostra_corsa(indice=indice)

    def mostra_report(self) -> None:
        """Visualizza l'HTML del report direttamente nel notebook."""
        bundle = self.get("report_bundle", {})
        rep_files = bundle.get("rep_files", {})
        html_p = rep_files.get("html") or rep_files.get("html_path")
        if html_p is None:
            run_dir = Path(self.get("dir_path", self.get("run_dir", ".")))
            cand = run_dir / "report_esperimento.html"
            if cand.exists():
                html_p = cand
        if html_p and Path(html_p).exists():
            try:
                from IPython.display import HTML, display
                content = Path(html_p).read_text(encoding="utf-8")
                display(HTML(content))
            except Exception as e:
                print(f"[play2learn] Impossibile visualizzare il report HTML: {e}")
        else:
            print("[play2learn] Nessun file HTML di report trovato.")

    def salva_preferito(self, nome_personalizzato: str = "Mio_Preferito", note: str = "", **kwargs: Any) -> dict[str, Any]:
        """Salva la run nel catalogo dei preferiti personali."""
        run_dir = Path(self.get("dir_path", self.get("run_dir", ".")))
        return salva_nei_preferiti(run_dir=run_dir, nome=nome_personalizzato, note=note)

    def genera_report(
        self,
        titolo: str = "Esperimento Hill Climb Racing",
        autore: str = "Studente Play2Learn",
        note: str = "",
        mostra_a_video: bool = True,
        **kwargs: Any,
    ) -> dict[str, Path]:
        """Rigenera la scheda tecnica dell'esperimento."""
        run_dir = Path(self.get("dir_path", self.get("run_dir", ".")))
        rep = genera_report_esperimento(
            run_dir=run_dir,
            metadata=dict(self),
            titolo=titolo,
            autore=autore,
            note=note,
        )
        self["report_bundle"] = {"rep_files": rep}
        metrics = self.get("eval_metrics", {})
        succ_rate = float(metrics.get("success_rate", metrics.get("eval_success_rate", 0.0)))
        if mostra_a_video:
            _display_report_popup(run_dir, rep, succ_rate)
        return rep


def run_lab(
    reward_fn: Optional[Callable] = None,
    mission: str = "campagna_standard",
    total_timesteps: int = DEFAULT_TIMESTEPS,
    nome_run: Optional[str] = None,
    num_envs: int = 1,
    learning_rate: float = 3e-4,
    crea_gif_evoluzione: bool = True,
    seed: int = 42,
    device: str = "cpu",
    output_dir: Optional[str | Path] = None,
    n_envs: Optional[int] = None,
    crea_gif: Optional[bool] = None,
    display_in_notebook: bool = True,
    use_reward_norm: bool = True,
    **kwargs: Any,
) -> LabResult:
    """
    Avvia l'esperimento didattico completo:
      1. Configura l'ambiente vettoriale con la custom_reward dello studente.
      2. Allena l'agente PPO visualizzando la barra di avanzamento e la telemetria live.
      3. Valuta le prestazioni su 20 seed deterministici.
      4. Registra video MP4 in streaming per le migliori scalate e i fallimenti.
      5. Genera la Scheda Tecnica (PDF, HTML5, Markdown) e il timelapse GIF.
    """
    if n_envs is not None:
        num_envs = n_envs
    if crea_gif is not None:
        crea_gif_evoluzione = crea_gif

    tag_str = f" - {nome_run}" if nome_run else ""
    print("═════════════════════════════════════════════════════════════════════")
    print(f" 🏎️  AVVIO ESPERIMENTO DIDATTICO: HILL CLIMB RACING ({mission}){tag_str}")
    print(f" • Timesteps Totali: {total_timesteps:,}")
    print(f" • Learning Rate:    {learning_rate}")
    print(f" • Seed Casuale:     {seed}")
    print("═════════════════════════════════════════════════════════════════════\n")

    prefix = nome_run or "run_hillclimb"
    if output_dir is not None:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_directory(prefix=prefix)

    # 1. Creazione ambiente vettoriale
    vec_env = make_parallel_envs(
        num_envs=num_envs,
        mission=mission,
        reward_fn=reward_fn,
        target_distance=TARGET_DISTANCE_SUCCESS,
        seed=seed,
    )

    # 2. Inizializzazione agente PPO
    model = create_ppo_agent(
        env=vec_env,
        learning_rate=learning_rate,
        total_timesteps=total_timesteps,
        seed=seed,
        device=device,
    )

    # 3. Tracciamento pietre miliari per il timelapse GIF (usando gli snapshot in RAM del callback)
    checkpoint_interval = max(10_000, total_timesteps // 5)
    cb = HillClimbRewardTrackerCallback(
        total_timesteps=total_timesteps,
        window_size=20,
        save_checkpoints=True,
        checkpoint_dir=run_dir,
        milestone_freq=checkpoint_interval,
    )

    # 4. Addestramento
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=cb,
            progress_bar=False,
        )
    finally:
        vec_env.close()

    # 5. Salvataggio checkpoint modello finale e best
    final_model_path = run_dir / "final_model.zip"
    model.save(str(final_model_path))

    best_model_path = run_dir / "best_model.zip"
    if not best_model_path.exists():
        model.save(str(best_model_path))

    # 6. Valutazione e Registrazione Video
    print("\n🔍 Valutazione finale su 20 scenari di collaudo...")
    eval_metrics = evaluate_and_record(
        model=model,
        mission=mission,
        reward_fn=reward_fn,
        output_dir=run_dir,
        max_steps=1600,
        frame_skip=3,
    )

    # 7. Generazione GIF Evoluzione
    gif_path = None
    snapshots = getattr(cb, "milestone_snapshots", [])
    if crea_gif_evoluzione and snapshots:
        gif_out = run_dir / "evoluzione_apprendimento.gif"
        try:
            gif_path = crea_gif_evoluzione_apprendimento(
                milestones=snapshots,
                model=model,
                output_path=gif_out,
                mission=mission,
                seed=seed,
            )
            if gif_path:
                print(f"🎬 Timelapse GIF generato con successo: {gif_out.name}")
        except Exception as e:
            print(f"[play2learn] Avviso generazione GIF: {e}")

    # 8. Generazione grafico telemetria e Salvataggio Metadati
    telemetry_img = run_dir / "telemetria_esperimento.png"
    history = eval_metrics.get("history", [])
    if history and _MPL_AVAILABLE:
        try:
            plot_hill_telemetry(history, output_path=telemetry_img)
        except Exception as e:
            print(f"[play2learn] Avviso plot telemetria: {e}")

    # Estrazione codice sorgente custom_reward per il report
    reward_source_code = None
    if reward_fn is not None:
        try:
            import inspect
            reward_source_code = inspect.getsource(reward_fn)
        except Exception:
            pass

    mission_title = MISSIONS.get(mission, {}).get("nome", mission)
    metadata = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mission": {"key": mission, "title": mission_title},
        "mission_name": mission,
        "nome_run": nome_run,
        "tag": nome_run,
        "total_timesteps": total_timesteps,
        "learning_rate": learning_rate,
        "seed": seed,
        "eval_metrics": eval_metrics,
        "history": history,
        "run_dir": str(run_dir),
        "dir_path": str(run_dir),
        "gif_path": str(gif_path) if gif_path else None,
        "telemetry_img": str(telemetry_img) if telemetry_img.exists() else None,
        "reward_source_code": reward_source_code,
    }
    save_run_metadata(run_dir, metadata)

    # 9. Generazione Scheda Tecnica e Report Completo
    report_title = f"Esperimento Hill Climb Racing - {nome_run}" if nome_run else "Esperimento Hill Climb Racing"
    rep_files = genera_report_esperimento(run_dir, metadata=metadata, titolo=report_title)

    # 10. Stampa diagnosi finale
    print(diagnose_hill_climb_behavior(eval_metrics))

    # 11. Visualizzazione interattiva nel Notebook (Dashboard telemetrica, Video e Popup)
    if display_in_notebook:
        display_notebook_summary(
            plot_path=telemetry_img if telemetry_img.exists() else None,
            videos=eval_metrics.get("video_paths", []),
            success_videos=eval_metrics.get("success_videos", []),
            failure_videos=eval_metrics.get("failure_videos", []),
            gif_path=gif_path,
        )
        succ_rate = float(eval_metrics.get("success_rate", eval_metrics.get("eval_success_rate", 0.0)))
        _display_report_popup(run_dir, rep_files, succ_rate)

    res = LabResult(metadata)
    res["model"] = model
    res["eval_metrics"] = eval_metrics
    res["dir_path"] = run_dir
    res["run_dir"] = run_dir
    res["plot_path"] = telemetry_img
    res["recorded_videos"] = eval_metrics.get("video_paths", [])
    res["videos"] = eval_metrics.get("video_paths", [])
    res["success_videos"] = eval_metrics.get("success_videos", [])
    res["failure_videos"] = eval_metrics.get("failure_videos", [])
    res["gif_path"] = gif_path
    res["report_bundle"] = {"rep_files": rep_files}
    return res


def collauda_missione(
    model: Any,
    mission: str = "deserto_dune",
    eval_episodes: int = 10,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collauda un modello già addestrato su una missione specifica estrema."""
    print(f"\n🚗 Collaudo Modello su Missione Sfida: '{mission}'...")
    results = evaluate_and_record(
        model=model,
        mission=mission,
        eval_seeds=[seed + i * 17 for i in range(eval_episodes)],
        max_steps=1600,
        frame_skip=3,
    )
    print(diagnose_hill_climb_behavior(results))
    return results


def confronta_run(
    lista_risultati: Sequence[LabResult | dict[str, Any]] | str,
    output_image_path: Optional[Path | str] = None,
    **kwargs: Any,
) -> None:
    """Mette a confronto le metriche e la telemetria di più sessioni di addestramento."""
    if not lista_risultati:
        print("[play2learn] Fornisci almeno un risultato da confrontare.")
        return

    raw_items: list[Any] = []
    if isinstance(lista_risultati, str):
        raw_items = [x.strip() for x in lista_risultati.replace(";", ",").split(",") if x.strip()]
    elif isinstance(lista_risultati, (list, tuple)):
        raw_items = list(lista_risultati)
    else:
        raw_items = [lista_risultati]

    resolved_runs: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for item in raw_items:
        if isinstance(item, (dict, LabResult)):
            meta = dict(item)
            resolved_runs.append(meta)
        elif isinstance(item, (str, Path)):
            p = Path(item)
            if p.exists() and p.is_dir() and (p / "run_metadata.json").exists():
                resolved_runs.append(load_run_metadata(p))
            else:
                target_dir = find_run_by_timestamp(BASE_RUNS_DIR, str(item))
                if target_dir is None:
                    matches = list(BASE_RUNS_DIR.glob(f"*{item}*"))
                    if matches:
                        target_dir = matches[0]
                if target_dir and target_dir.exists():
                    resolved_runs.append(load_run_metadata(target_dir))
                else:
                    unresolved.append(str(item))

    if unresolved:
        print(f"⚠️ Attenzione: impossibile trovare le seguenti run dal timestamp inserito: {', '.join(unresolved)}")

    if not resolved_runs:
        print("[play2learn] Nessun dato valido trovato per il confronto.")
        return

    print("══════════════════════════════════════════════════════════════════════════════════")
    print(f"{'TIMESTAMP':<18} | {'RUN / MISSIONE':<24} | {'SUCCESSO %':<10} | {'DIST MEDIA':<12} | {'DIST MAX'}")
    print("──────────────────────────────────────────────────────────────────────────────────")
    for r in resolved_runs:
        metrics = r.get("eval_metrics", {})
        miss_data = r.get("mission", "standard")
        if isinstance(miss_data, dict):
            miss = miss_data.get("title", miss_data.get("key", "standard"))[:22]
        else:
            miss = str(miss_data)[:22]
        ts = str(r.get("timestamp", ""))[:17]
        succ = f"{metrics.get('success_rate', metrics.get('eval_success_rate', 0.0)):.1f}%"
        dmean = f"{metrics.get('mean_distance', 0.0):.1f}m"
        dmax = f"{metrics.get('max_distance', 0.0):.1f}m"
        print(f"{ts:<18} | {miss:<24} | {succ:<10} | {dmean:<12} | {dmax}")
    print("══════════════════════════════════════════════════════════════════════════════════")

    # Visualizzazione tabella HTML arricchita su notebook se disponibile IPython
    try:
        from IPython.display import HTML, display
        rows = []
        for r in resolved_runs:
            metrics = r.get("eval_metrics", {})
            succ_val = float(metrics.get("success_rate", metrics.get("eval_success_rate", 0.0)))
            badge_color = "#2e7d32" if succ_val >= 80.0 else ("#f57c00" if succ_val >= 50.0 else "#c62828")
            bg_badge = "#e8f5e9" if succ_val >= 80.0 else ("#fff3e0" if succ_val >= 50.0 else "#ffebee")
            miss_data = r.get("mission", "standard")
            m_title = miss_data.get("title", miss_data.get("key", "standard")) if isinstance(miss_data, dict) else str(miss_data)
            ts = str(r.get("timestamp", ""))
            dmean_val = float(metrics.get("mean_distance", 0.0))
            dmax_val = float(metrics.get("max_distance", 0.0))
            air_val = float(metrics.get("airtime_pct", 0.0))
            rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px; font-family:monospace; font-weight:bold; color:#1a237e; background:#f8f9fa;'>{ts}</td>"
                f"<td style='padding:8px 12px; font-weight:bold;'>{m_title}</td>"
                f"<td style='padding:8px 12px; text-align:center;'><span style='background:{bg_badge}; color:{badge_color}; padding:3px 8px; border-radius:12px; font-weight:bold;'>{succ_val:.1f}%</span></td>"
                f"<td style='padding:8px 12px; text-align:center;'>{dmean_val:.1f} m</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{dmax_val:.1f} m</td>"
                f"<td style='padding:8px 12px; text-align:center;'>{air_val:.1f}%</td>"
                f"</tr>"
            )
        html_str = f"""
        <div style="font-family:'Segoe UI', sans-serif; margin:16px 0;">
            <h3 style="color:#0d47a1; margin-bottom:8px;">📊 Confronto Esperimenti Hill Climb Racing</h3>
            <table style="border-collapse:collapse; width:100%; border:1px solid #ccc; text-align:left; font-size:13.5px;">
                <thead style="background:#e3f2fd; color:#0d47a1;">
                    <tr>
                        <th style="padding:10px 12px;">Timestamp</th>
                        <th style="padding:10px 12px;">Missione / Scenario</th>
                        <th style="padding:10px 12px; text-align:center;">% Successo (&ge; 800m)</th>
                        <th style="padding:10px 12px; text-align:center;">Distanza Media</th>
                        <th style="padding:10px 12px; text-align:center;">Distanza Max</th>
                        <th style="padding:10px 12px; text-align:center;">Tempo in Volo</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """
        display(HTML(html_str))
    except Exception:
        pass


comprimi_runs = comprimi_archivio
mostra_cronologia_run = mostra_cronologia
