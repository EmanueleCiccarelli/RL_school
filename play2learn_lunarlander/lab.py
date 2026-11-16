"""
Modulo orchestratore di Play2Learn: coordina addestramento, salvataggio, telemetria,
diagnostica e valutazione finale.
Interfaccia pubblica 100% retrocompatibile con i notebook degli studenti.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Any

import torch

from concurrent.futures import ThreadPoolExecutor

_ARTIFACT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p2l_artifact")

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    _LIBS_AVAILABLE = True
except ImportError:
    gym = Any  # type: ignore
    PPO = Any  # type: ignore
    _LIBS_AVAILABLE = False

from play2learn_lunarlander.config import (
    ENV_ID,
    MISSIONS,
    DEFAULT_TIMESTEPS,
    DEFAULT_PPO_PARAMS,
)
from play2learn_lunarlander.environment import (
    make_parallel_envs,
    wrap_vec_normalize,
    CustomRewardWrapper,
    classify_lunar_lander_outcome,
)
from play2learn_lunarlander.agent import (
    create_ppo_agent,
    RewardTrackerCallback,
    AdaptiveBatchSizeCallback,
    attach_jit_rollout_forward,
)
from play2learn_lunarlander.visualizer import plot_training_results, display_notebook_summary
from play2learn_lunarlander.evaluator import evaluate_and_record
from play2learn_lunarlander.storage import (
    create_run_directory,
    save_run_metadata,
    archivia_esperimento_successo,
)
from play2learn_lunarlander.favorites import salva_nei_preferiti, salva_video_preferito
from play2learn_lunarlander.reporting import genera_report_esperimento, genera_report_maturita


def _configure_cpu_threads() -> None:
    """
    Su MLP piccoli (64x64) in CPU, 1 thread PyTorch evita l'overhead di
    sincronizzazione OpenMP/futex e accelera forward+backward di 2-4x rispetto a 8-16 thread.
    Disattiva inoltre la validazione simplex di Categorical per +8-12% throughput CPU.
    """
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        torch.distributions.Distribution.set_default_validate_args(False)
    except Exception:
        pass


class LabResult(dict):
    """
    Oggetto risultato restituito da `run_lab`.
    Si comporta come un normale dizionario Python con supporto aggiuntivo all'accesso
    tramite attributi (es. res.run_dir o res["run_dir"]) e include metodi ausiliari
    per la reportistica, il debug e il confronto dei modelli.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'LabResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def wait_artifacts(self, timeout: float | None = None) -> None:
        """Attende il completamento della generazione in background di video, timelapse e report."""
        fut = self.get("_artifact_future")
        if fut is not None and not fut.done():
            try:
                res = fut.result(timeout=timeout)
                if isinstance(res, dict):
                    if res.get("report_bundle"):
                        self["report_bundle"] = res["report_bundle"]
                        self["maturita_bundle"] = res["report_bundle"]
                    if res.get("meta_path"):
                        self["metadata_path"] = res["meta_path"]
            except Exception as e:
                print(f"[play2learn] Avviso completamento background: {e}")

    def salva_preferito(
        self,
        nome_personalizzato: str | None = None,
        note: str = "",
        tag: str = "Mio Preferito",
    ) -> dict[str, Any]:
        """Salva questo modello nell'archivio dei preferiti personali dell'utente."""
        self.wait_artifacts()
        return salva_nei_preferiti(
            self,
            nome_personalizzato=nome_personalizzato,
            note=note,
            tag=tag,
        )

    def salva_video_preferito(
        self,
        video: int | str | Path = 0,
        nome: str | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        """Salva un video specifico nei preferiti personali dell'utente."""
        self.wait_artifacts()
        target_video = video
        if isinstance(video, int):
            v_list = self.get("success_videos", []) or self.get("recorded_videos", [])
            if v_list and 0 <= video < len(v_list):
                target_video = v_list[video]
            else:
                target_video = self
        return salva_video_preferito(
            video_ref=target_video,
            nome=nome,
            note=note,
        )

    def genera_report(
        self,
        autore: str = "Studente",
        titolo: str = "Report Esperimento Lunar Lander",
        note: str = "",
        force: bool = False,
    ) -> dict[str, Path]:
        """Genera la scheda tecnica riassuntiva in PDF, HTML e Markdown."""
        self.wait_artifacts()
        bundle = self.get("report_bundle")
        if not force and bundle and "rep_files" in bundle:
            if autore == "Studente" and not note and titolo == "Report Esperimento Lunar Lander":
                print("ℹ️ Scheda tecnica già pronta (generata durante l'archiviazione automatica).")
                return bundle["rep_files"]

        rep = genera_report_maturita(
            self,
            autore=autore,
            titolo=titolo,
            note=note,
        )
        self["report_bundle"] = {"rep_files": rep}
        return rep

    def mostra_volo(self, indice: int = 0) -> None:
        """Visualizza uno dei video di prova registrati."""
        self.wait_artifacts()
        videos = self.get("recorded_videos", [])
        if videos and 0 <= indice < len(videos):
            display_notebook_summary(self.get("plot_path"), [videos[indice]])
        else:
            print(f"[play2learn] Nessun video trovato all'indice {indice}.")

    def mostra_evoluzione(self, force: bool = False) -> Any:
        """
        Visualizza (e genera su richiesta se non ancora presente) il timelapse video/GIF
        dell'evoluzione dell'agente nel corso del training.
        """
        self.wait_artifacts()
        existing = self.get("gif_evoluzione_path")
        run_dir = Path(self.get("checkpoint_dir", self.get("run_dir", ".")))

        valid_path = None
        candidates = [
            Path(existing) if existing else None,
            run_dir / "evoluzione_apprendimento.mp4",
            run_dir / "evoluzione_apprendimento.gif",
        ]
        for c in candidates:
            if c and c.exists():
                valid_path = c
                break

        if not force and valid_path:
            self["gif_evoluzione_path"] = valid_path
            return self._display_timelapse(valid_path)

        milestones = self.get("milestone_snapshots")
        model = self.get("model")
        if not milestones or model is None:
            print("[play2learn] Nessun checkpoint o milestone di addestramento disponibile per generare l'evoluzione.")
            return None

        print("🎬 Generazione su richiesta del timelapse dell'evoluzione dell'apprendimento...")
        try:
            from play2learn_lunarlander.evaluator import crea_gif_evoluzione_apprendimento
            target_path = run_dir / "evoluzione_apprendimento.mp4"
            out = crea_gif_evoluzione_apprendimento(
                model=model,
                milestones=milestones,
                env_id=self.get("env_id", ENV_ID),
                output_path=target_path,
                env_kwargs=self.get("env_kwargs"),
                seed=self.get("seed", 42),
            )
            if out and out.exists():
                self["gif_evoluzione_path"] = out
                print(f"🎬 Timelapse salvato: runs/{run_dir.name}/{out.name}")
                return self._display_timelapse(out)
            else:
                print("[play2learn] Impossibile generare il timelapse.")
                return None
        except Exception as e:
            print(f"[play2learn] Avviso durante la generazione del timelapse: {e}")
            return None

    def _display_timelapse(self, media_path: Path) -> Any:
        try:
            import base64
            from IPython.display import HTML, display
            b64_data = base64.b64encode(media_path.read_bytes()).decode("ascii")
            if media_path.suffix.lower() == ".mp4":
                media_tag = (
                    f'<video width="480" controls autoplay loop muted style="border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.3);">'
                    f'<source src="data:video/mp4;base64,{b64_data}" type="video/mp4">'
                    f'</video>'
                )
            else:
                media_tag = f'<img src="data:image/gif;base64,{b64_data}" style="max-width:480px; border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.3);" />'

            html_code = (
                f'<div style="margin-top:10px; margin-bottom:10px; text-align:center; background:#1e1e2f; padding:14px; border-radius:8px; border:1.5px solid #9c27b0;">'
                f'<h4 style="color:#e1bee7; margin-top:2px; margin-bottom:6px;">🎞️ Evoluzione dell\'Apprendimento nel Tempo (Fasi di Training)</h4>'
                f'<p style="color:#ce93d8; font-size:12px; margin-bottom:12px;">Progressione delle traiettorie sullo stesso scenario nelle varie fasi di addestramento:</p>'
                f'{media_tag}'
                f'</div>'
            )
            display(HTML(html_code))
            return media_path
        except Exception:
            print(f"🎞️ Timelapse evoluzione salvato in: {media_path}")
            return media_path

    genera_evoluzione = mostra_evoluzione

    def test_dal_vivo(self, n_episodi: int = 1, seed: int = 999) -> None:
        """Esegue una dimostrazione rapida dal vivo del modello."""
        dimostrazione_live(
            model=self.get("model"),
            env_id=self.get("env_id", ENV_ID),
            reward_fn=self.get("reward_fn"),
            n_episodi=n_episodi,
            seed=seed,
            env_kwargs=self.get("env_kwargs"),
        )


def run_lab(
    reward_fn: Callable,
    mission: str = "luna_standard",
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int | None = None,
    seed: int = 42,
    device: str = "cpu",
    env_id: str = ENV_ID,
    gravity: float | None = None,
    enable_wind: bool | None = None,
    wind_power: float | None = None,
    turbulence_power: float | None = None,
    output_dir: str | Path = "runs",
    nome_run: str | None = None,
    n_eval_episodes: int = 3,
    crea_gif: bool = True,
    crea_gif_evoluzione: bool = False,
    display_in_notebook: bool = True,
    use_reward_norm: bool = True,
) -> LabResult:
    """
    Avvia l'addestramento PPO con la funzione di reward personalizzata,
    salva automaticamente checkpoint, telemetria e grafici nella cartella 'runs/',
    e mostra la dashboard completa nel notebook.
    """
    if not _LIBS_AVAILABLE:
        raise ImportError(
            "❌ Librerie mancanti per l'addestramento!\n"
            "Assicurati di aver eseguito la Cella 3 in cima al notebook:\n"
            "!apt-get install -y -qq swig\n"
            "!pip install -q 'gymnasium[box2d]' stable-baselines3 imageio imageio-ffmpeg tqdm rich"
        )

    _configure_cpu_threads()

    # 1. Configurazione fisica della missione
    mission_cfg = MISSIONS.get(mission, MISSIONS["luna_standard"])
    env_kwargs = dict(mission_cfg["env_kwargs"])
    if gravity is not None:
        env_kwargs["gravity"] = float(gravity)
    if enable_wind is not None:
        env_kwargs["enable_wind"] = bool(enable_wind)
    if wind_power is not None:
        env_kwargs["wind_power"] = float(wind_power)
    if turbulence_power is not None:
        env_kwargs["turbulence_power"] = float(turbulence_power)

    # 2. Creazione della directory della run in runs/
    run_dir = create_run_directory(
        mission=mission,
        base_dir=output_dir,
        custom_name=nome_run,
    )
    video_dir = run_dir / "videos"

    # 3. Creazione degli ambienti paralleli ottimizzati + VecNormalize solo reward
    raw_vec_env = make_parallel_envs(
        env_id=env_id,
        reward_fn=reward_fn,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        total_timesteps=total_timesteps,
    )
    vec_env = raw_vec_env
    t_start = time.perf_counter()
    model = None
    tracker = None

    try:
        actual_n_envs = raw_vec_env.num_envs
        vec_backend = raw_vec_env.__class__.__name__

        vec_env = (
            wrap_vec_normalize(raw_vec_env, gamma=DEFAULT_PPO_PARAMS["gamma"])
            if use_reward_norm
            else raw_vec_env
        )

        print("=" * 76)
        print(f"  🚀 MISSIONE: {mission_cfg['title'].upper()}")
        print(f"  📋 OBIETTIVO: {mission_cfg['description']}")
        print(
            f"  ⚙️  FISICA: Gravità={env_kwargs['gravity']} | Vento={env_kwargs['enable_wind']} "
            f"(Potenza={env_kwargs['wind_power']}, Turbolenza={env_kwargs['turbulence_power']})"
        )
        print(
            f"  ⚡ AMBIENTI: {actual_n_envs} ({vec_backend}) | STEP TOTALI: {total_timesteps:,} | "
            f"RewardNorm: {'ON' if use_reward_norm else 'OFF'}"
        )
        print(f"  📁 SALVATAGGIO RUN IN: runs/{run_dir.name}")
        print("=" * 76)

        # 4. Callback con checkpoint in RAM e progress bar leggera throttled
        tracker = RewardTrackerCallback(
            best_model_save_dir=run_dir,
            total_timesteps=total_timesteps,
            show_progress=True,
            progress_interval_sec=1.0,
            capture_milestones=True,
        )

        # 5. Creazione modello PPO (verbose=0 per output pulito e veloce)
        model = create_ppo_agent(
            env=vec_env,
            seed=seed,
            device=device,
            verbose=0,
        )
        attach_jit_rollout_forward(model)

        adaptive_cb = AdaptiveBatchSizeCallback(
            total_timesteps=total_timesteps,
            switch_fraction=0.40,
            target_batch_size=512,
            epoch_switch_fraction=0.65,
            target_n_epochs=3,
        )

        # 6. Addestramento (progress_bar=False per evitare overhead Rich su Colab)
        print("\n[play2learn] Inizio addestramento dell'agente...")
        t_start = time.perf_counter()
        model.learn(
            total_timesteps=total_timesteps,
            callback=[tracker, adaptive_cb],
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[play2learn] Addestramento interrotto manualmente dall'utente.")
    finally:
        vec_env.close()

    if model is None:
        print("[play2learn] Nessun modello inizializzato prima dell'interruzione.")
        return LabResult({"run_dir": str(run_dir), "model": None, "eval_metrics": {}})

    elapsed_time = time.perf_counter() - t_start
    completed_steps = max(1, getattr(model, "num_timesteps", 0))
    fps_avg = int(completed_steps / max(0.1, elapsed_time))
    print(
        f"[play2learn] Addestramento concluso in {elapsed_time:.1f} secondi "
        f"({fps_avg:,} step/sec)!"
    )

    # 7. Applicazione istantanea in RAM del miglior modello se superiore (< 0.1 ms)
    used_best_weights = tracker.apply_best_weights_if_superior(superiority_margin=5.0)
    if used_best_weights:
        print(
            f"[play2learn] Ripristinato per la valutazione il miglior checkpoint in corsa "
            f"(picco successo mobile: {tracker.best_moving_success_rate:.1f}%)."
        )

    # 8. Generazione grafico telemetria con Figure template pre-allocata e OpenCV (< 100 ms)
    plot_path = run_dir / "reward_plot.png"
    diag_dict = plot_training_results(
        tracker=tracker,
        mission_title=mission_cfg["title"],
        save_path=plot_path,
    )

    # 9. Valutazione rapida da RAM Episode Cache e Replay Video parallelo su 2 vCPU
    eval_stats = evaluate_and_record(
        model=model,
        env_id=env_id,
        reward_fn=reward_fn,
        video_dir=video_dir,
        episode_cache=None if used_best_weights else getattr(tracker, "episode_cache", None),
        target_successes=max(1, n_eval_episodes),
        target_failures=max(1, n_eval_episodes),
        sweep_timeout_seconds=4.0,
        min_eval_episodes=20,
        max_eval_episodes=60,
        seed=seed + 1000,
        env_kwargs=env_kwargs,
        create_gif=False,
        fast_early_exit=True,
        fast_solver=True,
    )

    final_model_path = run_dir / "ppo_lunar_lander.zip"
    best_model_path = run_dir / "best_model.zip"

    # 9b. Generazione opzionale del timelapse GIF dell'evoluzione dell'apprendimento
    gif_evoluzione_path = None
    if crea_gif_evoluzione and tracker.milestone_snapshots:
        try:
            print("\n[play2learn] Generazione del timelapse GIF dell'evoluzione dell'apprendimento...")
            from play2learn_lunarlander.evaluator import crea_gif_evoluzione_apprendimento
            gif_ev_target = run_dir / "evoluzione_apprendimento.mp4"
            gif_evoluzione_path = crea_gif_evoluzione_apprendimento(
                model=model,
                milestones=tracker.milestone_snapshots,
                env_id=env_id,
                output_path=gif_ev_target,
                env_kwargs=env_kwargs,
                seed=seed,
            )
            if gif_evoluzione_path and gif_evoluzione_path.exists():
                print(f"🎬 Timelapse evoluzione salvato: runs/{run_dir.name}/{gif_evoluzione_path.name}")
        except Exception as e:
            print(f"[play2learn] Avviso generazione timelapse evoluzione: {e}")

    # 10. Persistenza asincrona in background di modelli, metadati e archivio successi
    succ_rate = eval_stats.get("eval_success_rate", 0.0)

    def _background_persist() -> dict[str, Any]:
        try:
            model.save(str(final_model_path))
            tracker.flush_best_model_to_disk()
            meta = save_run_metadata(
                run_dir=run_dir,
                mission=mission,
                mission_cfg=mission_cfg,
                env_kwargs=env_kwargs,
                total_timesteps=completed_steps,
                elapsed_seconds=elapsed_time,
                reward_fn=reward_fn,
                eval_stats=eval_stats,
                diagnostics=diag_dict,
            )
            rep_b = None
            if succ_rate >= 80.0:
                succ_info = archivia_esperimento_successo(
                    run_dir=run_dir,
                    nome_run=nome_run or mission,
                    succ_rate=succ_rate,
                    eval_stats=eval_stats,
                )
                rep_b = {
                    "succ_entry": succ_info["entry"],
                    "rep_files": succ_info["rep_files"],
                    "dir_path": succ_info["dest_dir"],
                }
                _display_success_popup(succ_info["entry"], succ_info["rep_files"])
            return {"meta_path": meta, "report_bundle": rep_b}
        except Exception as err:
            print(f"[play2learn] Avviso persistenza background: {err}")
            return {"meta_path": None, "report_bundle": None}

    artifact_future = _ARTIFACT_EXECUTOR.submit(_background_persist)

    # 11. Stampa riassuntiva dei risultati
    n_succ = len(eval_stats.get("success_videos", []))
    n_fail = len(eval_stats.get("failure_videos", []))

    print("\n" + "=" * 76)
    print("  📊 RISULTATI DEI VOLI DI VERIFICA (VALUTAZIONE DETERMINISTICA):")
    print(
        f"  • Tasso di Allunaggio Riuscito: {succ_rate:.1f}% "
        f"({eval_stats.get('eval_successes', 0)} su {eval_stats.get('total_evaluated_episodes', 0)} prove)"
    )
    print(f"  • Video registrati:            {n_succ} allunaggi riusciti 🟢 | {n_fail} fallimenti 🔴")
    print(
        f"  • Durata Media Volo:           {eval_stats['mean_length']:.0f} step "
        f"({eval_stats['mean_length'] / 50:.1f} s)"
    )
    print(
        f"  • Carburante Medio Speso:      {eval_stats['mean_fuel']:.1f} unità "
        f"(Riusciti: {eval_stats.get('mean_fuel_success', 0):.1f} un.)"
    )
    print(
        f"  • Custom Reward Media:         {eval_stats['mean_reward']:.1f} ± "
        f"{eval_stats['std_reward']:.1f}"
    )
    print(f"  💾 Tutti i checkpoint e i video sono archiviati in: runs/{run_dir.name}")
    print("=" * 76 + "\n")

    # 12. GESTIONE OBIETTIVO RAGGIUNTO (SUCCESSO >= 80%): Notifica immediata
    if succ_rate >= 80.0:
        print("🎉 " + "=" * 72)
        print("  🏆 OBIETTIVO ALLUNAGGIO RAGGIUNTO! (Tasso di successo >= 80%)")
        print("  Archiviazione automatica in 'runs/successi/' e generazione Scheda Tecnica...")
        print("=" * 76)

    # 13. Mostra dashboard e video nel Notebook
    if display_in_notebook:
        display_notebook_summary(
            plot_path=plot_path,
            videos=eval_stats.get("videos"),
            success_videos=eval_stats.get("success_videos"),
            failure_videos=eval_stats.get("failure_videos"),
            gif_path=gif_evoluzione_path if (crea_gif_evoluzione and gif_evoluzione_path) else None,
        )

    return LabResult({
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "checkpoint_dir": run_dir,
        "mission": mission,
        "model_path": final_model_path,
        "best_model_path": best_model_path if tracker._best_policy_state is not None else final_model_path,
        "plot_path": plot_path,
        "recorded_videos": eval_stats.get("videos", []),
        "success_videos": eval_stats.get("success_videos", []),
        "failure_videos": eval_stats.get("failure_videos", []),
        "gif_path": eval_stats.get("gif_path"),
        "gif_evoluzione_path": gif_evoluzione_path,
        "milestone_snapshots": getattr(tracker, "milestone_snapshots", []),
        "seed": seed,
        "eval_success_rate": eval_stats.get("eval_success_rate", 0.0),
        "eval_mean_length": eval_stats.get("mean_length", 0.0),
        "eval_mean_fuel": eval_stats.get("mean_fuel", 0.0),
        "eval_stats": eval_stats,
        "diagnosis": diag_dict.get("formatted_text", ""),
        "elapsed_seconds": elapsed_time,
        "total_timesteps": total_timesteps,
        "metadata_path": run_dir / "run_metadata.json",
        "model": model,
        "reward_fn": reward_fn,
        "env_id": env_id,
        "env_kwargs": env_kwargs,
        "report_bundle": None,
        "maturita_bundle": None,
        "_artifact_future": artifact_future,
    })


def _display_success_popup(succ_entry: dict[str, Any], rep_files: dict[str, Path]) -> None:
    """Mostra un box notifica interattivo nel Notebook per il report e i file generati."""
    try:
        from IPython.display import HTML, display
        import html
        pdf_name = html.escape(rep_files.get("pdf_path", Path("scheda_esperimento.pdf")).name)
        succ_dir_name = html.escape(Path(succ_entry.get("dir_path", "")).name)

        box_html = f"""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
                    border: 2px solid #2e7d32; border-radius: 12px; padding: 20px;
                    margin: 18px 0; font-family: sans-serif; box-shadow: 0 4px 15px rgba(46,125,50,0.18);">
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <span style="font-size:32px; margin-right:12px;">🎯</span>
                <div>
                    <h3 style="color:#1b5e20; margin:0; font-size:18px;">
                        OBIETTIVO ALLUNAGGIO RAGGIUNTO (&ge; 80%)!
                    </h3>
                    <p style="color:#2e7d32; margin:2px 0 0 0; font-size:13.5px;">
                        Esperimento archiviato con successo in <code>runs/successi/</code> e Scheda Tecnica generata.
                    </p>
                </div>
            </div>
            <div style="background:#ffffff; border-radius:8px; padding:12px 16px; margin-top:10px; font-size:13px; color:#333;">
                <div>📁 <strong>Cartella Esperimento di Successo:</strong> <code>runs/successi/{succ_dir_name}</code></div>
                <div style="margin-top:5px;">📄 <strong>Scheda Tecnica PDF:</strong> <code>{pdf_name}</code> (con reward usata, metriche e grafici)</div>
                <div style="margin-top:5px;">🎬 <strong>Video di Valutazione:</strong> archiviati in <code>runs/successi/{succ_dir_name}/videos/</code></div>
            </div>
            <div style="margin-top:12px; padding:10px 14px; background:#fff9c4; border-radius:8px; border-left:4px solid #fbc02d; font-size:12.8px; color:#5d4037;">
                ⭐ <strong>I Tuoi Preferiti (Favorites):</strong> La sezione Preferiti è ora riservata esclusivamente ai modelli e video che salvi tu manualmente!<br>
                Se desideri aggiungere questo allunaggio tra i tuoi preferiti, esegui:
                <code>risultati.salva_preferito("Mio Nome", note="Note sul volo")</code>
            </div>
        </div>
        """
        display(HTML(box_html))
    except Exception:
        pass


def dimostrazione_live(
    model: PPO,
    env_id: str = ENV_ID,
    reward_fn: Callable | None = None,
    n_episodi: int = 1,
    seed: int = 999,
    env_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Esegue uno o più episodi dimostrativi stampando a video la telemetria
    in tempo reale per osservare il comportamento dinamico dell'agente.
    """
    if not _LIBS_AVAILABLE:
        raise ImportError(
            "❌ Librerie mancanti per la dimostrazione!\n"
            "Assicurati di aver eseguito la Cella 3 in cima al notebook:\n"
            "!apt-get install -y -qq swig\n"
            "!pip install -q 'gymnasium[box2d]' stable-baselines3 imageio imageio-ffmpeg tqdm rich"
        )

    env_kwargs = env_kwargs or {}
    env = gym.make(env_id, **env_kwargs)
    if reward_fn is not None:
        env = CustomRewardWrapper(env, reward_fn)

    print("\n" + "=" * 65)
    print("  🎬 DIMOSTRAZIONE DI ALLUNAGGIO IN DIRETTA")
    print("=" * 65)

    try:
        for ep in range(n_episodi):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            step = 0
            tot_fuel = 0.0
            last_rew = 0.0
            last_term = False
            last_trunc = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(action)
                step += 1
                done = term or trunc
                last_rew = float(rew)
                last_term = bool(term)
                last_trunc = bool(trunc)

                act_int = int(action)
                if act_int == 2:
                    tot_fuel += 1.0
                elif act_int in (1, 3):
                    tot_fuel += 0.3

                if step % 35 == 0 or done:
                    x, y, vx, vy, angle = obs[0], obs[1], obs[2], obs[3], obs[4]
                    print(
                        f"  Step {step:3d} | x={x:+.2f} | y={y:.2f} | vy={vy:+.2f} | "
                        f"angolo={angle:+.2f} rad | fuel={tot_fuel:.1f}"
                    )

            if reward_fn is not None and isinstance(info, dict):
                is_succ = bool(info.get("is_success", False))
            else:
                un = getattr(env, "unwrapped", env)
                is_succ, _, _, _ = classify_lunar_lander_outcome(
                    obs=obs,
                    original_reward=last_rew,
                    terminated=last_term,
                    truncated=last_trunc,
                    unwrapped=un,
                )

            esito_str = "🎯 ALLUNAGGIO RIUSCITO CON SUCCESSO!" if is_succ else "💥 SCHIANTO / IMPATTO"
            print(f"\n  ESITO VOLO #{ep + 1}: {esito_str}")
            print(f"  Durata: {step} step ({step / 50:.1f} s) | Carburante totale speso: {tot_fuel:.1f} unità")
            print("-" * 65)
    finally:
        env.close()


def mostra_evoluzione(risultati_o_path: Any = None, force: bool = False) -> Any:
    """Funzione helper didattica per mostrare/generare il timelapse dell'evoluzione dell'apprendimento."""
    if risultati_o_path is None:
        print("[play2learn] Passa l'oggetto 'risultati' restituito da run_lab() per visualizzare l'evoluzione.")
        return None
    if isinstance(risultati_o_path, LabResult):
        return risultati_o_path.mostra_evoluzione(force=force)
    p = Path(risultati_o_path)
    if p.is_dir():
        cands = [p / "evoluzione_apprendimento.mp4", p / "evoluzione_apprendimento.gif"]
        for c in cands:
            if c.exists():
                p = c
                break
    if p.is_file():
        lr = LabResult({"checkpoint_dir": str(p.parent), "gif_evoluzione_path": p})
        return lr._display_timelapse(p)
    print(f"[play2learn] Nessun timelapse trovato per: {risultati_o_path}")
    return None


genera_evoluzione = mostra_evoluzione
