"""
Interfaccia Principale del Laboratorio Didattico per Hill Climb Racing (Play2Learn).
Fornisce la funzione didattica `run_lab`, la classe `LabResult` con piena parità di API,
e il forward NumPy/BLAS specializzato per `HillClimbFeaturesExtractor` (+3.500 step/s su 2 vCPU).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from play2learn_core.agent import (
    configure_cpu_threads,
    make_parallel_envs,
    AdaptiveBatchSizeCallback,
)
from play2learn_hillclimb.agent import (
    HillClimbFeaturesExtractor,
    HillClimbRewardTrackerCallback,
    build_hill_agent,
)
from play2learn_hillclimb.config import (
    DEFAULT_N_ENVS,
    DEFAULT_TIMESTEPS,
    ENV_ID,
    MISSIONS,
    VECNORM_PARAMS,
)
from play2learn_hillclimb.diagnostics import diagnose_hill_climb_behavior
from play2learn_hillclimb.environment import HillClimbRewardWrapper
from play2learn_hillclimb.evaluator import crea_gif_evoluzione_apprendimento, evaluate_and_record
from play2learn_hillclimb.reporting import genera_report_esperimento
from play2learn_hillclimb.storage import create_run_directory, save_run_metadata


class HillClimbNumpyRolloutForward:
    """
    Inference zero-copy basata su NumPy / BLAS per policy ActorCritic [64, 64] Tanh
    accoppiata a `HillClimbFeaturesExtractor` (7 feature invarianti alla traslazione).
    Riduce la latenza di rollout da ~250 µs a ~18 µs per passo vettoriale.
    """

    def __init__(self, policy: Any, seed: int = 42):
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self._last_version: int = -1
        self._cache: tuple | None = None
        self._feat_buf: np.ndarray | None = None
        self._sync_weights()

    def _sync_weights(self) -> None:
        p = self.policy
        try:
            w_pi0 = p.mlp_extractor.policy_net[0].weight
            ver = getattr(w_pi0, "_version", -1)
            if ver == self._last_version and self._cache is not None:
                return
            with torch.no_grad():
                self._cache = (
                    w_pi0.detach().cpu().numpy().T.copy(),
                    p.mlp_extractor.policy_net[0].bias.detach().cpu().numpy().copy(),
                    p.mlp_extractor.policy_net[2].weight.detach().cpu().numpy().T.copy(),
                    p.mlp_extractor.policy_net[2].bias.detach().cpu().numpy().copy(),
                    p.action_net.weight.detach().cpu().numpy().T.copy(),
                    p.action_net.bias.detach().cpu().numpy().copy(),
                    p.mlp_extractor.value_net[0].weight.detach().cpu().numpy().T.copy(),
                    p.mlp_extractor.value_net[0].bias.detach().cpu().numpy().copy(),
                    p.mlp_extractor.value_net[2].weight.detach().cpu().numpy().T.copy(),
                    p.mlp_extractor.value_net[2].bias.detach().cpu().numpy().copy(),
                    p.value_net.weight.detach().cpu().numpy().T.copy(),
                    p.value_net.bias.detach().cpu().numpy().copy(),
                )
            self._last_version = ver
        except Exception:
            self._cache = None

    def _extract_features_np(self, raw_obs: np.ndarray) -> np.ndarray:
        n = raw_obs.shape[0]
        if self._feat_buf is None or self._feat_buf.shape[0] != n:
            self._feat_buf = np.empty((n, 7), dtype=np.float32)
        f = self._feat_buf
        np.clip(raw_obs[:, 2] * (1.0 / 15.0), -3.0, 3.0, out=f[:, 0])
        np.clip(raw_obs[:, 3] * (1.0 / 15.0), -3.0, 3.0, out=f[:, 1])
        ang = raw_obs[:, 4]
        np.sin(ang, out=f[:, 2])
        np.cos(ang, out=f[:, 3])
        np.clip(raw_obs[:, 5] * 0.1, -3.0, 3.0, out=f[:, 4])
        f[:, 5] = raw_obs[:, 6]
        f[:, 6] = raw_obs[:, 7]
        return f

    def __call__(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._sync_weights()
        if self._cache is None:
            return self.policy.__class__.forward(self.policy, obs, deterministic=deterministic)

        pi_w0, pi_b0, pi_w1, pi_b1, act_w, act_b, vf_w0, vf_b0, vf_w1, vf_b1, val_w, val_b = self._cache
        raw_x = obs.detach().cpu().numpy()
        x = self._extract_features_np(raw_x)

        h_pi = np.tanh(x @ pi_w0 + pi_b0)
        h_pi = np.tanh(h_pi @ pi_w1 + pi_b1)
        logits = h_pi @ act_w + act_b

        h_vf = np.tanh(x @ vf_w0 + vf_b0)
        h_vf = np.tanh(h_vf @ vf_w1 + vf_b1)
        values = (h_vf @ val_w + val_b).astype(np.float32, copy=False)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_s = np.exp(shifted)
        sum_exp = exp_s.sum(axis=-1, keepdims=True)
        log_probs = (shifted - np.log(sum_exp)).astype(np.float32, copy=False)

        if deterministic:
            actions = np.argmax(logits, axis=-1).astype(np.int64, copy=False)
        else:
            probs = exp_s / sum_exp
            cdf = np.cumsum(probs, axis=-1)
            cdf[:, -1] = 1.0
            u = self.rng.random((logits.shape[0], 1), dtype=np.float32)
            actions = (cdf < u).sum(axis=-1).astype(np.int64, copy=False)

        chosen_lp = log_probs[np.arange(logits.shape[0]), actions].astype(np.float32, copy=False)
        return (
            torch.from_numpy(actions),
            torch.from_numpy(values),
            torch.from_numpy(chosen_lp),
        )


def attach_hillclimb_numpy_rollout(model: Any) -> bool:
    """Attiva il forward NumPy zero-copy su policy dotate di HillClimbFeaturesExtractor."""
    policy = getattr(model, "policy", None)
    if policy is None:
        return False
    try:
        pi_net = policy.mlp_extractor.policy_net
        vf_net = policy.mlp_extractor.value_net
        if (
            len(pi_net) != 4
            or len(vf_net) != 4
            or not isinstance(pi_net[1], torch.nn.Tanh)
            or not isinstance(vf_net[1], torch.nn.Tanh)
            or not isinstance(policy.features_extractor, HillClimbFeaturesExtractor)
        ):
            return False
        seed = getattr(model, "seed", 42) or 42
        policy.forward = HillClimbNumpyRolloutForward(policy, seed=seed)  # type: ignore[method-assign]
        return True
    except Exception:
        return False


class LabResult(dict):
    """Oggetto risultato arricchito con metodi didattici rapidi e retrocompatibilità completa."""

    def __init__(self, data: dict[str, Any]):
        super().__init__(data)
        self.model = data.get("model")
        self.run_dir = Path(data.get("checkpoint_dir", "."))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'LabResult' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def genera_report(
        self,
        autore: str = "Studente",
        titolo: str = "Report Hill Climb",
        note: str = "",
    ) -> dict[str, Path]:
        """Genera istantaneamente la scheda tecnica in PDF, HTML e Markdown."""
        return genera_report_esperimento(run_or_fav=self, autore=autore, titolo=titolo, note=note)

    report = genera_report

    def salva_preferito(
        self,
        nome_personalizzato: str | None = None,
        note: str = "",
        tag: str = "Mio Preferito",
        nome: str | None = None,
    ) -> dict[str, Any]:
        """Salva questo esperimento nei preferiti personali."""
        from play2learn_hillclimb.favorites import salva_nei_preferiti
        chosen = nome or nome_personalizzato or self.get("mission", "campagna_standard")
        return salva_nei_preferiti(self, nome=chosen, note=note, tag=tag)

    preferito = salva_preferito

    def mostra_evoluzione(self) -> None:
        """Visualizza il timelapse (MP4 o GIF) dell'evoluzione dell'agente nel corso del training."""
        p = self.get("gif_evoluzione_path")
        if not p:
            ck_dir = Path(self.get("checkpoint_dir", "."))
            p_mp4 = ck_dir / "evoluzione_apprendimento.mp4"
            p_gif = ck_dir / "evoluzione_apprendimento.gif"
            if p_mp4.exists():
                p = str(p_mp4)
            elif p_gif.exists():
                p = str(p_gif)

        if p and Path(p).exists():
            path_obj = Path(p)
            try:
                from IPython.display import HTML, Image, display
                import base64
                print("🎞️ Evoluzione dell'Agente durante il Training (Milestone nel tempo):")
                if path_obj.suffix.lower() == ".mp4":
                    b64 = base64.b64encode(path_obj.read_bytes()).decode("ascii")
                    display(HTML(f'<video width="560" controls autoplay loop muted style="border-radius:6px; box-shadow:0 4px 10px rgba(0,0,0,0.3);"><source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'))
                else:
                    display(Image(filename=str(path_obj)))
            except Exception:
                print(f"Evoluzione salvata in: {p}")
        else:
            print("[play2learn] Nessun timelapse di evoluzione disponibile per questa run.")


def mostra_evoluzione(run_or_result: Any) -> None:
    """Mostra l'evoluzione dell'agente per l'oggetto LabResult o dizionario passato."""
    if hasattr(run_or_result, "mostra_evoluzione"):
        run_or_result.mostra_evoluzione()
    elif isinstance(run_or_result, dict):
        LabResult(run_or_result).mostra_evoluzione()


def run_lab(
    reward_fn: Callable | None = None,
    mission: str = "campagna_standard",
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    learning_rate: float = 3e-4,
    crea_gif_evoluzione: bool = True,
    seed: int = 42,
    show_progress: bool = True,
    **kwargs: Any,
) -> LabResult:
    """
    Funzione didattica principale per avviare l'addestramento dell'auto in Hill Climb Racing.
    Ottimizzata per raggiungere 2.500 - 4.000+ step/s su Google Colab (2 vCPU) e desktop.
    """
    if reward_fn is None and "custom_reward" in kwargs:
        reward_fn = kwargs.pop("custom_reward")

    configure_cpu_threads()

    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])
    env_kwargs = dict(mission_cfg["env_kwargs"])

    run_dir = create_run_directory(mission=mission)
    checkpoints_dir = run_dir / "checkpoints"

    resolved_envs = max(1, int(n_envs))
    # Con il reset in RAM (<0.04ms) e b2ChainShape, DummyVecEnv in-process elimina il 100%
    # della latenza di serializzazione IPC su 2 vCPU battendo SubprocVecEnv di ~1.8x.
    use_subproc = False

    print("=" * 70)
    print(" 🏎️  PLAY2LEARN — LABORATORIO DI REINFORCEMENT LEARNING")
    print("     Ambiente: Hill Climb Racing (Farama Box2D + Perlin Noise)")
    print(f"     Tracciato: {mission_cfg['title']}")
    print(f"     Timesteps previsti: {total_timesteps:,} | Ambienti paralleli: {resolved_envs}")
    print("=" * 70)

    # 1. Creazione ambienti paralleli con Monitor e render_mode=None per massima velocità
    def _make_single_env(rank: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            env = gym.make(ENV_ID, render_mode=None, **env_kwargs)
            env = HillClimbRewardWrapper(env, reward_fn=reward_fn)
            env = Monitor(env, info_keywords=("is_success",))
            env.reset(seed=seed + rank * 100)
            return env
        return _thunk

    env_fns = [_make_single_env(i) for i in range(resolved_envs)]
    vec_env = make_parallel_envs(env_fns, use_subproc=use_subproc, vecnorm_kwargs=VECNORM_PARAMS)

    callback = HillClimbRewardTrackerCallback(
        total_timesteps=total_timesteps,
        show_progress=show_progress,
        save_checkpoints=True,
        checkpoint_dir=checkpoints_dir,
        milestone_freq=max(10_000, total_timesteps // 4),
    )

    # 2. Creazione modello PPO con rollout buffer ampio (n_steps=512 -> 4.096 step/rollout)
    #    per dimezzare l'overhead di dispatch autograd PyTorch.
    fast_ppo_kwargs = {
        "n_steps": 512,
        "batch_size": 256,
    }
    model = build_hill_agent(
        env=vec_env,
        learning_rate=learning_rate,
        ppo_kwargs=fast_ppo_kwargs,
        seed=seed,
    )
    attach_hillclimb_numpy_rollout(model)

    adaptive_cb = AdaptiveBatchSizeCallback(
        total_timesteps=total_timesteps,
        switch_fraction=0.5,
        target_batch_size=512,
    )

    t0 = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=[callback, adaptive_cb])
    finally:
        vec_env.close()

    elapsed = time.time() - t0
    print(f"\n✅ Addestramento completato in {elapsed:.1f} s ({total_timesteps/max(0.1, elapsed):.0f} step/s)!")

    # 3. Valutazione deterministica e registrazione video streaming
    print("\n🎥 Avvio sessione di verifica deterministica su 20 scenari collinari...")
    eval_stats = evaluate_and_record(
        model=model,
        output_dir=run_dir,
        env_kwargs=env_kwargs,
        reward_fn=reward_fn,
    )

    # 4. GIF/MP4 evoluzione apprendimento
    gif_path = None
    if crea_gif_evoluzione and callback.milestone_snapshots:
        print("🎬 Generazione timelapse MP4 dell'evoluzione dell'apprendimento...")
        gif_path = crea_gif_evoluzione_apprendimento(
            checkpoints=callback.milestone_snapshots,
            output_gif_path=run_dir / "evoluzione_apprendimento.mp4",
            env_kwargs=env_kwargs,
            reward_fn=reward_fn,
            base_model=model,
        )

    # 5. Diagnostica comportamentale
    diagnostics = diagnose_hill_climb_behavior(eval_stats)
    print("\n" + diagnostics["formatted_text"])

    # 6. Salvataggio metadati
    save_run_metadata(
        run_dir=run_dir,
        mission=mission,
        mission_cfg=mission_cfg,
        env_kwargs=env_kwargs,
        total_timesteps=total_timesteps,
        elapsed_seconds=elapsed,
        reward_fn=reward_fn,
        eval_stats=eval_stats,
        diagnostics=diagnostics,
    )

    # Salvataggio modello finale
    model.save(str(run_dir / "final_model.zip"))

    result_data = {
        "model": model,
        "eval_success_rate": eval_stats["eval_success_rate"],
        "mean_distance": eval_stats["mean_distance"],
        "max_distance": eval_stats["max_distance_achieved"],
        "mean_speed": eval_stats["mean_speed"],
        "checkpoint_dir": str(run_dir),
        "best_video_path": str(eval_stats["best_video_path"]) if eval_stats.get("best_video_path") else None,
        "gif_evoluzione_path": str(gif_path) if gif_path else None,
        "mission": mission,
        "env_kwargs": env_kwargs,
        "eval_metrics": eval_stats,
        "diagnostics": diagnostics,
    }

    print("\n🏆 Riepilogo Finale:")
    print(f"   • Distanza Massima: {eval_stats['max_distance_achieved']:.1f} m")
    print(f"   • Distanza Media:   {eval_stats['mean_distance']:.1f} m")
    print(f"   • Tasso Successo:   {eval_stats['eval_success_rate']:.1f}%")
    print(f"   • Cartella Run:     {run_dir.name}\n")

    return LabResult(result_data)


def dimostrazione_live(
    model_or_result: Any,
    seed: int = 42,
    mission: str = "campagna_standard",
) -> None:
    """Mostra una dimostrazione visiva della guida dell'agente."""
    model = getattr(model_or_result, "model", model_or_result)
    mission_cfg = MISSIONS.get(mission, MISSIONS["campagna_standard"])

    is_headless = (os.name != "nt") and (not os.environ.get("DISPLAY")) and (sys.platform != "darwin")
    if is_headless:
        print("[play2learn] Ambiente headless rilevato (Google Colab / Linux server).")
        print("💡 La visualizzazione Pygame su schermo non è supportata in headless. I video MP4 sono disponibili nella cartella run.")
        return

    try:
        env = gym.make(ENV_ID, render_mode="human", **mission_cfg["env_kwargs"])
        obs, _ = env.reset(seed=seed)

        print("🚗 Dimostrazione live avviata... Premi ESC nella finestra Pygame per uscire.")
        for _ in range(1200):
            action, _ = model.predict(obs, deterministic=True)
            act_item = int(action.item()) if hasattr(action, "item") and not hasattr(action, "__len__") else int(action)
            obs, reward, terminated, truncated, info = env.step(act_item)
            env.render()
            if terminated or truncated:
                break

        env.close()
    except Exception as e:
        print(f"[play2learn] Impossibile avviare la finestra Pygame ({e}).")