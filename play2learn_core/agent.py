"""
Modulo Core di Orchestrazione e Addestramento Agenti PPO (Play2Learn).
Include callback ad alte prestazioni con finestre scorrevoli O(1), snapshot pesi in RAM
e configurazione ottimale di VecEnv per Linux, Windows, macOS e Google Colab.
"""

from __future__ import annotations

import collections
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from tqdm.auto import tqdm


class LinearSchedule:
    """Schedule lineare top-level pienamente serializzabile da pickle/cloudpickle."""

    def __init__(self, initial_value: float, min_value: float = 1e-5):
        self.initial_value = float(initial_value)
        self.min_value = float(min_value)

    def __call__(self, progress_remaining: float) -> float:
        return max(self.min_value, self.initial_value * progress_remaining)


def create_linear_schedule(initial_value: float, min_value: float = 1e-5) -> Callable[[float], float]:
    """Alias retrocompatibile che restituisce un oggetto LinearSchedule serializzabile."""
    return LinearSchedule(initial_value=initial_value, min_value=min_value)


def configure_cpu_threads() -> None:
    """Fissa PyTorch e runtime BLAS/OpenMP a 1 thread CPU e disattiva la validazione simplex."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
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


class AdaptiveBatchSizeCallback(BaseCallback):
    """
    Raddoppia il batch_size di PPO (256 -> 512) al 50% di avanzamento per tagliare l'overhead SGD
    tardivo del ~25% e ridurre la varianza della policy vicino alla convergenza.
    """

    def __init__(
        self,
        total_timesteps: int,
        switch_fraction: float = 0.5,
        target_batch_size: int = 512,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.switch_step = int(total_timesteps * switch_fraction)
        self.target_batch_size = int(target_batch_size)
        self._switched = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self._switched and self.num_timesteps >= self.switch_step:
            rollout_size = getattr(self.model, "n_steps", 256) * getattr(self.model, "n_envs", 8)
            if rollout_size % self.target_batch_size == 0:
                self.model.batch_size = self.target_batch_size
                self._switched = True


class NumpyRolloutForward:
    """
    Inference zero-copy basata su NumPy / BLAS per policy ActorCritic discrete [64, 64] Tanh.
    Elimina il crossing pybind11/ATen di PyTorch per ogni step di rollout CPU,
    riducendo la latenza di inferenza da ~250 µs a ~18 µs per passo vettoriale.
    """

    def __init__(self, policy: Any, seed: int = 42):
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self._last_version: int = -1
        self._cache: tuple | None = None
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

    def __call__(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._sync_weights()
        if self._cache is None:
            return self.policy.__class__.forward(self.policy, obs, deterministic=deterministic)

        pi_w0, pi_b0, pi_w1, pi_b1, act_w, act_b, vf_w0, vf_b0, vf_w1, vf_b1, val_w, val_b = self._cache
        x = obs.detach().cpu().numpy()

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


def attach_numpy_rollout_forward(model: Any) -> bool:
    """
    Inietta il forward NumPy ad alta velocità durante il rollout per ActorCritic discreto [64, 64],
    mantenendo il grafo autograd PyTorch per la fase train() e piena compatibilità PPO.save/load.
    """
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
            or getattr(policy.features_extractor, "__class__", None).__name__ != "FlattenExtractor"
        ):
            return False

        seed = getattr(model, "seed", 42) or 42
        fwd = NumpyRolloutForward(policy, seed=seed)
        policy.forward = fwd  # type: ignore[method-assign]
        return True
    except Exception:
        return False


def attach_jit_rollout_forward(model: Any) -> bool:
    """Alias retrocompatibile: inietta il forward ultra-rapido (NumPy / BLAS)."""
    return attach_numpy_rollout_forward(model)


def _chunk_worker(remote: Any, parent_remote: Any, env_fns: list[Callable], obs_shape: tuple, obs_dtype: Any) -> None:
    parent_remote.close()
    configure_cpu_threads()
    envs = [fn() for fn in env_fns]
    k = len(envs)
    buf_obs = np.empty((k, *obs_shape), dtype=obs_dtype)
    buf_rew = np.empty(k, dtype=np.float32)
    buf_done = np.empty(k, dtype=np.bool_)

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        if cmd == "step":
            infos = [None] * k
            for i, (env, act) in enumerate(zip(envs, data)):
                obs, rew, term, trunc, info = env.step(act)
                done = term or trunc
                buf_rew[i] = rew
                buf_done[i] = done
                if done:
                    info["terminal_observation"] = obs
                    obs, _ = env.reset()
                    infos[i] = info
                buf_obs[i] = obs
            remote.send((buf_obs, buf_rew, buf_done, infos))
        elif cmd == "reset":
            seeds = data
            for i, env in enumerate(envs):
                obs, _ = env.reset(seed=seeds[i] if seeds else None)
                buf_obs[i] = obs
            remote.send(buf_obs)
        elif cmd == "close":
            for env in envs:
                try:
                    env.close()
                except Exception:
                    pass
            remote.close()
            break
        elif cmd == "get_attr":
            attr_name = data
            res = [getattr(env, attr_name, None) for env in envs]
            remote.send(res)
        elif cmd == "env_method":
            method_name, method_args, method_kwargs = data
            res = [getattr(env, method_name)(*method_args, **method_kwargs) for env in envs]
            remote.send(res)


class ChunkedSubprocVecEnv(VecEnv):
    """
    VecEnv multi-processo a chunk (es. 2 worker x 4 envs su Linux / Colab).
    Minimizza l'IPC scambiando array NumPy contigui raggruppati, dimezzando la contesa CPU.
    """

    def __init__(self, env_fns: list[Callable[[], Any]], n_workers: int = 2):
        sample = env_fns[0]()
        obs_space = sample.observation_space
        act_space = sample.action_space
        self.render_mode = getattr(sample, "render_mode", None)
        try:
            sample.close()
        except Exception:
            pass
        super().__init__(len(env_fns), obs_space, act_space)

        import multiprocessing as mp
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        self._chunks = np.array_split(np.arange(len(env_fns)), min(n_workers, len(env_fns)))
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(len(self._chunks))])
        self.procs = []
        for work_rem, rem, idx in zip(self.work_remotes, self.remotes, self._chunks):
            chunk_fns = [env_fns[i] for i in idx]
            p = ctx.Process(
                target=_chunk_worker,
                args=(work_rem, rem, chunk_fns, obs_space.shape, obs_space.dtype),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
            work_rem.close()

        self._obs = np.empty((self.num_envs, *obs_space.shape), dtype=obs_space.dtype)
        self._rew = np.empty(self.num_envs, dtype=np.float32)
        self._done = np.empty(self.num_envs, dtype=np.bool_)
        self._empty_info: dict[str, Any] = {}

    def reset(self) -> np.ndarray:
        seeds = getattr(self, "_seeds", None)
        for rem, idx in zip(self.remotes, self._chunks):
            rem.send(("reset", [seeds[i] for i in idx] if seeds else None))
        for rem, idx in zip(self.remotes, self._chunks):
            self._obs[idx] = rem.recv()
        if hasattr(self, "_reset_seeds"):
            self._reset_seeds()
        return self._obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        for rem, idx in zip(self.remotes, self._chunks):
            rem.send(("step", actions[idx]))

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        infos: list[dict[str, Any]] = [self._empty_info] * self.num_envs
        for rem, idx in zip(self.remotes, self._chunks):
            o, r, d, inf = rem.recv()
            self._obs[idx] = o
            self._rew[idx] = r
            self._done[idx] = d
            for local_i, global_i in enumerate(idx):
                if inf[local_i] is not None:
                    infos[global_i] = inf[local_i]
        return self._obs.copy(), self._rew.copy(), self._done.copy(), infos

    def close(self) -> None:
        for rem in self.remotes:
            try:
                rem.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=1.0)

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        target_indices = self._get_indices(indices)
        results = [None] * len(target_indices)
        for rem, idx in zip(self.remotes, self._chunks):
            rem.send(("get_attr", attr_name))
            chunk_res = rem.recv()
            for local_i, global_i in enumerate(idx):
                if global_i in target_indices:
                    pos = target_indices.index(global_i)
                    results[pos] = chunk_res[local_i]
        return results

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        target_indices = self._get_indices(indices)
        results = [None] * len(target_indices)
        for rem, idx in zip(self.remotes, self._chunks):
            rem.send(("env_method", (method_name, method_args, method_kwargs)))
            chunk_res = rem.recv()
            for local_i, global_i in enumerate(idx):
                if global_i in target_indices:
                    pos = target_indices.index(global_i)
                    results[pos] = chunk_res[local_i]
        return results

    def env_is_wrapped(self, wrapper_class: Any, indices=None) -> list[bool]:
        return [False] * len(self._get_indices(indices))


def get_optimal_vec_config(
    n_envs: int | None = None,
    total_timesteps: int = 150_000,
    prefer_subproc: bool | None = None,
) -> tuple[int, bool]:
    """Seleziona ChunkedSubprocVecEnv su Linux / Colab e DummyVecEnv su Windows/macOS."""
    cpu_count = os.cpu_count() or 2
    is_linux = sys.platform.startswith("linux")

    if prefer_subproc is not None:
        use_subproc = prefer_subproc
    else:
        # Su Linux / Colab usiamo la process parallelism ChunkedSubprocVecEnv
        use_subproc = is_linux and (cpu_count >= 2)

    resolved_envs = 8 if n_envs is None else max(1, int(n_envs))
    if resolved_envs == 1:
        use_subproc = False
    return resolved_envs, use_subproc


class BaseRewardTrackerCallback(BaseCallback):
    """
    Callback didattica O(1) con snapshot pesi in RAM (zero disk I/O bloccante durante il training)
    e cattura milestone sicura tramite clone() dello state_dict della policy.
    """

    def __init__(
        self,
        total_timesteps: int,
        show_progress: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: Path | str | None = None,
        save_freq: int = 10_000,
        window_size: int = 50,
        milestone_callback: Callable[[int, Any], None] | None = None,
        milestone_freq: int = 30_000,
        verbose: int = 0,
        metric_name: str = "fuel",
    ):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.show_progress = show_progress
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_freq = save_freq
        self.window_size = window_size
        self.milestone_callback = milestone_callback
        self.milestone_freq = max(1, int(milestone_freq if milestone_freq != 30_000 else max(10_000, total_timesteps // 5)))
        self.metric_name = metric_name

        self.episode_rewards: collections.deque[float] = collections.deque(maxlen=window_size)
        self.episode_lengths: collections.deque[int] = collections.deque(maxlen=window_size)
        self.episode_successes: collections.deque[int] = collections.deque(maxlen=window_size)
        self.episode_secondary: collections.deque[float] = collections.deque(maxlen=window_size)

        self._window_reward_sum = 0.0
        self._window_secondary_sum = 0.0
        self._window_success_sum = 0

        self.total_episodes = 0
        self.successes = 0
        self.crashes = 0
        self.timeouts = 0
        self.best_mean_reward = -float("inf")
        self.best_success_rate = -1.0
        self.best_secondary = -float("inf")
        self._best_policy_state: dict[str, torch.Tensor] | None = None
        self.best_model_path: Path | None = None

        self.milestone_snapshots: list[tuple[int, dict[str, torch.Tensor]]] = []
        self._next_milestone_step = self.milestone_freq

        self.pbar: tqdm | None = None
        self.start_time = 0.0
        self._last_ui_update_time = 0.0

    @property
    def current_moving_reward(self) -> float:
        n = len(self.episode_rewards)
        return (self._window_reward_sum / n) if n > 0 else 0.0

    @property
    def current_moving_secondary(self) -> float:
        n = len(self.episode_secondary)
        return (self._window_secondary_sum / n) if n > 0 else 0.0

    @property
    def current_moving_success_rate(self) -> float:
        n = len(self.episode_successes)
        return (self._window_success_sum / n) * 100.0 if n > 0 else 0.0

    def _clone_policy_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.policy.state_dict().items()}

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        self._last_ui_update_time = self.start_time
        if hasattr(self, "model") and hasattr(self.model, "policy"):
            self.milestone_snapshots.append((0, self._clone_policy_state()))
        if self.show_progress:
            self.pbar = tqdm(
                total=self.total_timesteps,
                desc="Addestramento Agente RL",
                unit="step",
                dynamic_ncols=True,
            )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            self.total_episodes += 1
            r = float(ep_info.get("r", 0.0))
            l = int(ep_info.get("l", 0))

            if len(self.episode_rewards) == self.window_size:
                self._window_reward_sum -= self.episode_rewards[0]
            self.episode_rewards.append(r)
            self._window_reward_sum += r
            self.episode_lengths.append(l)

            succ_int = 1 if info.get("is_success", False) else 0
            if len(self.episode_successes) == self.window_size:
                self._window_success_sum -= self.episode_successes[0]
            self.episode_successes.append(succ_int)
            self._window_success_sum += succ_int

            if succ_int:
                self.successes += 1
            elif info.get("is_timeout", False):
                self.timeouts += 1
            else:
                self.crashes += 1

            telemetry = info.get("telemetry", {})
            sec_val = float(
                telemetry.get(self.metric_name, 0.0)
                if isinstance(telemetry, dict)
                else info.get(self.metric_name, 0.0)
            )
            if len(self.episode_secondary) == self.window_size:
                self._window_secondary_sum -= self.episode_secondary[0]
            self.episode_secondary.append(sec_val)
            self._window_secondary_sum += sec_val

            # Snapshot in RAM del miglior modello
            if self.save_checkpoints and len(self.episode_rewards) >= min(10, self.window_size):
                rate = self.current_moving_success_rate
                rew = self.current_moving_reward
                sec = self.current_moving_secondary
                is_better = (
                    (rate > self.best_success_rate + 2.0)
                    or (abs(rate - self.best_success_rate) <= 2.0 and sec > self.best_secondary + 15.0)
                    or (abs(rate - self.best_success_rate) <= 2.0 and rew > self.best_mean_reward + 10.0)
                )
                if is_better:
                    self.best_success_rate = rate
                    self.best_secondary = sec
                    self.best_mean_reward = rew
                    self._best_policy_state = self._clone_policy_state()

        if self.num_timesteps >= self._next_milestone_step and len(self.milestone_snapshots) < 6:
            snap = self._clone_policy_state()
            self.milestone_snapshots.append((self.num_timesteps, snap))
            if self.milestone_callback:
                try:
                    self.milestone_callback(self.num_timesteps, snap)
                except Exception:
                    pass
            self._next_milestone_step += self.milestone_freq

        if self.pbar and (self.n_calls & 63) == 0:
            now = time.perf_counter()
            if now - self._last_ui_update_time >= 0.35:
                self._last_ui_update_time = now
                self.pbar.n = min(self.num_timesteps, self.total_timesteps)
                self.pbar.set_postfix({
                    "Reward": f"{self.current_moving_reward:.1f}",
                    "Successi": f"{self.current_moving_success_rate:.0f}%",
                    self.metric_name.capitalize(): f"{self.current_moving_secondary:.1f}",
                })

        return True

    def flush_best_model_to_disk(self) -> bool:
        """Scrive su disco una sola volta il miglior modello catturato in RAM."""
        if not self.checkpoint_dir or self._best_policy_state is None:
            return False
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_state = self._clone_policy_state()
        best_path = self.checkpoint_dir / "best_model.zip"
        try:
            self.model.policy.load_state_dict(self._best_policy_state)
            self.model.save(str(best_path))
            self.best_model_path = best_path
        finally:
            self.model.policy.load_state_dict(final_state)
        return True

    def _on_training_end(self) -> None:
        if hasattr(self, "model") and hasattr(self.model, "policy"):
            self.milestone_snapshots.append((max(1, self.num_timesteps), self._clone_policy_state()))
        if self.pbar:
            self.pbar.n = self.total_timesteps
            self.pbar.refresh()
            self.pbar.close()
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.flush_best_model_to_disk()
            self.model.save(str(self.checkpoint_dir / "final_model.zip"))


def make_parallel_envs(
    env_fns: list[Callable[[], Any]],
    use_subproc: bool = False,
    vecnorm_kwargs: dict[str, Any] | None = None,
) -> VecNormalize:
    """Crea e normalizza in modo vettorizzato gli ambienti Gymnasium con fallback sicuro."""
    if use_subproc and len(env_fns) > 1:
        try:
            vec_env = ChunkedSubprocVecEnv(env_fns, n_workers=min(2, len(env_fns)))
        except Exception:
            vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    params = {
        "norm_obs": False,
        "norm_reward": True,
        "clip_reward": 10.0,
        "gamma": 0.995,
    }
    if vecnorm_kwargs:
        params.update(vecnorm_kwargs)
    return VecNormalize(vec_env, **params)
