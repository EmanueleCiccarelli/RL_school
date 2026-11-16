"""
play2learn_core/agent.py (Optimized Architecture)
- ShmChunkedVecEnv: Zero-pickle /dev/shm SharedMemory + 1-byte os.pipe() sync
- NumpyRolloutForward: Pre-allocated scratch buffers, policy-only & value-only paths
- FastPPO: Pure-NumPy collect_rollouts bypassing per-step PyTorch tensor boxing
- AdaptiveBatchSizeCallback: Switches batch_size (256->512) & late n_epochs (4->3)
- BaseRewardTrackerCallback: Sliding window O(1) telemetry preserved for multi-env compatibility
"""

from __future__ import annotations

import atexit
import collections
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
    return LinearSchedule(initial_value=initial_value, min_value=min_value)


def configure_cpu_threads() -> None:
    """Fissa PyTorch e runtime BLAS/OpenMP a 1 thread CPU e disattiva validazioni superflue."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"
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
    try:
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.emit_nvtx(False)
        torch.autograd.profiler.profile(enabled=False)
    except Exception:
        pass


class AdaptiveBatchSizeCallback(BaseCallback):
    """
    Raddoppia batch_size (256 -> 512) al 40% e riduce n_epochs (4 -> 3) al 65%
    per tagliare ~30-35% dell'overhead SGD CPU nella seconda metà del training.
    """

    def __init__(
        self,
        total_timesteps: int,
        switch_fraction: float = 0.40,
        target_batch_size: int = 512,
        epoch_switch_fraction: float = 0.65,
        target_n_epochs: int = 3,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.switch_step = int(total_timesteps * switch_fraction)
        self.target_batch_size = int(target_batch_size)
        self.epoch_switch_step = int(total_timesteps * epoch_switch_fraction)
        self.target_n_epochs = int(target_n_epochs)
        self._switched_bs = False
        self._switched_ep = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self._switched_bs and self.num_timesteps >= self.switch_step:
            rollout_size = getattr(self.model, "n_steps", 256) * getattr(self.model, "n_envs", 8)
            if rollout_size % self.target_batch_size == 0:
                self.model.batch_size = self.target_batch_size
                self._switched_bs = True
        if not self._switched_ep and self.num_timesteps >= self.epoch_switch_step:
            if getattr(self.model, "n_epochs", 4) > self.target_n_epochs:
                self.model.n_epochs = self.target_n_epochs
                self._switched_ep = True


class NumpyRolloutForward:
    """
    Zero-allocation NumPy/BLAS inference per ActorCritic discreto [64, 64] Tanh.
    Fornisce:
      - forward_numpy(obs_np, det) -> (actions, values, log_probs) puro NumPy
      - predict_numpy(obs_np, det) -> solo policy head (2x più veloce in eval)
      - value_numpy(obs_np)        -> solo value head (per bootstrapping)
      - __call__(obs_tensor, det)  -> retrocompatibile con policy.forward()
    """

    def __init__(self, policy: Any, seed: int = 42):
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self._last_version: int = -1
        self._cache: tuple | None = None
        self._batch_n: int = -1
        self._arange: np.ndarray | None = None
        self._h_pi: np.ndarray | None = None
        self._h_vf: np.ndarray | None = None
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
                    np.ascontiguousarray(w_pi0.detach().cpu().numpy().T, dtype=np.float32),
                    np.ascontiguousarray(p.mlp_extractor.policy_net[0].bias.detach().cpu().numpy(), dtype=np.float32),
                    np.ascontiguousarray(p.mlp_extractor.policy_net[2].weight.detach().cpu().numpy().T, dtype=np.float32),
                    np.ascontiguousarray(p.mlp_extractor.policy_net[2].bias.detach().cpu().numpy(), dtype=np.float32),
                    np.ascontiguousarray(p.action_net.weight.detach().cpu().numpy().T, dtype=np.float32),
                    np.ascontiguousarray(p.action_net.bias.detach().cpu().numpy(), dtype=np.float32),
                    np.ascontiguousarray(p.mlp_extractor.value_net[0].weight.detach().cpu().numpy().T, dtype=np.float32),
                    np.ascontiguousarray(p.mlp_extractor.value_net[0].bias.detach().cpu().numpy(), dtype=np.float32),
                    np.ascontiguousarray(p.mlp_extractor.value_net[2].weight.detach().cpu().numpy().T, dtype=np.float32),
                    np.ascontiguousarray(p.mlp_extractor.value_net[2].bias.detach().cpu().numpy(), dtype=np.float32),
                    np.ascontiguousarray(p.value_net.weight.detach().cpu().numpy().T, dtype=np.float32),
                    np.ascontiguousarray(p.value_net.bias.detach().cpu().numpy(), dtype=np.float32),
                )
            self._last_version = ver
        except Exception:
            self._cache = None

    def invalidate(self) -> None:
        self._last_version = -1

    def _ensure_scratch(self, n: int, h_dim: int) -> None:
        if n != self._batch_n:
            self._batch_n = n
            self._arange = np.arange(n, dtype=np.intp)
            self._h_pi = np.empty((n, h_dim), dtype=np.float32)
            self._h_vf = np.empty((n, h_dim), dtype=np.float32)

    def forward_numpy(
        self, obs_np: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._sync_weights()
        pi_w0, pi_b0, pi_w1, pi_b1, act_w, act_b, vf_w0, vf_b0, vf_w1, vf_b1, val_w, val_b = self._cache
        x = np.asarray(obs_np, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        n = x.shape[0]
        self._ensure_scratch(n, pi_w0.shape[1])

        # Policy MLP in-place
        np.matmul(x, pi_w0, out=self._h_pi)
        self._h_pi += pi_b0
        np.tanh(self._h_pi, out=self._h_pi)
        h2 = self._h_pi @ pi_w1
        h2 += pi_b1
        np.tanh(h2, out=h2)
        logits = h2 @ act_w
        logits += act_b

        # Value MLP in-place
        np.matmul(x, vf_w0, out=self._h_vf)
        self._h_vf += vf_b0
        np.tanh(self._h_vf, out=self._h_vf)
        hv2 = self._h_vf @ vf_w1
        hv2 += vf_b1
        np.tanh(hv2, out=hv2)
        values = (hv2 @ val_w + val_b).ravel()

        # Log-softmax & sampling
        logits -= logits.max(axis=-1, keepdims=True)
        exp_s = np.exp(logits)
        sum_exp = exp_s.sum(axis=-1, keepdims=True)
        log_probs = logits - np.log(sum_exp)

        if deterministic:
            actions = np.argmax(logits, axis=-1).astype(np.int64, copy=False)
        else:
            probs = exp_s / sum_exp
            cdf = np.cumsum(probs, axis=-1)
            cdf[:, -1] = 1.0
            u = self.rng.random((n, 1), dtype=np.float32)
            actions = (cdf < u).sum(axis=-1).astype(np.int64, copy=False)

        chosen_lp = log_probs[self._arange, actions]
        return actions, values, chosen_lp

    def predict_numpy(self, obs_np: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Solo Policy Head (salta Value Net): ~8 µs/step per valutazione."""
        self._sync_weights()
        pi_w0, pi_b0, pi_w1, pi_b1, act_w, act_b = self._cache[:6]
        x = np.asarray(obs_np, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        h = np.tanh(x @ pi_w0 + pi_b0)
        h = np.tanh(h @ pi_w1 + pi_b1)
        logits = h @ act_w + act_b
        if deterministic:
            acts = np.argmax(logits, axis=-1)
        else:
            logits -= logits.max(axis=-1, keepdims=True)
            exp_s = np.exp(logits)
            probs = exp_s / exp_s.sum(axis=-1, keepdims=True)
            cdf = np.cumsum(probs, axis=-1)
            cdf[:, -1] = 1.0
            u = self.rng.random((logits.shape[0], 1), dtype=np.float32)
            acts = (cdf < u).sum(axis=-1)
        return int(acts[0]) if single else acts.astype(np.int64, copy=False)

    def value_numpy(self, obs_np: np.ndarray) -> np.ndarray:
        """Solo Value Head per bootstrapping terminal_observation e last_values."""
        self._sync_weights()
        vf_w0, vf_b0, vf_w1, vf_b1, val_w, val_b = self._cache[6:]
        x = np.asarray(obs_np, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        h = np.tanh(x @ vf_w0 + vf_b0)
        h = np.tanh(h @ vf_w1 + vf_b1)
        return (h @ val_w + val_b).ravel()

    def __call__(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._sync_weights()
        if self._cache is None:
            return self.policy.__class__.forward(self.policy, obs, deterministic=deterministic)
        actions, values, chosen_lp = self.forward_numpy(obs.detach().cpu().numpy(), deterministic=deterministic)
        return (
            torch.from_numpy(actions),
            torch.from_numpy(values[:, None]),
            torch.from_numpy(chosen_lp),
        )


class FastPPO(PPO):
    """
    Sottoclasse PPO 100% compatibile con PPO.save/load che elimina il crossing PyTorch
    durante collect_rollouts scrivendo direttamente nei buffer NumPy del RolloutBuffer.
    """

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps: int) -> bool:
        np_fwd: NumpyRolloutForward | None = getattr(self, "_numpy_rollout_fwd", None)
        if np_fwd is None or np_fwd._cache is None or not isinstance(self._last_obs, np.ndarray):
            return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        np_fwd._sync_weights()

        rollout_buffer.reset()
        callback.on_rollout_start()

        n_envs = env.num_envs
        obs_buf = rollout_buffer.observations
        act_buf = rollout_buffer.actions
        rew_buf = rollout_buffer.rewards
        start_buf = rollout_buffer.episode_starts
        val_buf = rollout_buffer.values
        lp_buf = rollout_buffer.log_probs
        gamma = self.gamma

        n_steps = 0
        while n_steps < n_rollout_steps:
            actions, values, log_probs = np_fwd.forward_numpy(self._last_obs, deterministic=False)
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += n_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)

            # Bootstrapping su timeout (TimeLimit.truncated) in puro NumPy
            if dones.any():
                for idx in np.flatnonzero(dones):
                    info = infos[idx]
                    if info.get("TimeLimit.truncated", False):
                        term_obs = info.get("terminal_observation")
                        if term_obs is not None:
                            rewards[idx] += gamma * float(np_fwd.value_numpy(term_obs)[0])

            # Scrittura diretta zero-overhead nel RolloutBuffer NumPy
            obs_buf[n_steps] = self._last_obs
            act_buf[n_steps] = actions[:, None]
            rew_buf[n_steps] = rewards
            start_buf[n_steps] = self._last_episode_starts
            val_buf[n_steps] = values
            lp_buf[n_steps] = log_probs

            self._last_obs = new_obs
            self._last_episode_starts = dones
            n_steps += 1

        rollout_buffer.pos = n_steps
        rollout_buffer.full = (n_steps == rollout_buffer.buffer_size)

        last_values = np_fwd.value_numpy(new_obs)
        rollout_buffer.compute_returns_and_advantage(
            last_values=torch.from_numpy(last_values), dones=dones
        )
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True


def attach_numpy_rollout_forward(model: Any) -> bool:
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
        model._numpy_rollout_fwd = fwd
        return True
    except Exception:
        return False


def attach_jit_rollout_forward(model: Any) -> bool:
    return attach_numpy_rollout_forward(model)


# =============================================================================
# ZERO-COPY SHARED-MEMORY CHUNKED VECENV (/dev/shm + 1-byte os.pipe signals)
# =============================================================================

def _align8(n: int) -> int:
    return (n + 7) & ~7


def _shm_chunk_worker(
    p2w_r: int,
    w2p_w: int,
    ctrl_remote: Any,
    env_fns: list[Callable],
    shm_name: str,
    layout: dict[str, Any],
    global_indices: np.ndarray,
) -> None:
    configure_cpu_threads()
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = shm.buf

    n_total = layout["n_envs"]
    obs_shape = layout["obs_shape"]
    obs_dtype = np.dtype(layout["obs_dtype"])

    shm_act = np.ndarray((n_total,), dtype=np.int64, buffer=buf, offset=layout["off_act"])
    shm_obs = np.ndarray((n_total, *obs_shape), dtype=obs_dtype, buffer=buf, offset=layout["off_obs"])
    shm_rew = np.ndarray((n_total,), dtype=np.float32, buffer=buf, offset=layout["off_rew"])
    shm_done = np.ndarray((n_total,), dtype=np.uint8, buffer=buf, offset=layout["off_done"])

    idx_list = [int(i) for i in global_indices]
    envs = [fn() for fn in env_fns]

    try:
        while True:
            cmd = os.read(p2w_r, 1)
            if not cmd or cmd == b"C":
                break

            if cmd == b"S":
                term_infos: list[tuple[int, dict[str, Any]]] | None = None
                for env, gi in zip(envs, idx_list):
                    act = int(shm_act[gi])
                    obs, rew, term, trunc, info = env.step(act)
                    done = term or trunc
                    shm_rew[gi] = rew
                    if done:
                        shm_done[gi] = 1
                        info["terminal_observation"] = obs
                        if term_infos is None:
                            term_infos = [(gi, info)]
                        else:
                            term_infos.append((gi, info))
                        obs, _ = env.reset()
                    else:
                        shm_done[gi] = 0
                    shm_obs[gi] = obs

                if term_infos is not None:
                    os.write(w2p_w, b"1")
                    ctrl_remote.send(term_infos)
                else:
                    os.write(w2p_w, b"0")

            elif cmd == b"R":
                seeds = ctrl_remote.recv()
                for local_i, (env, gi) in enumerate(zip(envs, idx_list)):
                    s = seeds[local_i] if seeds is not None else None
                    obs, _ = env.reset(seed=s) if s is not None else env.reset()
                    shm_obs[gi] = obs
                    shm_done[gi] = 0
                os.write(w2p_w, b"0")

            elif cmd == b"M":
                m_cmd, payload = ctrl_remote.recv()
                if m_cmd == "get_attr":
                    ctrl_remote.send([getattr(env, payload, None) for env in envs])
                elif m_cmd == "set_attr":
                    attr_name, val = payload
                    for env in envs:
                        setattr(env, attr_name, val)
                    ctrl_remote.send(None)
                elif m_cmd == "env_method":
                    m_name, m_args, m_kwargs = payload
                    ctrl_remote.send([getattr(env, m_name)(*m_args, **m_kwargs) for env in envs])
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        try:
            os.close(p2w_r)
        except OSError:
            pass
        try:
            os.close(w2p_w)
        except OSError:
            pass
        try:
            ctrl_remote.close()
        except Exception:
            pass
        try:
            shm.close()
        except Exception:
            pass


class ShmChunkedVecEnv(VecEnv):
    """
    Zero-copy SharedMemory VecEnv (2 worker x 4 envs su Colab Linux).
    - Zero pickling su ~97% degli step (solo 1 byte os.write/os.read per worker).
    - Buffer contiguo unico in /dev/shm per massima località L1/L2.
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

        n_envs = len(env_fns)
        obs_shape = obs_space.shape
        obs_dtype = np.dtype(obs_space.dtype)

        off_act = 0
        off_obs = _align8(off_act + n_envs * 8)
        off_rew = _align8(off_obs + n_envs * int(np.prod(obs_shape)) * obs_dtype.itemsize)
        off_done = _align8(off_rew + n_envs * 4)
        total_bytes = _align8(off_done + n_envs)

        self._layout = {
            "n_envs": n_envs,
            "obs_shape": obs_shape,
            "obs_dtype": obs_dtype.str,
            "off_act": off_act,
            "off_obs": off_obs,
            "off_rew": off_rew,
            "off_done": off_done,
        }
        self._closed = False
        self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        atexit.register(self._atexit_unlink)

        buf = self._shm.buf
        self._shm_act = np.ndarray((n_envs,), dtype=np.int64, buffer=buf, offset=off_act)
        self._shm_obs = np.ndarray((n_envs, *obs_shape), dtype=obs_dtype, buffer=buf, offset=off_obs)
        self._shm_rew = np.ndarray((n_envs,), dtype=np.float32, buffer=buf, offset=off_rew)
        self._shm_done = np.ndarray((n_envs,), dtype=np.uint8, buffer=buf, offset=off_done)
        self._shm_done_bool = self._shm_done.view(np.bool_)

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        self._chunks = np.array_split(np.arange(n_envs), min(n_workers, n_envs))

        self._p2w_w: list[int] = []
        self._w2p_r: list[int] = []
        self._ctrl_remotes: list[Any] = []
        self.procs: list[Any] = []

        for idx in self._chunks:
            p2w_r, p2w_w = os.pipe()
            w2p_r, w2p_w = os.pipe()
            p_ctrl, w_ctrl = ctx.Pipe()
            chunk_fns = [env_fns[i] for i in idx]
            p = ctx.Process(
                target=_shm_chunk_worker,
                args=(p2w_r, w2p_w, w_ctrl, chunk_fns, self._shm.name, self._layout, idx),
                daemon=True,
            )
            p.start()
            os.close(p2w_r)
            os.close(w2p_w)
            w_ctrl.close()

            self._p2w_w.append(p2w_w)
            self._w2p_r.append(w2p_r)
            self._ctrl_remotes.append(p_ctrl)
            self.procs.append(p)

        self._empty_info: dict[str, Any] = {}

    def _atexit_unlink(self) -> None:
        try:
            self._shm.unlink()
        except Exception:
            pass

    def reset(self) -> np.ndarray:
        seeds = getattr(self, "_seeds", None)
        for wfd, ctrl, idx in zip(self._p2w_w, self._ctrl_remotes, self._chunks):
            os.write(wfd, b"R")
            ctrl.send([seeds[i] for i in idx] if seeds else None)
        for rfd in self._w2p_r:
            os.read(rfd, 1)
        if hasattr(self, "_reset_seeds"):
            self._reset_seeds()
        return self._shm_obs.copy()

    def step_async(self, actions: np.ndarray) -> None:
        np.copyto(self._shm_act, actions)
        for wfd in self._p2w_w:
            os.write(wfd, b"S")

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        infos: list[dict[str, Any]] = [self._empty_info] * self.num_envs
        for rfd, ctrl in zip(self._w2p_r, self._ctrl_remotes):
            flag = os.read(rfd, 1)
            if flag == b"1":
                for gi, inf in ctrl.recv():
                    infos[gi] = inf
        return self._shm_obs.copy(), self._shm_rew.copy(), self._shm_done_bool.copy(), infos

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for wfd in self._p2w_w:
            try:
                os.write(wfd, b"C")
                os.close(wfd)
            except OSError:
                pass
        for rfd in self._w2p_r:
            try:
                os.close(rfd)
            except OSError:
                pass
        for ctrl in self._ctrl_remotes:
            try:
                ctrl.close()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        try:
            self._shm.close()
        except Exception:
            pass
        try:
            self._shm.unlink()
        except Exception:
            pass

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        target = self._get_indices(indices)
        results = [None] * len(target)
        for wfd, ctrl, idx in zip(self._p2w_w, self._ctrl_remotes, self._chunks):
            os.write(wfd, b"M")
            ctrl.send(("get_attr", attr_name))
            chunk_res = ctrl.recv()
            for local_i, global_i in enumerate(idx):
                if global_i in target:
                    results[target.index(global_i)] = chunk_res[local_i]
        return results

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for wfd, ctrl in zip(self._p2w_w, self._ctrl_remotes):
            os.write(wfd, b"M")
            ctrl.send(("set_attr", (attr_name, value)))
            ctrl.recv()

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        target = self._get_indices(indices)
        results = [None] * len(target)
        for wfd, ctrl, idx in zip(self._p2w_w, self._ctrl_remotes, self._chunks):
            os.write(wfd, b"M")
            ctrl.send(("env_method", (method_name, method_args, method_kwargs)))
            chunk_res = ctrl.recv()
            for local_i, global_i in enumerate(idx):
                if global_i in target:
                    results[target.index(global_i)] = chunk_res[local_i]
        return results

    def env_is_wrapped(self, wrapper_class: Any, indices=None) -> list[bool]:
        return [False] * len(self._get_indices(indices))


# Alias retrocompatibile per moduli che importano ChunkedSubprocVecEnv
ChunkedSubprocVecEnv = ShmChunkedVecEnv


def get_optimal_vec_config(
    n_envs: int | None = None,
    total_timesteps: int = 150_000,
    prefer_subproc: bool | None = None,
) -> tuple[int, bool]:
    cpu_count = os.cpu_count() or 2
    is_linux = sys.platform.startswith("linux")
    use_subproc = prefer_subproc if prefer_subproc is not None else (is_linux and cpu_count >= 2)
    resolved_envs = 8 if n_envs is None else max(1, int(n_envs))
    if resolved_envs == 1:
        use_subproc = False
    return resolved_envs, use_subproc


def make_parallel_envs(
    env_fns: list[Callable[[], Any]],
    use_subproc: bool = False,
    vecnorm_kwargs: dict[str, Any] | None = None,
) -> VecNormalize:
    if use_subproc and len(env_fns) > 1:
        try:
            vec_env = ShmChunkedVecEnv(env_fns, n_workers=min(2, len(env_fns)))
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
