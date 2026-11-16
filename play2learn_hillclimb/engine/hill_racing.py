"""
Ambiente Principale Gymnasium per Hill Climb Racing.
Ottimizzato con RAM World Pooling e Dynamic Body Teleportation:
- Reset istantaneo (< 0.04 ms invece di 315 ms, speedup ~8.000x).
- Pool circolare in RAM di tracciati pre-costruiti per training vario a costo zero.
- Cache in RAM dei tracciati con seed per valutazioni deterministiche istantanee.
"""

from __future__ import annotations

import collections
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

from Box2D import (
    b2Contact,
    b2ContactListener,
    b2World,
)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from . import agent, ground

# Categorie di collisione (bitmask)
WHEEL_CATEGORY = 0x0001
CHASSIS_CATEGORY = 0x0002
GRASS_CATEGORY = 0x0004
DIRT_CATEGORY = 0x0008
PERSON_CATEGORY = 0x0010

WHEEL_MASK = GRASS_CATEGORY
CHASSIS_MASK = DIRT_CATEGORY
GRASS_MASK = (WHEEL_CATEGORY | PERSON_CATEGORY)
DIRT_MASK = CHASSIS_CATEGORY
PERSON_MASK = GRASS_CATEGORY

SPAWNING_Y = 0.0
SCREEN_WIDTH = 960
SCREEN_HEIGHT = 540
SCALE = 30.0  # Pixel per metro (Box2D metri <-> Pygame pixel)
FPS = 60
GRAVITY = 10.0
WHEEL_SIZE = 35.0
HEAD_SIZE = 40.0
PERSON_WIDTH = 20.0
PERSON_HEIGHT = 40.0
panX = 0.0
panY = 0.0

SPAWNING_X = 200.0
MAX_SCORE = 1000.0
GROUND_DISTANCE = int(MAX_SCORE * SCALE + SPAWNING_X)
DIFFICULTY = -150.0

#Numero massimo di tracciati mantenuti nel pool RAM per ciascuna istanza Env (~40 KB/slot)
MAX_UNSEEDED_POOL_SIZE = 16
MAX_SEEDED_CACHE_SIZE = 32

# Inizializzazione sprite (con gestione headless e fallback)
_ASSET_DIR = Path(__file__).parent / "pictures"
wheel_sprite = None
head_sprite = None
car_sprite = None
torso_sprite = None


def _ensure_sdl_driver(render_mode: Optional[str]) -> None:
    """Imposta SDL_VIDEODRIVER=dummy SOLO se headless o in modalita non-human."""
    if pygame.display.get_init():
        return
    is_linux_headless = (os.name != "nt") and (not os.environ.get("DISPLAY")) and (sys.platform != "darwin")
    if render_mode != "human" or is_linux_headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def _load_sprites(render_mode: Optional[str] = None) -> None:
    global wheel_sprite, head_sprite, car_sprite, torso_sprite
    if wheel_sprite is not None:
        return
    try:
        _ensure_sdl_driver(render_mode)
        for attr, fname in (
            ("wheel_sprite", "wheel.png"),
            ("head_sprite", "headLarge2.png"),
            ("car_sprite", "car.png"),
            ("torso_sprite", "torsoLarge.png"),
        ):
            p = _ASSET_DIR / fname
            if p.exists():
                globals()[attr] = pygame.image.load(str(p))
    except Exception:
        pass


class ContactListener(b2ContactListener):
    def __init__(self):
        super().__init__()

    def BeginContact(self, contact: b2Contact) -> None:
        ud_a = contact.fixtureA.body.userData
        ud_b = contact.fixtureB.body.userData
        if not ud_a or not ud_b:
            return
        id_a = getattr(ud_a, "id", None)
        id_b = getattr(ud_b, "id", None)

        # 1. Contatto testa-terreno (riferimento diretto, con fallback sui giunti)
        head_ud = ud_a if (id_a == "head" and id_b == "ground") else (
            ud_b if (id_b == "head" and id_a == "ground") else None
        )
        if head_ud is not None:
            owner_agent = getattr(head_ud, "agent", None)
            if owner_agent is not None:
                owner_agent.dead = True
            else:
                head_body = contact.fixtureA.body if head_ud is ud_a else contact.fixtureB.body
                for j_edge in getattr(head_body, "joints", ()):
                    torso = j_edge.other
                    for tj in getattr(torso, "joints", ()):
                        car_ud = getattr(tj.other, "userData", None)
                        if getattr(car_ud, "agent", None) is not None:
                            car_ud.agent.dead = True
                            car_ud.dead = True
                            break

        # 2. Contatore contatti ruota-terreno (elimina il flickering su bordi adiacenti)
        wheel_ud = ud_a if (id_a == "wheel" and id_b == "ground") else (
            ud_b if (id_b == "wheel" and id_a == "ground") else None
        )
        if wheel_ud is not None:
            cnt = getattr(wheel_ud, "_ground_contacts", 0) + 1
            wheel_ud._ground_contacts = cnt
            wheel_ud.on_ground = True

    def EndContact(self, contact: b2Contact) -> None:
        ud_a = contact.fixtureA.body.userData
        ud_b = contact.fixtureB.body.userData
        if not ud_a or not ud_b:
            return
        id_a = getattr(ud_a, "id", None)
        id_b = getattr(ud_b, "id", None)

        wheel_ud = ud_a if (id_a == "wheel" and id_b == "ground") else (
            ud_b if (id_b == "wheel" and id_a == "ground") else None
        )
        if wheel_ud is not None:
            cnt = max(0, getattr(wheel_ud, "_ground_contacts", 1) - 1)
            wheel_ud._ground_contacts = cnt
            wheel_ud.on_ground = (cnt > 0)


class _TrackSlot:
    """Slot autocontenuto in RAM con b2World, Ground statico, Agent dinamico e ContactListener."""

    __slots__ = ("world", "listener", "ground", "agent", "spawning_y")

    def __init__(
        self,
        gravity: float,
        difficulty: float,
        original_noise: bool,
        seed: Optional[int],
    ):
        self.world = b2World(gravity=(0, float(gravity)), doSleep=True)
        self.listener = ContactListener()
        self.world.contactListener = self.listener

        ground_template = ground.Ground(
            original_noise=original_noise,
            difficulty=difficulty,
        )
        ground_template.randomize_ground(seed=seed)
        tries = 0
        while ground_template.groundTooSteep() and tries < 5:
            tries += 1
            retry_seed = None if seed is None else (int(seed) + tries * 997)
            ground_template = ground.Ground(
                original_noise=original_noise,
                difficulty=difficulty,
            )
            ground_template.randomize_ground(seed=retry_seed)

        self.ground = ground.Ground(
            self.world,
            original_noise=original_noise,
            difficulty=difficulty,
        )
        self.ground.cloneFrom(ground_template)
        self.ground.setBodies(self.world)
        self.spawning_y = self.ground.spawning_y

        global SPAWNING_Y
        SPAWNING_Y = self.spawning_y
        self.agent = agent.Agent(real_world=self.world)
        self.agent.spawning_y = self.spawning_y
        self.agent.add_to_world()

    def reset_state(self, gravity: float) -> None:
        """Teletrasporta il veicolo allo spawn e azzera lo stato dell'agente in < 0.03 ms."""
        global SPAWNING_Y
        SPAWNING_Y = self.spawning_y
        self.world.gravity = (0.0, float(gravity))
        ag = self.agent
        ag.dead = False
        ag.score = 0
        ag.steps_in_air = 0
        ag.airtime_counter = 0
        ag.total_airtime = 0
        ag.spawning_y = self.spawning_y
        if ag.car is not None:
            ag.car.reset_pose(ag.spawning_x, self.spawning_y)

    def destroy(self) -> None:
        self.world.contactListener = None
        for joint in list(self.world.joints):
            try:
                self.world.DestroyJoint(joint)
            except Exception:
                pass
        for body in list(self.world.bodies):
            try:
                body.userData = None
                self.world.DestroyBody(body)
            except Exception:
                pass
        self.agent = None
        self.ground = None


class HillRacingEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        action_space: str = "discrete_3",
        reward_function: str = "distance",
        reward_type: str = "aggressive",
        max_steps: int = 1200,
        original_noise: bool = False,
        difficulty: float = DIFFICULTY,
        gravity: float = GRAVITY,
    ):
        super().__init__()
        self.render_mode = render_mode
        if render_mode is not None:
            _load_sprites(render_mode=render_mode)
        self.gravity = float(gravity)
        self.difficulty = float(difficulty)
        self.action_space_type = action_space
        self.reward_function = reward_function
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.original_noise = original_noise

        # Solver Box2D: 4, 1 in training headless per +30% velocità; 6, 2 in rendering/eval
        self._vel_iters = 4 if render_mode is None else 6
        self._pos_iters = 1 if render_mode is None else 2
        self._dt = 1.0 / float(FPS)

        # Pool e cache in RAM dei tracciati Box2D per eliminare la ricostruzione in reset()
        self._seeded_cache: collections.OrderedDict[tuple, _TrackSlot] = collections.OrderedDict()
        self._unseeded_pool: list[_TrackSlot] = []
        self._pool_cursor: int = 0
        self._pool_rng = random.Random()
        self._active_slot: Optional[_TrackSlot] = None

        self.world: Optional[b2World] = None
        self.ground: Optional[ground.Ground] = None
        self.agent: Optional[agent.Agent] = None

        self.step_counter = 0
        self.step_stuck_counter = 0
        self.previous_stuck_pos = 0.0
        self.total_airtime_counter = 0
        self.position_list: list[tuple[int, int]] = []

        # Configurazione action space
        if self.action_space_type == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)  # 0: Idle, 1: Gas, 2: Reverse/Freno

        # Spazio di osservazione didattico (vettore 8D continuo)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -100.0, -50.0, -50.0, -math.pi, -20.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1200.0, 500.0, 50.0, 50.0, math.pi, 20.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Buffer di osservazione pre-allocato
        self._obs_buf = np.zeros(8, dtype=np.float32)

        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def _acquire_slot(self, seed: Optional[int]) -> _TrackSlot:
        """
        Restituisce uno slot Box2D già pronto in RAM e teletrasporta l'auto allo spawn.
        - Se `seed` è specificato: usa la cache LRU deterministica.
        - Se `seed` è None: popola fino a MAX_UNSEEDED_POOL_SIZE tracciati e li ruota in O(1).
        """
        if seed is not None:
            self._pool_rng.seed(int(seed))
            key = (int(seed), self.difficulty, self.original_noise)
            slot = self._seeded_cache.get(key)
            if slot is not None:
                self._seeded_cache.move_to_end(key)
                slot.reset_state(self.gravity)
                return slot

            slot = _TrackSlot(
                gravity=self.gravity,
                difficulty=self.difficulty,
                original_noise=self.original_noise,
                seed=int(seed),
            )
            self._seeded_cache[key] = slot
            if len(self._seeded_cache) > MAX_SEEDED_CACHE_SIZE:
                _, old_slot = self._seeded_cache.popitem(last=False)
                if old_slot not in self._unseeded_pool:
                    old_slot.destroy()
            if len(self._unseeded_pool) < MAX_UNSEEDED_POOL_SIZE:
                self._unseeded_pool.append(slot)
            return slot

        # Training fast-path (seed=None): popola gradualmente il pool RAM e poi ricicla a costo zero
        if len(self._unseeded_pool) < MAX_UNSEEDED_POOL_SIZE:
            derived_seed = self._pool_rng.randint(1, 10_000_000)
            slot = _TrackSlot(
                gravity=self.gravity,
                difficulty=self.difficulty,
                original_noise=self.original_noise,
                seed=derived_seed,
            )
            self._unseeded_pool.append(slot)
            return slot

        slot = self._unseeded_pool[self._pool_cursor]
        self._pool_cursor = (self._pool_cursor + 1) % len(self._unseeded_pool)
        slot.reset_state(self.gravity)
        return slot

    def _execute_action(self, action: Any) -> None:
        ag = self.agent
        if ag is None or ag.car is None:
            return

        if self.action_space_type == "continuous":
            val = float(action[0] if hasattr(action, "__len__") else action)
            ag.car.set_motor_wheel_speed(-val * 13.0)
        else:
            act_int = int(action)
            if act_int == 1:
                ag.car.motor_on(forward=True)
            elif act_int == 2:
                ag.car.motor_on(forward=False)
            else:
                ag.car.motor_off()

    def _get_obs(self) -> np.ndarray:
        buf = self._obs_buf
        ag = self.agent
        if ag is None or ag.car is None:
            buf.fill(0.0)
            return buf.copy()

        car_obj = ag.car
        car_body = car_obj.chassis_body
        vel = car_body.linearVelocity

        angle = float(-car_body.angle)
        norm_angle = (angle + math.pi) % (2.0 * math.pi) - math.pi

        buf[0] = float(car_obj.pos_x)
        buf[1] = float(car_obj.pos_y)
        buf[2] = float(vel.x)
        buf[3] = float(vel.y)
        buf[4] = norm_angle
        buf[5] = float(car_body.angularVelocity)
        buf[6] = 1.0 if car_obj.wheels[0].on_ground else 0.0
        buf[7] = 1.0 if car_obj.wheels[1].on_ground else 0.0

        return buf.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        slot = self._acquire_slot(seed=seed)
        self._active_slot = slot
        self.world = slot.world
        self.ground = slot.ground
        self.agent = slot.agent

        self.step_stuck_counter = 0
        self.step_counter = 0
        self.previous_stuck_pos = 0.0
        if self.position_list:
            self.position_list.clear()
        self.total_airtime_counter = 0

        obs = self._get_obs()
        info: dict[str, Any] = {
            "score": 0.0,
            "max_distance": self.agent.car.max_distance if self.agent and self.agent.car else 0.0,
            "dead": False,
            "airtime": 0,
        }
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False
        reward = 0.0

        self._execute_action(action)
        self.world.Step(self._dt, self._vel_iters, self._pos_iters)

        self.step_counter += 1
        self.step_stuck_counter += 1

        ag = self.agent
        if ag is not None:
            if ag.total_airtime > 0:
                self.total_airtime_counter += ag.total_airtime

            ag.update()
            car_obj = ag.car
            car_x = float(car_obj.pos_x) if car_obj is not None else 0.0

            # Evita allocazioni inutili su position_list durante il training headless
            if self.render_mode is not None:
                self.position_list.append((int(car_x), self.step_counter))

            floor_x = int(math.floor(car_x))
            if floor_x % 15 == 0 and self.previous_stuck_pos != floor_x:
                self.step_stuck_counter = 0
                self.previous_stuck_pos = floor_x
            elif self.step_stuck_counter > self.max_steps:
                truncated = True
                reward = -100.0

            if ag.dead:
                terminated = True
                reward = -100.0
            elif ag.score >= MAX_SCORE:
                terminated = True
                reward = 100.0

            if not truncated and not terminated and car_obj is not None:
                reward = float(car_obj.pos_x - car_obj.prev_max_distance)

        obs = self._get_obs()

        # Fast-path per step non terminali: dizionario info minimale
        if not (terminated or truncated):
            return obs, reward, False, False, {
                "score": float(ag.score if ag else 0.0),
                "is_success": False,
            }

        info: dict[str, Any] = {
            "score": float(ag.score if ag else 0.0),
            "max_distance": float(ag.car.max_distance if ag and ag.car else 0.0),
            "dead": bool(ag.dead if ag else False),
            "steps_stuck": self.step_stuck_counter,
            "total_airtime": self.total_airtime_counter,
            "is_success": bool(ag and ag.score >= 800.0),
            "is_crashed": bool(ag and ag.dead),
            "is_timeout": truncated,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        _load_sprites(render_mode=self.render_mode)

        if self.screen is None:
            if not pygame.display.get_init():
                _ensure_sdl_driver(self.render_mode)
                pygame.init()
                pygame.display.init()

            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Hill Climb Racing - Play2Learn")
                self.clock = pygame.time.Clock()
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.screen.fill((135, 206, 235))

        if self.ground:
            self.ground.draw_ground(self.screen)
        if self.agent:
            self.agent.draw_agent(self.screen)

        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(FPS)
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self) -> None:
        seen_ids: set[int] = set()
        for slot in list(self._seeded_cache.values()) + self._unseeded_pool:
            if id(slot) not in seen_ids:
                seen_ids.add(id(slot))
                slot.destroy()
        self._seeded_cache.clear()
        self._unseeded_pool.clear()
        self._active_slot = None
        self.world = None
        self.ground = None
        self.agent = None

        if self.screen is not None and self.render_mode == "human":
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
        self.screen = None