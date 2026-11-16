"""
Ambiente Principale Gymnasium per Hill Climb Racing.
Supporta modalita headless 'rgb_array' per registrazione video in streaming e Pygame 'human'.
Include gestione difensiva dei contatti Box2D con contatori anti-flicker e cleanup completo della memoria.
"""

from __future__ import annotations

import math
import os
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
        _load_sprites(render_mode=render_mode)
        self.gravity = gravity
        self.world = b2World(gravity=(0, self.gravity), doSleep=True)
        self.ground: Optional[ground.Ground] = None
        self.agent: Optional[agent.Agent] = None
        self.difficulty = difficulty
        self.action_space_type = action_space
        self.reward_function = reward_function
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.original_noise = original_noise

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

        # Spazio di osservazione didattico (vettore 8D continuo, ideale per SB3 MlpPolicy):
        # [0]: x_distanza (m)
        # [1]: y_quota (m)
        # [2]: vx (velocita orizzontale)
        # [3]: vy (velocita verticale)
        # [4]: angolo_inclinazione (radianti [-pi, pi])
        # [5]: velocita_angolare
        # [6]: contatto_ruota_posteriore (0 o 1)
        # [7]: contatto_ruota_anteriore (0 o 1)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -100.0, -50.0, -50.0, -math.pi, -20.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1200.0, 500.0, 50.0, 50.0, math.pi, 20.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def _destroy_world(self) -> None:
        self.world.contactListener = None
        if self.agent:
            try:
                self.agent.destroy_agent()
            except Exception:
                pass
            self.agent = None

        if self.ground:
            for b in (getattr(self.ground, "grassBody", None), getattr(self.ground, "dirtBody", None)):
                if b is not None:
                    try:
                        b.userData = None
                        self.world.DestroyBody(b)
                    except Exception:
                        pass
            self.ground = None

        # Sweep difensivo: garantisce zero body/joint leak in Box2D
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

    def _generate_ground(self, seed: Optional[int] = None) -> None:
        ground_template = ground.Ground(original_noise=self.original_noise)
        ground_template.randomize_ground(seed=seed)
        tries = 0
        while ground_template.groundTooSteep() and tries < 5:
            tries += 1
            retry_seed = None if seed is None else (int(seed) + tries * 997)
            ground_template = ground.Ground(original_noise=self.original_noise)
            ground_template.randomize_ground(seed=retry_seed)

        self.ground = ground.Ground(self.world)
        self.ground.cloneFrom(ground_template)
        self.ground.setBodies(self.world)

    def _generate_agent(self) -> None:
        self.agent = agent.Agent(real_world=self.world)
        self.agent.add_to_world()

    def _execute_action(self, action: Any) -> None:
        if self.agent is None or self.agent.car is None:
            return

        if self.action_space_type == "continuous":
            val = float(action[0] if hasattr(action, "__len__") else action)
            self.agent.car.set_motor_wheel_speed(-val * 13.0)
        else:
            act_int = int(action)
            if act_int == 1:
                self.agent.car.motor_on(forward=True)
            elif act_int == 2:
                self.agent.car.motor_on(forward=False)
            else:
                self.agent.car.motor_off()

    def _get_obs(self) -> np.ndarray:
        if self.agent is None or self.agent.car is None:
            return np.zeros(8, dtype=np.float32)

        car_body = self.agent.car.chassis_body
        pos_x = float(self.agent.car.pos_x)
        pos_y = float(self.agent.car.pos_y)
        vel = car_body.linearVelocity
        ang_vel = float(car_body.angularVelocity)

        # Angolo normalizzato tra -pi e pi
        angle = float(-car_body.angle)
        norm_angle = (angle + math.pi) % (2 * math.pi) - math.pi

        rear_on = 1.0 if getattr(self.agent.car.wheels[0], "on_ground", False) else 0.0
        front_on = 1.0 if getattr(self.agent.car.wheels[1], "on_ground", False) else 0.0

        return np.array(
            [pos_x, pos_y, float(vel.x), float(vel.y), norm_angle, ang_vel, rear_on, front_on],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._destroy_world()
        self.world.contactListener = ContactListener()
        self._generate_ground(seed=seed)
        self._generate_agent()

        self.step_stuck_counter = 0
        self.step_counter = 0
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
        self.world.Step(timeStep=1.0 / FPS, velocityIterations=6, positionIterations=2)

        self.step_counter += 1
        self.step_stuck_counter += 1

        if self.agent is not None:
            if self.agent.total_airtime > 0:
                self.total_airtime_counter += self.agent.total_airtime

            car_x = float(self.agent.car.pos_x) if self.agent.car else 0.0
            self.position_list.append((int(car_x), self.step_counter))
            self.agent.update()

            # Rilevamento blocco nel fosso o pendenza insormontabile
            floor_x = math.floor(car_x)
            if floor_x % 15 == 0 and self.previous_stuck_pos != floor_x:
                self.step_stuck_counter = 0
                self.previous_stuck_pos = floor_x
            elif self.step_stuck_counter > self.max_steps:
                truncated = True
                reward = -100.0

            # Morte del pilota (impatto testa al suolo)
            if self.agent.dead:
                terminated = True
                reward = -100.0
            elif self.agent.score >= MAX_SCORE:
                terminated = True
                reward = 100.0

            # Default baseline reward se non sovrascritta dal wrapper
            if not truncated and not terminated:
                reward = float(self.agent.car.pos_x - self.agent.car.prev_max_distance)

        obs = self._get_obs()
        info: dict[str, Any] = {
            "score": float(self.agent.score if self.agent else 0.0),
            "max_distance": float(self.agent.car.max_distance if self.agent and self.agent.car else 0.0),
            "dead": bool(self.agent.dead if self.agent else False),
            "steps_stuck": self.step_stuck_counter,
            "total_airtime": self.total_airtime_counter,
            "is_success": bool(self.agent and self.agent.score >= 800.0),
            "is_crashed": bool(self.agent and self.agent.dead),
            "is_timeout": truncated,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

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

        # Cielo celeste
        self.screen.fill((135, 206, 235))

        # Disegna terreno e agente
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
        self._destroy_world()
        if self.screen is not None and self.render_mode == "human":
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
        self.screen = None
