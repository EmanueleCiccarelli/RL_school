from gymnasium.error import DependencyNotInstalled

try:
    from Box2D import (
        b2BodyDef,
        b2Contact,
        b2ContactListener,
        b2FixtureDef,
        b2World,
        b2_dynamicBody,
    )
except ImportError:
    raise DependencyNotInstalled("box2d is not installed, try 'pip install box2d box2d-kengz'")

import math
import os
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from . import agent
from . import ground

_ASSET_DIR = Path(__file__).parent / "pictures"

# collisionCategories represented in bits
WHEEL_CATEGORY = 0x0001
CHASSIS_CATEGORY = 0x0002
GRASS_CATEGORY = 0x0004
DIRT_CATEGORY = 0x0008
PERSON_CATEGORY = 0x0010

# collisionMasks, which category it collides with
WHEEL_MASK = GRASS_CATEGORY
CHASSIS_MASK = DIRT_CATEGORY
GRASS_MASK = (WHEEL_CATEGORY | PERSON_CATEGORY)
DIRT_MASK = CHASSIS_CATEGORY
PERSON_MASK = GRASS_CATEGORY

# Fundamental constants (not recommended to change)
SPAWNING_Y = 0  # Spawn location y-coordinate (in pixels), is calculated in ground class
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCALE = 30  # Pixels per meter / Scale, Box2D counts in meters, pygame counts in pixels.
FPS = 60  # frames per second
GRAVITY = 10
WHEEL_SIZE = 35
HEAD_SIZE = 40
PERSON_WIDTH = 20
PERSON_HEIGHT = 40
panX = 0
panY = 0

# Gameplay variables
SPAWNING_X = 200  # Spawn location x-coordinate (in pixels)
MAX_SCORE = 1000  # Max score achievable (-/+ 10)
GROUND_DISTANCE = int(MAX_SCORE * SCALE + SPAWNING_X)  # How long the ground terrain should in pixel size
DIFFICULTY = -150  # Difficulty of terrain, max 30, min -230 (almost flat terrain), -150 is normal difficulty

# Lazy loaded and cached sprites
wheel_sprite = None
head_sprite = None
car_sprite = None
torso_sprite = None
_SCALED_SPRITES: Optional[dict[str, pygame.Surface]] = None


def _load_sprites():
    global wheel_sprite, head_sprite, car_sprite, torso_sprite
    if wheel_sprite is None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        wheel_sprite = pygame.image.load(_ASSET_DIR / "wheel.png")
        head_sprite = pygame.image.load(_ASSET_DIR / "headLarge2.png")
        car_sprite = pygame.image.load(_ASSET_DIR / "car.png")
        torso_sprite = pygame.image.load(_ASSET_DIR / "torsoLarge.png")


def get_scaled_sprites() -> dict[str, pygame.Surface]:
    global _SCALED_SPRITES
    if _SCALED_SPRITES is None:
        _load_sprites()
        _SCALED_SPRITES = {
            "wheel": pygame.transform.scale(wheel_sprite, (WHEEL_SIZE, WHEEL_SIZE)),
            "head": pygame.transform.scale(head_sprite, (WHEEL_SIZE, WHEEL_SIZE)),
            "car": pygame.transform.scale(car_sprite, (125 + 23, 40 * 2 + 10)),
            "torso": pygame.transform.scale(torso_sprite, (PERSON_WIDTH, PERSON_HEIGHT)),
        }
    return _SCALED_SPRITES


# Contact listener for head and ground
class ContactListener(b2ContactListener):
    def __init__(self):
        super().__init__()

    def BeginContact(self, contact: b2Contact) -> None:
        ud_a = contact.fixtureA.body.userData
        ud_b = contact.fixtureB.body.userData
        if ud_a is None or ud_b is None:
            return
        id_a = getattr(ud_a, "id", None)
        id_b = getattr(ud_b, "id", None)

        # Impatto testa-terreno (esclude invisible_wall che ha id="wall")
        if (id_a == "head" and id_b == "ground") or (id_b == "head" and id_a == "ground"):
            head_obj = ud_a if id_a == "head" else ud_b
            ag = getattr(head_obj, "agent", None)
            if ag is not None:
                ag.dead = True
                if ag.car is not None:
                    ag.car.dead = True

        # Contatto ruota-terreno con contatore anti-flickering
        if id_a == "wheel" and id_b == "ground":
            ud_a.ground_contacts += 1
        elif id_b == "wheel" and id_a == "ground":
            ud_b.ground_contacts += 1

    def EndContact(self, contact: b2Contact) -> None:
        ud_a = contact.fixtureA.body.userData
        ud_b = contact.fixtureB.body.userData
        if ud_a is None or ud_b is None:
            return
        id_a = getattr(ud_a, "id", None)
        id_b = getattr(ud_b, "id", None)

        if id_a == "wheel" and id_b == "ground":
            ud_a.ground_contacts = max(0, ud_a.ground_contacts - 1)
        elif id_b == "wheel" and id_a == "ground":
            ud_b.ground_contacts = max(0, ud_b.ground_contacts - 1)


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
        max_steps: int = FPS * 20,
        original_noise: bool = False,
        difficulty: float = DIFFICULTY,
    ):
        self.world = b2World(gravity=(0, GRAVITY), doSleep=True)
        self.ground: Optional[ground.Ground] = None
        self.agent: Optional[agent.Agent] = None
        self.difficulty = difficulty
        self.action_space_type = action_space
        self.reward_function = reward_function
        self.step_stuck_counter = 0
        self.step_counter = 0
        self.max_steps = max_steps
        self.previous_stuck_pos = None
        self.total_airtime_counter = 0
        self.position_list = []
        self.reward_type = reward_type
        self.original_noise = original_noise
        self._next_progress_x = (SPAWNING_X / SCALE) + 20.0

        match self.action_space_type:
            case "discrete_3":
                self.action_space = spaces.Discrete(n=3, start=0)
            case "continuous":
                self.action_space = gym.spaces.Box(low=-13, high=13, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "chassis_position": spaces.Box(
                    low=np.array([0, 0]), high=np.array([1000, 700]), shape=(2,), dtype=np.float32
                ),
                "chassis_angle": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
                "wheels_speed": spaces.Box(
                    low=-13 * math.pi - 0.1, high=13 * math.pi + 0.1, shape=(2,), dtype=np.float32
                ),
                "on_ground": spaces.MultiBinary(n=2),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen: Optional[pygame.Surface] = None
        self.clock = None

    def _destroy_world(self):
        if self.ground is not None:
            self.world.contactListener = None
            if self.ground.grassBody is not None:
                self.world.DestroyBody(self.ground.grassBody)
                self.ground.grassBody = None
            if self.ground.dirtBody is not None:
                self.world.DestroyBody(self.ground.dirtBody)
                self.ground.dirtBody = None
            if getattr(self.ground, "wallBody", None) is not None:
                self.world.DestroyBody(self.ground.wallBody)
                self.ground.wallBody = None
            self.ground = None

        if self.agent is not None:
            self.agent.destroy_agent()
            self.agent = None

    def _generate_ground(self, seed: Optional[int] = None):
        ground_template = ground.Ground(
            original_noise=self.original_noise,
            difficulty=self.difficulty,
        )
        ground_template.randomize_ground(seed=seed)

        tries = 0
        while ground_template.groundTooSteep() and tries < 5:
            tries += 1
            retry_seed = None if seed is None else seed + tries
            ground_template = ground.Ground(
                original_noise=self.original_noise,
                difficulty=self.difficulty,
            )
            ground_template.randomize_ground(seed=retry_seed)

        self.ground = ground.Ground(
            self.world,
            original_noise=self.original_noise,
            difficulty=self.difficulty,
        )
        self.ground.cloneFrom(ground_template)
        self.ground.setBodies(self.world)

    def _generate_agent(self):
        self.agent = agent.Agent(real_world=self.world)
        self.agent.add_to_world()

    def _execute_action(self, action):
        match self.action_space_type:
            case "discrete_3":
                match int(action):
                    case 0:
                        self.agent.car.motor_off()
                    case 1:
                        self.agent.car.motor_on(forward=True)
                    case 2:
                        self.agent.car.motor_on(forward=False)
            case "continuous":
                self.agent.car.set_motor_wheel_speed(action[0])

    def _get_reward(self, action):
        reverse_reward = -1.0 if self.reward_type == "aggressive" else -0.2
        idle_reward = -0.5 if self.reward_type == "aggressive" else -0.1

        match self.reward_function:
            case "distance":
                if self.agent.car.pos_x < self.agent.car.prev_max_distance:
                    return reverse_reward + (self.agent.car.pos_x - self.agent.car.prev_max_distance)
                elif self.agent.car.pos_x - self.agent.car.prev_max_distance < 0.001:
                    return idle_reward
                else:
                    return 1.0 + (self.agent.car.pos_x - self.agent.car.prev_max_distance)
            case "action":
                if action == 0:
                    return idle_reward
                elif action == 1:
                    return 1.0
                elif action == 2:
                    return reverse_reward
                return 0.0
            case _:
                return 0.0

    def _get_obs(self):
        return {
            "chassis_position": np.array(
                [self.agent.car.pos_x, self.agent.car.pos_y], dtype=np.float32
            ),
            "chassis_angle": np.array(
                [(math.degrees(-self.agent.car.chassis_body.angle) % 360)], dtype=np.float32
            ),
            "wheels_speed": np.array(
                [self.agent.car.wheels[0].joint.speed, self.agent.car.wheels[1].joint.speed],
                dtype=np.float32,
            ),
            "on_ground": np.array(
                [int(self.agent.car.wheels[0].on_ground), int(self.agent.car.wheels[1].on_ground)],
                dtype=np.int8,
            ),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        info: dict[str, Any] = {}
        self._destroy_world()
        self.world.contactListener = ContactListener()
        self._generate_ground(seed=seed)
        self._generate_agent()
        self.step_stuck_counter = 0
        self.step_counter = 0
        self._next_progress_x = (SPAWNING_X / SCALE) + 20.0
        self.previous_stuck_pos = None
        self.total_airtime_counter = 0
        self.position_list.clear()

        observations = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return observations, info

    def step(self, action: int | np.float32):
        terminated = False
        truncated = False
        reward = 0.0

        self._execute_action(action)
        # Solver Box2D calibrato a 8/3 per massima efficienza e stabilità
        self.world.Step(
            timeStep=1.0 / self.metadata["render_fps"],
            velocityIterations=8,
            positionIterations=3,
        )
        self.step_stuck_counter += 1
        self.step_counter += 1
        if self.agent.total_airtime > 0:
            self.total_airtime_counter += self.agent.total_airtime

        self.agent.update()

        # Monotonic stuck progress detection
        if self.agent.car.max_distance >= self._next_progress_x:
            self.step_stuck_counter = 0
            self._next_progress_x = self.agent.car.max_distance + 20.0
        elif self.step_stuck_counter > self.max_steps:
            truncated = True
            reward = -100.0

        if self.agent.dead:
            terminated = True
            reward = -100.0
        elif self.agent.score >= MAX_SCORE:
            terminated = True

        if not truncated and not terminated:
            reward = self._get_reward(action)

        observation = self._get_obs()
        info = {
            "car_position": (self.agent.car.pos_x, self.agent.car.pos_y),
            "prev_max_distance": self.agent.car.prev_max_distance,
            "score": self.agent.score,
            "dead": self.agent.car.dead,
            "steps_stuck": self.step_stuck_counter,
            "airtime_counter": self.agent.airtime_counter,
            "total_airtime": self.total_airtime_counter,
            "on_ground": (self.agent.car.wheels[0].on_ground, self.agent.car.wheels[1].on_ground),
        }
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if self.screen is None:
            _load_sprites()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Hill climb Racing RL")
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.screen.fill((135, 206, 235))
        self.ground.draw_ground(self.screen)
        self.agent.draw_agent(self.screen)

        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None
        elif self.render_mode == "rgb_array":
            raw_bytes = pygame.image.tobytes(self.screen, "RGB")
            return np.frombuffer(raw_bytes, dtype=np.uint8).reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 3)).copy()

    def close(self):
        if self.screen is not None:
            if self.render_mode == "human":
                pygame.display.quit()
            self.screen = None
