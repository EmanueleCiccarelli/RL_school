from __future__ import annotations

from typing import Optional
from Box2D import (
    b2BodyDef,
    b2EdgeShape,
    b2FixtureDef,
    b2PolygonShape,
    b2Vec2,
    b2World,
    b2_staticBody,
)
import numpy as np
import pygame
from . import hill_racing
from . import perlin


class _WallTag:
    def __init__(self, name: str = "wall"):
        self.id = name


class Ground:
    def __init__(
        self,
        world: Optional[b2World] = None,
        original_noise: bool = False,
        difficulty: Optional[float] = None,
    ):
        self.world = world
        self.id = "ground"
        self.ground_vectors: list[b2Vec2] = []
        self.dirtBody = None
        self.grassBody = None
        self.wallBody = None
        self.distance = getattr(hill_racing, "GROUND_DISTANCE", 30200)
        self.x = 0
        self.y = 0
        self.smoothness = 15
        self.grass_thickness = 5
        self.steepness_Level = 0
        self.scaled_ground_vectors: list[b2Vec2] = []
        self.original_noise = original_noise
        self.difficulty = difficulty if difficulty is not None else getattr(hill_racing, "DIFFICULTY", -150.0)
        self.seed = 42

    def randomize_ground(self, seed: Optional[int] = None):
        self.seed = 42 if seed is None else int(seed)
        rng = np.random.default_rng(self.seed & 0xFFFFFFFF)
        ground_seed = float(rng.uniform(0, 100000))
        min_height = 30.0

        x_coords = np.arange(0, self.distance, self.smoothness, dtype=np.float64)
        steepness = np.interp(x_coords, [0, self.distance], [130, 250])
        noise_input = ground_seed + x_coords / (700.0 - steepness)
        noised_vals = np.abs(perlin.generate_perlin_1d(noise_input, seed=self.seed))

        self.ground_vectors = []
        for i, idx in enumerate(x_coords):
            st = steepness[i]
            max_height = self.difficulty + np.interp(st, [0, 200], [0, 320])
            y_val = hill_racing.SCREEN_HEIGHT - np.interp(noised_vals[i], [0.0, 1.0], [min_height, max_height])
            self.ground_vectors.append(b2Vec2(float(idx), float(y_val)))

        self.ground_vectors.append(b2Vec2(float(self.distance), float(hill_racing.SCREEN_HEIGHT)))
        self.ground_vectors.append(b2Vec2(0.0, float(hill_racing.SCREEN_HEIGHT)))
        self.spawning_y = self.ground_vectors[10].y - 100.0
        hill_racing.SPAWNING_Y = self.spawning_y

        for vect in self.ground_vectors:
            vect.x /= hill_racing.SCALE
            vect.y /= hill_racing.SCALE

    def groundTooSteep(self) -> bool:
        if len(self.ground_vectors) < 12:
            return False
        y_vals = np.array([v.y for v in self.ground_vectors[:-2]], dtype=np.float64)
        diffs = np.maximum(0.0, y_vals[:-1] - y_vals[1:])
        rolling_diff = np.convolve(diffs, np.ones(9, dtype=np.float64), mode="valid")
        return bool(np.any(rolling_diff > 5.0))

    def getPositions(self, x: float, numberOfPositions: int, skip: int) -> list[float]:
        returnList: list[float] = []
        for i, v in enumerate(self.ground_vectors):
            if v.x >= x:
                for j in range(0, min(skip * numberOfPositions, len(self.ground_vectors) - i), skip):
                    returnList.append(self.ground_vectors[i + j].y)
                break
        while len(returnList) < numberOfPositions:
            returnList.append(returnList[-1] if returnList else 0.0)
        return returnList

    def cloneFrom(self, otherGround: Ground):
        self.ground_vectors = [b2Vec2(v.x, v.y) for v in otherGround.ground_vectors]
        self.difficulty = otherGround.difficulty
        self.seed = otherGround.seed

    def setBodies(self, worldToAddTo: b2World):
        self.world = worldToAddTo
        self.makeBody()
        for i in range(1, len(self.ground_vectors)):
            self.addEdge(
                self.ground_vectors[i - 1],
                self.ground_vectors[i],
                hill_racing.DIRT_MASK,
                hill_racing.DIRT_CATEGORY,
                False,
            )

        for i in range(1, len(self.ground_vectors)):
            self.addEdge(
                b2Vec2(
                    self.ground_vectors[i - 1].x,
                    self.ground_vectors[i - 1].y - self.grass_thickness / hill_racing.SCALE,
                ),
                b2Vec2(
                    self.ground_vectors[i].x,
                    self.ground_vectors[i].y - self.grass_thickness / hill_racing.SCALE,
                ),
                hill_racing.GRASS_MASK,
                hill_racing.GRASS_CATEGORY,
                True,
            )
        self.create_invisible_wall()

    def create_invisible_wall(self):
        wall_body = b2BodyDef(type=b2_staticBody, position=(0, 0), angle=0)
        wall_fixture = b2FixtureDef(
            density=0,
            friction=0.99,
            restitution=0,
            shape=b2PolygonShape(box=(4, 10000)),
            categoryBits=hill_racing.DIRT_CATEGORY,
            maskBits=hill_racing.WHEEL_CATEGORY | hill_racing.CHASSIS_CATEGORY,
        )
        self.wallBody = self.world.CreateBody(wall_body)
        self.wallBody.userData = _WallTag("wall")
        self.wallBody.CreateFixture(wall_fixture)

    def makeBody(self):
        bodyDef = b2BodyDef(type=b2_staticBody, position=(0, 0))
        self.dirtBody = self.world.CreateBody(bodyDef)
        self.grassBody = self.world.CreateBody(bodyDef)
        self.dirtBody.userData = self
        self.grassBody.userData = self

    def addEdge(self, vec1: b2Vec2, vec2: b2Vec2, mask: int, category: int, isGrass: bool):
        fixDef = b2FixtureDef(
            categoryBits=category,
            maskBits=mask,
            friction=0.99,
            restitution=0.1,
            shape=b2EdgeShape(),
        )
        fixDef.shape.vertices = [vec1, vec2]
        if isGrass:
            self.grassBody.CreateFixture(fixDef)
        else:
            self.dirtBody.CreateFixture(fixDef)

    def draw_ground(self, surface_screen):
        ground_color = (88, 35, 0)
        grass_color = (0, 120, 0)

        # Fresh copy to ensure ground_vectors is NEVER mutated
        if not self.scaled_ground_vectors:
            self.scaled_ground_vectors = [
                b2Vec2(v.x * hill_racing.SCALE, v.y * hill_racing.SCALE)
                for v in self.ground_vectors
            ]

        vertices = [
            (0 - hill_racing.panX, hill_racing.SCREEN_HEIGHT + self.grass_thickness * 2 - hill_racing.panY)
        ]
        for i in range(0, len(self.scaled_ground_vectors) - 2):
            vertices.append(
                (
                    self.scaled_ground_vectors[i].x - hill_racing.panX,
                    self.scaled_ground_vectors[i].y - hill_racing.panY,
                )
            )
        vertices.append(
            (self.distance - hill_racing.panX, hill_racing.SCREEN_HEIGHT + self.grass_thickness * 2 - hill_racing.panY)
        )

        pygame.draw.polygon(surface_screen, ground_color, vertices)
        pygame.draw.polygon(surface_screen, grass_color, vertices, width=self.grass_thickness * 2)
