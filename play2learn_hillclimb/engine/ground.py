"""
Modulo Generazione Terreno Procedurale e Collisioni per Hill Climb Racing.
"""

from __future__ import annotations

import random
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

from . import hill_racing, perlin


class Ground:
    def __init__(self, world: b2World = None, original_noise: bool = False):
        self.world = world
        self.id = "ground"
        self.ground_vectors: list[b2Vec2] = []
        self.dirtBody = None
        self.grassBody = None
        self.distance = hill_racing.GROUND_DISTANCE
        self.x = 0
        self.y = 0
        self.smoothness = 15
        self.grass_thickness = 5
        self.steepness_Level = 0
        self.scaled_ground_vectors: list[b2Vec2] = []
        self.original_noise = original_noise

    def randomize_ground(self, seed: Optional[int] = None) -> None:
        self.ground_vectors.clear()
        rng = random.Random(seed) if seed is not None else random.Random()
        ground_seed = rng.uniform(0, 100000)
        min_height = 30
        flat_length = 500
        height_addition = 0.0

        for i in range(0, self.distance, self.smoothness):
            self.steepness_Level = float(np.interp(i, [0, self.distance], [130, 250]))

            if self.original_noise:
                flat_length = 0
                noised_y = abs(perlin.original_pnoise(ground_seed + (i - flat_length) / (700 - self.steepness_Level)))
            else:
                noised_y = abs(perlin.pnoise1(ground_seed + (i - flat_length) / (700 - self.steepness_Level), octaves=4))

            max_height = hill_racing.DIFFICULTY + float(np.interp(self.steepness_Level, [0, 200], [0, 320]))

            if i < flat_length:
                noised_y = abs(perlin.pnoise1(ground_seed, octaves=4))
                height_addition = (flat_length - i) / 25.0

            y_val = hill_racing.SCREEN_HEIGHT - float(np.interp(noised_y, [0, 1], [min_height, max_height])) + height_addition
            self.ground_vectors.append(b2Vec2(i, y_val))

        self.ground_vectors.append(b2Vec2(self.distance, hill_racing.SCREEN_HEIGHT))
        self.ground_vectors.append(b2Vec2(0, hill_racing.SCREEN_HEIGHT))
        hill_racing.SPAWNING_Y = self.ground_vectors[10].y - 100

        for vect in self.ground_vectors:
            vect.x /= hill_racing.SCALE
            vect.y /= hill_racing.SCALE

    def groundTooSteep(self) -> bool:
        for vector in self.ground_vectors:
            oi = self.getPositions(int(vector.x), 10, 1)
            total_diff = sum(max(0.0, oi[i - 1] - oi[i]) for i in range(1, len(oi)))
            if total_diff > 5.5:
                return True
        return False

    def getPositions(self, x: int, numberOfPositions: int, skip: int) -> list[float]:
        returnList = []
        for i in range(len(self.ground_vectors)):
            if self.ground_vectors[i].x >= x:
                for j in range(0, min(skip * numberOfPositions, len(self.ground_vectors) - i), skip):
                    returnList.append(self.ground_vectors[i + j].y)
                break
        while len(returnList) < numberOfPositions:
            returnList.append(returnList[-1] if returnList else 0.0)
        return returnList

    def cloneFrom(self, otherGround: Ground) -> None:
        self.ground_vectors = [b2Vec2(v.x, v.y) for v in otherGround.ground_vectors]

    def setBodies(self, worldToAddTo: b2World) -> None:
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

    def create_invisible_wall(self) -> None:
        wall_body = b2BodyDef(type=b2_staticBody, position=(0, 0), angle=0)
        wall_fixture = b2FixtureDef(
            density=0,
            friction=0.99,
            restitution=0,
            shape=b2PolygonShape(box=(4, 10000)),
        )
        invisible_wall = self.world.CreateBody(wall_body)
        invisible_wall.userData = self
        invisible_wall.CreateFixture(wall_fixture)

    def makeBody(self) -> None:
        bodyDef = b2BodyDef(type=b2_staticBody, position=(0, 0))
        self.dirtBody = self.world.CreateBody(bodyDef)
        self.grassBody = self.world.CreateBody(bodyDef)
        self.dirtBody.userData = self
        self.grassBody.userData = self

    def addEdge(self, vec1: b2Vec2, vec2: b2Vec2, mask: int, category: int, isGrass: bool) -> None:
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

    def draw_ground(self, surface_screen: pygame.Surface) -> None:
        ground_color = (88, 35, 0)
        grass_color = (34, 160, 34)

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
            (
                self.distance - hill_racing.panX,
                hill_racing.SCREEN_HEIGHT + self.grass_thickness * 2 - hill_racing.panY,
            )
        )

        if len(vertices) >= 3:
            pygame.draw.polygon(surface_screen, ground_color, vertices)
            pygame.draw.polygon(surface_screen, grass_color, vertices, width=self.grass_thickness * 2)
