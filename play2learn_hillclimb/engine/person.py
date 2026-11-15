"""
Modulo Pilota (Person, Head, Torso) con fisica Ragdoll per Hill Climb Racing.
Include collision listener per rilevare impatto della testa contro il suolo.
"""

from __future__ import annotations

import math
from Box2D import (
    b2BodyDef,
    b2CircleShape,
    b2DistanceJointDef,
    b2FixtureDef,
    b2PolygonShape,
    b2RevoluteJointDef,
    b2_dynamicBody,
)
import pygame

from . import hill_racing


class Person:
    def __init__(self, x: float, y: float, person_width: float, person_height: float, world=None):
        self.world = world
        self.x = x
        self.y = y
        self.height = person_height
        self.width = person_width
        self.torso = Torso(x, y - (person_height / 2), person_height, person_width, self.world)
        self.head = Head(x, y - (person_height + person_width), person_width, self.world)

        rev_joint_def = b2RevoluteJointDef()
        joint_angle = [x / hill_racing.SCALE, (y - person_height) / hill_racing.SCALE]
        rev_joint_def.Initialize(bodyA=self.head.body, bodyB=self.torso.body, anchor=joint_angle)
        self.rev_joint_head_torso = self.world.CreateJoint(rev_joint_def)

        dist_joint_def = b2DistanceJointDef()
        anchor_torso = [x / hill_racing.SCALE, y / hill_racing.SCALE]
        anchor_head = [x / hill_racing.SCALE, self.head.starting_position.y / hill_racing.SCALE]
        dist_joint_def.Initialize(
            bodyA=self.head.body,
            bodyB=self.torso.body,
            anchorA=anchor_head,
            anchorB=anchor_torso,
        )
        self.dist_joint_head_torso = self.world.CreateJoint(dist_joint_def)

    def draw_person(self, surface_screen: pygame.Surface) -> None:
        self.head.draw_head(surface_screen)
        self.torso.draw_torso(surface_screen)


class Head:
    def __init__(self, x: float, y: float, r: float, world=None):
        self.world = world
        self.starting_position = pygame.Vector2(x, y)
        self.radius = r
        self.body = None
        self.id = "head"
        self.make_head_body()

    def make_head_body(self) -> None:
        body_def = b2BodyDef()
        body_def.type = b2_dynamicBody
        body_def.position.x = self.starting_position.x / hill_racing.SCALE
        body_def.position.y = self.starting_position.y / hill_racing.SCALE
        body_def.angle = 0
        fix_def = b2FixtureDef(
            categoryBits=hill_racing.PERSON_CATEGORY,
            maskBits=hill_racing.PERSON_MASK,
            density=0.001,
            friction=0.01,
            restitution=0.01,
            shape=b2CircleShape(radius=self.radius / hill_racing.SCALE),
        )
        self.body = self.world.CreateBody(body_def)
        self.body.userData = self
        self.body.CreateFixture(fix_def)

    def draw_head(self, surface_screen: pygame.Surface) -> None:
        pos_x = self.body.position.x * hill_racing.SCALE
        pos_y = self.body.position.y * hill_racing.SCALE
        degrees_angle = math.degrees(-self.body.angle) % 360

        sprite = getattr(hill_racing, "head_sprite", None)
        if sprite is not None:
            scaled = pygame.transform.scale(sprite, (hill_racing.WHEEL_SIZE, hill_racing.WHEEL_SIZE))
            rotated = pygame.transform.rotate(scaled, degrees_angle)
            surface_screen.blit(
                rotated,
                (pos_x - hill_racing.panX - self.radius + 12, pos_y - hill_racing.panY - self.radius + 18),
            )
        else:
            center = (int(pos_x - hill_racing.panX), int(pos_y - hill_racing.panY))
            pygame.draw.circle(surface_screen, (255, 204, 153), center, int(self.radius))


class Torso:
    def __init__(self, center_x: float, center_y: float, height: float, width: float, world=None):
        self.id = "torso"
        self.world = world
        self.width = width
        self.height = height
        self.starting_position = pygame.Vector2(center_x, center_y)
        self.body = None
        self.make_torso_body()

    def make_torso_body(self) -> None:
        body_def = b2BodyDef()
        body_def.type = b2_dynamicBody
        body_def.position.x = self.starting_position.x / hill_racing.SCALE
        body_def.position.y = self.starting_position.y / hill_racing.SCALE
        body_def.angle = 0

        fix_def = b2FixtureDef(
            categoryBits=hill_racing.PERSON_CATEGORY,
            maskBits=hill_racing.PERSON_MASK,
            density=0.002,
            friction=0.01,
            restitution=0.01,
            shape=b2PolygonShape(),
        )
        fix_def.shape.SetAsBox(self.width / 2 / hill_racing.SCALE, self.height / hill_racing.SCALE)

        self.body = self.world.CreateBody(body_def)
        self.body.userData = self
        self.body.CreateFixture(fix_def)

    def draw_torso(self, surface_screen: pygame.Surface) -> None:
        pos_x = self.body.position.x * hill_racing.SCALE
        pos_y = self.body.position.y * hill_racing.SCALE
        degrees_angle = math.degrees(self.body.angle) * -1

        sprite = getattr(hill_racing, "torso_sprite", None)
        if sprite is not None:
            scaled = pygame.transform.scale(sprite, (hill_racing.PERSON_WIDTH, hill_racing.PERSON_HEIGHT))
            rotated = pygame.transform.rotate(scaled, degrees_angle)
            surface_screen.blit(
                rotated,
                (pos_x - hill_racing.panX, pos_y - hill_racing.panY),
            )
        else:
            rect = pygame.Rect(
                int(pos_x - hill_racing.panX - self.width / 2),
                int(pos_y - hill_racing.panY - self.height / 2),
                int(self.width),
                int(self.height),
            )
            pygame.draw.rect(surface_screen, (220, 50, 50), rect)
