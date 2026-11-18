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
    def __init__(self, x, y, person_width, person_height, world=None, agent=None):
        self.world = world
        self.agent = agent
        self.x = x
        self.y = y
        self.height = person_height
        self.width = person_width
        self.torso = Torso(x, y - (person_height / 2), person_height, person_width, self.world)
        self.head = Head(x, y - (person_height + person_width * 1), person_width * 1, self.world, agent=self.agent)

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

    def draw_person(self, surface_screen):
        self.head.draw_head(surface_screen)
        self.torso.draw_torso(surface_screen)


class Head:
    def __init__(self, x, y, r, world=None, agent=None):
        self.world = world
        self.agent = agent
        self.starting_position = pygame.Vector2(x, y)
        self.radius = r
        self.body = None
        self.id = "head"
        self.is_CB = False
        self.make_head_body()

    def make_head_body(self):
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

    def draw_head(self, surface_screen):
        pos_x = self.body.position.x * hill_racing.SCALE
        pos_y = self.body.position.y * hill_racing.SCALE
        degrees_angle = math.degrees(-self.body.angle) % 360
        sprite = hill_racing.get_scaled_sprites()["head"]
        rotated_head_sprite = pygame.transform.rotate(sprite, degrees_angle)
        surface_screen.blit(
            source=rotated_head_sprite,
            dest=(pos_x - hill_racing.panX - self.radius + 12, pos_y - hill_racing.panY - self.radius + 18),
        )


class Torso:
    def __init__(self, center_x, center_y, height, width, world):
        self.id = "torso"
        self.world = world
        self.width = width
        self.height = height
        self.starting_position = pygame.Vector2(center_x, center_y)
        self.body = None
        self.make_torso_body()

    def make_torso_body(self):
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

    def draw_torso(self, surface_screen):
        pos_x = self.body.position.x * hill_racing.SCALE
        pos_y = self.body.position.y * hill_racing.SCALE
        degrees_angle = math.degrees(self.body.angle) * -1
        sprite = hill_racing.get_scaled_sprites()["torso"]
        rotated_torso_sprite = pygame.transform.rotate(sprite, degrees_angle)
        surface_screen.blit(
            source=rotated_torso_sprite,
            dest=(pos_x - hill_racing.panX, pos_y - hill_racing.panY),
        )
