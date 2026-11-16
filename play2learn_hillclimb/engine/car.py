"""
Modulo Veicolo (Chassis e Motore) per Hill Climb Racing.
"""

from __future__ import annotations

import math
from Box2D import (
    b2BodyDef,
    b2DistanceJointDef,
    b2FixtureDef,
    b2PolygonShape,
    b2RevoluteJointDef,
    b2Vec2,
    b2_dynamicBody,
)
import pygame

from . import hill_racing, person, wheels


class Car:
    def __init__(self, x: float, y: float, world=None, agent=None):
        self.world = world
        self.id = "car"
        self.agent = agent
        self.wheels: list[wheels.Wheel] = []
        self.starting_position = pygame.Vector2(x, y)
        self.chassis_body = None
        self.chassis_width = 125
        self.chassis_height = 40
        self.wheel_size = 17
        self.dead = False
        self.shapes: list[list[b2Vec2]] = []
        self.car_density = 1.0
        self.car_restitution = 0.01
        self.pos_x = x / hill_racing.SCALE
        self.pos_y = y / hill_racing.SCALE
        self.max_distance = x / hill_racing.SCALE
        self.prev_max_distance = x / hill_racing.SCALE
        self.prev_pos_y = y / hill_racing.SCALE
        self.motor_state = 0
        self.rotation_torque = 2.0
        self.motor_speed = 13.0

        # Sagoma principale del telaio
        vectors = [
            b2Vec2(-self.chassis_width / 2, 0 - self.chassis_height / 2),
            b2Vec2(self.chassis_width / 4 + 5, 0 - self.chassis_height / 2),
            b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5),
            b2Vec2(self.chassis_width / 2, self.chassis_height / 2),
            b2Vec2(-self.chassis_width / 2, self.chassis_height / 2),
        ]
        for v in vectors:
            v.x /= hill_racing.SCALE
            v.y /= hill_racing.SCALE
        self.shapes.append(vectors)

        car_body = b2BodyDef(
            type=b2_dynamicBody,
            position=(x / hill_racing.SCALE, y / hill_racing.SCALE),
            angle=0,
        )
        car_fixture = b2FixtureDef(
            categoryBits=hill_racing.CHASSIS_CATEGORY,
            maskBits=hill_racing.CHASSIS_MASK,
            density=self.car_density,
            friction=0.5,
            restitution=self.car_restitution,
            shape=b2PolygonShape(vertices=vectors, vertexCount=len(vectors)),
        )

        self.chassis_body = self.world.CreateBody(car_body)
        self.chassis_body.CreateFixture(car_fixture)
        self.chassis_body.userData = self

        # Ruote: posteriore (sinistra) e anteriore (destra)
        self.wheels.append(
            wheels.Wheel(
                x - self.chassis_width / 2 + self.wheel_size * 1.2,
                y + self.chassis_height / 2 + self.wheel_size / 4,
                self.wheel_size,
                self.chassis_body,
                self.world,
            )
        )
        self.wheels.append(
            wheels.Wheel(
                x + self.chassis_width / 2 - self.wheel_size * 1.2,
                y + self.chassis_height / 2 + self.wheel_size / 4,
                self.wheel_size,
                self.chassis_body,
                self.world,
            )
        )

        # Pilota
        self.person = person.Person(
            x=self.starting_position.x - 10,
            y=self.starting_position.y - self.chassis_height / 2,
            person_width=hill_racing.PERSON_WIDTH,
            person_height=hill_racing.PERSON_HEIGHT,
            world=self.world,
        )
        self.person.head.agent = self.agent
        self.person.torso.agent = self.agent

        # Giunto corpo-pilota
        torso_joint_def = b2DistanceJointDef()
        anchor_torso = [self.person.torso.body.position.x, self.person.torso.body.position.y]
        anchor_chassis = [self.chassis_body.position.x, self.chassis_body.position.y]
        torso_joint_def.Initialize(
            bodyA=self.person.torso.body,
            bodyB=self.chassis_body,
            anchorA=anchor_torso,
            anchorB=anchor_chassis,
        )
        self.dist_joint_torso_chassis = self.world.CreateJoint(torso_joint_def)

        rev_joint_def = b2RevoluteJointDef()
        rev_joint_def.Initialize(
            bodyA=self.person.torso.body,
            bodyB=self.chassis_body,
            anchor=anchor_chassis,
        )
        self.rev_joint_torso_chassis = self.world.CreateJoint(rev_joint_def)

    def draw_person_car(self, surface_screen: pygame.Surface) -> None:
        self.person.draw_person(surface_screen)
        for wheel in self.wheels:
            wheel.draw_wheel(surface_screen)

        pos_x = self.chassis_body.position.x * hill_racing.SCALE
        pos_y = self.chassis_body.position.y * hill_racing.SCALE
        degrees_angle = math.degrees(-self.chassis_body.angle) % 360

        sprite = getattr(hill_racing, "car_sprite", None)
        if sprite is not None:
            scaled = pygame.transform.scale(sprite, (self.chassis_width + 10, self.chassis_height + 25))
            rotated = pygame.transform.rotate(scaled, degrees_angle)
            surface_screen.blit(
                rotated,
                (-self.chassis_width / 2 + pos_x - hill_racing.panX - 5,
                 -self.chassis_height / 2 + pos_y - hill_racing.panY - 15),
            )
        else:
            # Rettangolo telaio vettoriale di fallback
            rect = pygame.Rect(
                int(pos_x - hill_racing.panX - self.chassis_width / 2),
                int(pos_y - hill_racing.panY - self.chassis_height / 2),
                int(self.chassis_width),
                int(self.chassis_height),
            )
            pygame.draw.rect(surface_screen, (200, 30, 30), rect, border_radius=4)

    def update_status(self) -> None:
        self.pos_x = self.chassis_body.position.x
        self.pos_y = self.chassis_body.position.y
        if self.pos_x > self.max_distance:
            self.prev_max_distance = self.max_distance
            self.max_distance = self.pos_x

    def motor_on(self, forward: bool = True) -> None:
        self.wheels[0].joint.motorEnabled = True
        self.wheels[1].joint.motorEnabled = True
        old_state = self.motor_state
        if forward:
            self.motor_state = 1
            self.wheels[0].joint.motorSpeed = -self.motor_speed * math.pi
            self.wheels[1].joint.motorSpeed = -self.motor_speed * math.pi
            self.chassis_body.ApplyTorque(-self.rotation_torque, True)
        else:
            self.motor_state = -1
            self.wheels[0].joint.motorSpeed = self.motor_speed * math.pi
            self.wheels[1].joint.motorSpeed = self.motor_speed * math.pi
            self.chassis_body.ApplyTorque(self.rotation_torque, True)

        if old_state + self.motor_state == 0 and old_state == 1:
            self.chassis_body.ApplyTorque(self.motor_state * -1, True)

        self.wheels[0].joint.maxMotorTorque = 500
        self.wheels[1].joint.maxMotorTorque = 250

    def motor_off(self) -> None:
        if self.motor_state == 1:
            self.chassis_body.ApplyTorque(self.motor_state * self.rotation_torque, True)
        self.motor_state = 0
        self.wheels[0].joint.motorEnabled = False
        self.wheels[1].joint.motorEnabled = False

    def set_motor_wheel_speed(self, motor_wheel_speed: float) -> None:
        speed = float(motor_wheel_speed)
        self.wheels[0].joint.motorEnabled = True
        self.wheels[1].joint.motorEnabled = True
        old_state = self.motor_state
        if speed < 0:
            self.motor_state = 1
            self.wheels[0].joint.motorSpeed = speed * math.pi
            self.wheels[1].joint.motorSpeed = speed * math.pi
            self.chassis_body.ApplyTorque(-self.rotation_torque, True)
        elif speed >= 0:
            self.motor_state = -1
            self.wheels[0].joint.motorSpeed = speed * math.pi
            self.wheels[1].joint.motorSpeed = speed * math.pi

        if old_state + self.motor_state == 0 and old_state == 1:
            self.chassis_body.ApplyTorque(self.motor_state * -1, True)

        self.wheels[0].joint.maxMotorTorque = 500
        self.wheels[1].joint.maxMotorTorque = 250
