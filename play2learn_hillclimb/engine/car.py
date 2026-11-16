"""
Modulo Veicolo (Chassis e Motore) per Hill Climb Racing.
Include `reset_pose()` per il teletrasporto istantaneo (< 0.02 ms) dei corpi dinamici
Box2D senza distruggere né riallocare fixture o giunti.

Allineato alla struttura fisica dell'originale alexzh3/hillclimbracing:
- 3 fixture del telaio (corpo principale + parabrezza + paraurti posteriore)
- angularDamping=0.1 sul chassis (smorza oscillazioni rotazionali)
- revolute joint torso-chassis SENZA limiti angolari (come l'originale)
- distance joint diagonale pettorina-cofano (cintura elastica)
- motor_on(reverse) NON applica coppia al telaio (come l'originale)
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

        # =====================================================================
        # FIXTURE 1: Sagoma principale del telaio (5 vertici)
        # =====================================================================
        vectors = [
            b2Vec2(-self.chassis_width / 2, 0 - self.chassis_height / 2),
            b2Vec2(self.chassis_width / 4 + 5, 0 - self.chassis_height / 2),
            b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5),
            b2Vec2(self.chassis_width / 2, self.chassis_height / 2),
            b2Vec2(self.chassis_width / 2, self.chassis_height / 2),  # Vertice duplicato (come l'originale)
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

        # =====================================================================
        # FIXTURE 2: Parte anteriore (parabrezza / cofano rialzato)
        # =====================================================================
        vectors2 = [
            b2Vec2(self.chassis_width / 4, 0 - self.chassis_height / 2),
            b2Vec2(self.chassis_width / 4 - 15, 0 - self.chassis_height / 2 - 20),
            b2Vec2(self.chassis_width / 4 - 5, 0 - self.chassis_height / 2 - 20),
            b2Vec2(self.chassis_width / 4 + 10, 0 - self.chassis_height / 2),
        ]
        for v in vectors2:
            v.x /= hill_racing.SCALE
            v.y /= hill_racing.SCALE
        car_fixture2 = b2FixtureDef(
            categoryBits=hill_racing.CHASSIS_CATEGORY,
            maskBits=hill_racing.CHASSIS_MASK,
            density=self.car_density,
            friction=0.5,
            restitution=self.car_restitution,
            shape=b2PolygonShape(vertices=vectors2, vertexCount=len(vectors2)),
        )
        self.chassis_body.CreateFixture(car_fixture2)
        self.shapes.append(vectors2)

        # =====================================================================
        # FIXTURE 3: Parte posteriore (paraurti / alettone)
        # =====================================================================
        vectors3 = [
            b2Vec2(self.chassis_width / 2, 0 - self.chassis_height / 2 + 5),
            b2Vec2(self.chassis_width / 2 + 5, 0 - self.chassis_height / 2 + 8),
            b2Vec2(self.chassis_width / 2 + 5, self.chassis_height / 2 - 5),
            b2Vec2(self.chassis_width / 2, self.chassis_height / 2),
        ]
        for v in vectors3:
            v.x /= hill_racing.SCALE
            v.y /= hill_racing.SCALE
        car_fixture3 = b2FixtureDef(
            categoryBits=hill_racing.CHASSIS_CATEGORY,
            maskBits=hill_racing.CHASSIS_MASK,
            density=self.car_density,
            friction=0.1,
            restitution=0.1,
            shape=b2PolygonShape(vertices=vectors3, vertexCount=len(vectors3)),
        )
        self.chassis_body.CreateFixture(car_fixture3)
        self.shapes.append(vectors3)

        # =====================================================================
        # Ruote: posteriore (sinistra) e anteriore (destra)
        # =====================================================================
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

        # =====================================================================
        # Pilota (Person: torso + testa)
        # =====================================================================
        self.person = person.Person(
            x=x,
            y=y,
            person_width=hill_racing.PERSON_WIDTH,
            person_height=hill_racing.PERSON_HEIGHT,
            world=self.world,
        )
        self.person.head.agent = self.agent
        self.person.torso.agent = self.agent

        # =====================================================================
        # Giunto 1: Revolute joint torso-chassis con finecorsa di sicurezza
        # Evita che il busto si ribalti all'indietro oltre i 45° passando il punto
        # di equilibrio instabile (over-center) della molla pettorina-cofano.
        # =====================================================================
        rev_joint_def = b2RevoluteJointDef()
        joint_pos = b2Vec2(x / hill_racing.SCALE, y / hill_racing.SCALE)
        rev_joint_def.Initialize(bodyA=self.person.torso.body, bodyB=self.chassis_body, anchor=joint_pos)
        rev_joint_def.enableLimit = True
        rev_joint_def.lowerAngle = -math.pi / 4  # -45° max inclinazione posteriore
        rev_joint_def.upperAngle = math.pi / 4   # +45° max inclinazione anteriore
        self.rev_joint_torso_chassis = self.world.CreateJoint(rev_joint_def)

        # =====================================================================
        # Giunto 2: Distance joint diagonale (cintura elastica pettorina-cofano)
        # frequencyHz=5, dampingRatio=0.5 (smorzamento aumentato per stabilità)
        # =====================================================================
        dist_joint_def = b2DistanceJointDef()
        anchor_person = b2Vec2(x / hill_racing.SCALE, (y - self.person.height * 2.0 / 3.0) / hill_racing.SCALE)
        anchor_car = b2Vec2(
            (x + self.chassis_width / 2.0) / hill_racing.SCALE,
            (y - self.chassis_height / 2.0) / hill_racing.SCALE,
        )
        dist_joint_def.Initialize(
            bodyA=self.person.torso.body,
            bodyB=self.chassis_body,
            anchorA=anchor_person,
            anchorB=anchor_car,
        )
        dist_joint_def.frequencyHz = 5.0
        dist_joint_def.dampingRatio = 0.5
        dist_joint_def.length *= 1.1
        self.dist_joint_torso_chassis = self.world.CreateJoint(dist_joint_def)

        # =====================================================================
        # Proprietà finali del chassis (CRITICO: angularDamping dall'originale)
        # Senza questo, l'energia rotazionale accumulata sui dossi non si dissipa
        # e il pilota inizia a girare su se stesso fino al contatto testa-suolo.
        # =====================================================================
        self.chassis_body.angularDamping = 0.1
        self.chassis_body.userData = self

    def reset_pose(self, x_px: float, y_px: float) -> None:
        """
        Teletrasporta istantaneamente telaio, ruote, sospensioni e pilota alle coordinate
        di partenza (x_px, y_px) azzerando velocità lineari/angolari e contatti.
        Tempo di esecuzione: ~0.015 ms (nessuna riallocazione C++ Box2D).
        """
        scale = float(hill_racing.SCALE)
        self.starting_position.x = x_px
        self.starting_position.y = y_px

        cx = x_px / scale
        cy = y_px / scale
        self.pos_x = cx
        self.pos_y = cy
        self.max_distance = cx
        self.prev_max_distance = cx
        self.prev_pos_y = cy
        self.dead = False
        self.motor_state = 0

        # 1. Telaio
        self.chassis_body.transform = ((cx, cy), 0.0)
        self.chassis_body.linearVelocity = (0.0, 0.0)
        self.chassis_body.angularVelocity = 0.0
        self.chassis_body.awake = True

        # 2. Ruota posteriore e anteriore
        w0_x = (x_px - self.chassis_width / 2.0 + self.wheel_size * 1.2) / scale
        w1_x = (x_px + self.chassis_width / 2.0 - self.wheel_size * 1.2) / scale
        w_y = (y_px + self.chassis_height / 2.0 + self.wheel_size / 4.0) / scale

        for w, wx in ((self.wheels[0], w0_x), (self.wheels[1], w1_x)):
            w.body.transform = ((wx, w_y), 0.0)
            w.body.linearVelocity = (0.0, 0.0)
            w.body.angularVelocity = 0.0
            w.body.awake = True

            w.rim_body.transform = ((wx, w_y), 0.0)
            w.rim_body.linearVelocity = (0.0, 0.0)
            w.rim_body.angularVelocity = 0.0
            w.rim_body.awake = True

            w.joint.motorEnabled = False
            w.joint.motorSpeed = 0.0
            w._ground_contacts = 0
            w.on_ground = False

        # 3. Pilota (Torso e Testa)
        p_w = float(hill_racing.PERSON_WIDTH)
        p_h = float(hill_racing.PERSON_HEIGHT)

        torso_pos = (x_px / scale, (y_px - p_h / 2.0) / scale)
        head_pos = (x_px / scale, (y_px - (p_h + p_w)) / scale)

        tb = self.person.torso.body
        tb.transform = (torso_pos, 0.0)
        tb.linearVelocity = (0.0, 0.0)
        tb.angularVelocity = 0.0
        tb.awake = True

        hb = self.person.head.body
        hb.transform = (head_pos, 0.0)
        hb.linearVelocity = (0.0, 0.0)
        hb.angularVelocity = 0.0
        hb.awake = True

    def draw_person_car(self, surface_screen: pygame.Surface) -> None:
        self.person.draw_person(surface_screen)
        for wheel in self.wheels:
            wheel.draw_wheel(surface_screen)

        pos_x = self.chassis_body.position.x * hill_racing.SCALE
        pos_y = self.chassis_body.position.y * hill_racing.SCALE
        degrees_angle = math.degrees(-self.chassis_body.angle) % 360

        sprite = getattr(hill_racing, "car_sprite", None)
        if sprite is not None:
            scaled = pygame.transform.scale(sprite, (self.chassis_width + 23, self.chassis_height * 2 + 10))
            rotated = pygame.transform.rotate(scaled, degrees_angle)
            rad = -self.chassis_body.angle
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            local_offset_x = 4.5
            local_offset_y = -15.0
            cx = pos_x - hill_racing.panX + (local_offset_x * cos_a - local_offset_y * sin_a)
            cy = pos_y - hill_racing.panY + (local_offset_x * sin_a + local_offset_y * cos_a)
            rect = rotated.get_rect(center=(int(cx), int(cy)))
            surface_screen.blit(rotated, rect.topleft)
        else:
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
        w0 = self.wheels[0]
        w1 = self.wheels[1]
        w0.joint.motorEnabled = True
        w1.joint.motorEnabled = True
        old_state = self.motor_state
        if forward:
            self.motor_state = 1
            target_speed = -self.motor_speed * math.pi
            w0.joint.motorSpeed = target_speed
            w1.joint.motorSpeed = target_speed
            self.chassis_body.ApplyTorque(-self.rotation_torque, True)
        else:
            # NOTA: l'originale NON applica coppia in retromarcia.
            # Applicarla causa rotazione incontrollata del telaio e spinning del pilota.
            self.motor_state = -1
            target_speed = self.motor_speed * math.pi
            w0.joint.motorSpeed = target_speed
            w1.joint.motorSpeed = target_speed

        # Compensazione rotazionale alla transizione avanti→indietro
        if old_state + self.motor_state == 0 and old_state == 1:
            self.chassis_body.ApplyTorque(self.motor_state * -1, True)

        w0.joint.maxMotorTorque = 500
        w1.joint.maxMotorTorque = 250

    def motor_off(self) -> None:
        if self.motor_state == 1:
            self.chassis_body.ApplyTorque(self.motor_state * self.rotation_torque, True)
        self.motor_state = 0
        self.wheels[0].joint.motorEnabled = False
        self.wheels[1].joint.motorEnabled = False

    def set_motor_wheel_speed(self, motor_wheel_speed: float) -> None:
        speed = float(motor_wheel_speed)
        w0 = self.wheels[0]
        w1 = self.wheels[1]
        w0.joint.motorEnabled = True
        w1.joint.motorEnabled = True
        old_state = self.motor_state
        if speed < 0:
            self.motor_state = 1
            w0.joint.motorSpeed = speed * math.pi
            w1.joint.motorSpeed = speed * math.pi
            self.chassis_body.ApplyTorque(-self.rotation_torque, True)
        elif speed >= 0:
            # Come l'originale: nessuna coppia applicata in retromarcia
            self.motor_state = -1
            w0.joint.motorSpeed = speed * math.pi
            w1.joint.motorSpeed = speed * math.pi

        if old_state + self.motor_state == 0 and old_state == 1:
            self.chassis_body.ApplyTorque(self.motor_state * -1, True)

        w0.joint.maxMotorTorque = 500
        w1.joint.maxMotorTorque = 250
