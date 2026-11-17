import math
from Box2D import (
    b2BodyDef,
    b2CircleShape,
    b2DistanceJointDef,
    b2FixtureDef,
    b2PrismaticJointDef,
    b2RevoluteJointDef,
    b2Vec2,
    b2_dynamicBody,
)
import pygame
from . import hill_racing


class Wheel:
    def __init__(self, x, y, r, chassis_body=None, world=None):
        self.starting_position = pygame.Vector2(x, y)
        self.id = "wheel"
        self.radius = r
        self.body = None
        self.world = world
        self.ground_contacts = 0  # Contatore robusto contro flickering tra edge adiacenti

        self.create_wheel()

        # Mozzo ruota (rim_body): nessuna collisione con terreno/erba (maskBits=0)
        # Densità aumentata (0.3) per mantenere un rapporto di massa sano con il telaio
        body_def = b2BodyDef(
            type=b2_dynamicBody,
            position=(
                self.starting_position.x / hill_racing.SCALE,
                self.starting_position.y / hill_racing.SCALE,
            ),
            angle=0,
        )
        fix_def = b2FixtureDef(
            density=0.3,
            friction=0.0,
            restitution=0.0,
            shape=b2CircleShape(radius=(self.radius * 0.5 / hill_racing.SCALE)),
            categoryBits=0x0000,
            maskBits=0x0000,
        )
        self.rim_body = self.world.CreateBody(body_def)
        self.rim_body.CreateFixture(fix_def)
        self.rim_body.userData = None  # Non deve attivare il ContactListener

        if chassis_body is not None:
            rev_joint_def = b2RevoluteJointDef()
            rev_joint_def.Initialize(
                bodyA=self.body, bodyB=self.rim_body, anchor=self.body.position
            )
            self.joint = self.world.CreateJoint(rev_joint_def)

            pris_joint_def = b2PrismaticJointDef()
            pris_joint_def.Initialize(
                bodyA=self.rim_body,
                bodyB=chassis_body,
                anchor=self.body.position,
                axis=b2Vec2(0, -1),
            )
            pris_joint_def.enableLimit = True
            pris_joint_def.lowerTranslation = -0.35
            pris_joint_def.upperTranslation = 0.35
            self.pris_joint = self.world.CreateJoint(pris_joint_def)

            # Sospensione stabile entro il limite di Nyquist a 60 FPS (f_nyquist = 30 Hz)
            dist_joint_def = b2DistanceJointDef()
            anchor_wheel = b2Vec2(x / hill_racing.SCALE, y / hill_racing.SCALE)
            anchor_car = b2Vec2(x / hill_racing.SCALE, (y - r * 3) / hill_racing.SCALE)
            dist_joint_def.Initialize(
                bodyA=self.rim_body,
                bodyB=chassis_body,
                anchorA=anchor_wheel,
                anchorB=anchor_car,
            )
            dist_joint_def.frequencyHz = 8.5
            dist_joint_def.dampingRatio = 0.75
            self.dist_joint = self.world.CreateJoint(dist_joint_def)

        self.body.angularDamping = 1.8

    @property
    def on_ground(self) -> bool:
        return self.ground_contacts > 0

    def create_wheel(self):
        wheel_body = b2BodyDef(
            type=b2_dynamicBody,
            position=(
                self.starting_position.x / hill_racing.SCALE,
                self.starting_position.y / hill_racing.SCALE,
            ),
            angle=0,
        )
        wheel_fixture = b2FixtureDef(
            density=1.0,
            friction=1.5,
            restitution=0.1,
            shape=b2CircleShape(radius=self.radius / hill_racing.SCALE),
            categoryBits=hill_racing.WHEEL_CATEGORY,
            maskBits=hill_racing.WHEEL_MASK,
        )
        self.body = self.world.CreateBody(wheel_body)
        self.body.CreateFixture(wheel_fixture)
        self.body.userData = self

    def draw_wheel(self, surface_screen):
        pos_x = self.body.position.x * hill_racing.SCALE - hill_racing.panX
        pos_y = self.body.position.y * hill_racing.SCALE - hill_racing.panY
        degrees_angle = math.degrees(-self.body.angle) % 360
        sprite = hill_racing.get_scaled_sprites()["wheel"]
        rotated = pygame.transform.rotate(sprite, degrees_angle)
        rect = rotated.get_rect(center=(pos_x, pos_y))
        surface_screen.blit(rotated, rect)
