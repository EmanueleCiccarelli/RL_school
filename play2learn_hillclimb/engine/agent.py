"""
Modulo Agente (Car, Person, Airtime & Telemetria) per Hill Climb Racing.
"""

from __future__ import annotations

from . import car, hill_racing


class Agent:
    def __init__(self, real_world):
        self.dead = False
        self.score = 0
        self.world = real_world
        self.car: car.Car | None = None
        self.spawning_x = hill_racing.SPAWNING_X
        self.spawning_y = hill_racing.SPAWNING_Y
        self.steps_in_air = 0
        self.airtime_counter = 0
        self.total_airtime = 0

    def add_to_world(self) -> None:
        self.car = car.Car(x=self.spawning_x, y=self.spawning_y, world=self.world, agent=self)

    def draw_agent(self, surface_screen) -> None:
        if not self.dead and self.car is not None:
            self.car.draw_person_car(surface_screen)

    def update(self) -> None:
        if self.car is None:
            return

        hill_racing.panX = self.car.chassis_body.position.x * hill_racing.SCALE - 100

        if self.car.dead:
            self.dead = True
        else:
            self.car.update_status()

        self.score = max(0, int((self.car.max_distance - (self.spawning_x / hill_racing.SCALE))))

        if not self.car.wheels[0].on_ground and not self.car.wheels[1].on_ground:
            self.total_airtime = 0
            self.steps_in_air += 1
            if self.steps_in_air >= 20:
                self.airtime_counter += 1
                self.steps_in_air = 0
        else:
            self.total_airtime = self.airtime_counter
            self.airtime_counter = 0
            self.steps_in_air = 0

    def destroy_agent(self) -> None:
        if self.car is None or self.world is None:
            return
        try:
            self.world.DestroyBody(self.car.chassis_body)
            self.world.DestroyBody(self.car.wheels[0].body)
            self.world.DestroyBody(self.car.wheels[0].rim_body)
            self.world.DestroyBody(self.car.wheels[1].body)
            self.world.DestroyBody(self.car.wheels[1].rim_body)
            self.world.DestroyBody(self.car.person.head.body)
            self.world.DestroyBody(self.car.person.torso.body)
        except Exception:
            pass
        self.car = None
