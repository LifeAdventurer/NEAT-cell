import numpy as np
import pygame

from constants import (
    CELL_SIZE,
    HEIGHT,
    MARGIN,
    NUTRIENT_ENERGY,
    PI,
    PREDATOR_CELL_DECR_ENERGY,
    PREDATOR_CELL_ENERGY,
    PREDATOR_POP,
    PREDATOR_SENSORS,
    PREDATOR_VIEW_ANGLE,
    PREY_CELL_DECR_ENERGY,
    PREY_CELL_ENERGY,
    PREY_POP,
    PREY_SENSORS,
    PREY_VIEW_ANGLE,
    WIDTH,
)
from nn import NEAT


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class PreyCell(pygame.sprite.Sprite):
    def __init__(self, x, y, generation=0):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE))
        self.image.fill((90, 170, 90))  # green
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.degree = np.random.randint(0, 360)
        self.energy = PREY_CELL_ENERGY
        self.speed = 0
        self.generation = generation

    def network(self, predator_cell):
        self.input_layer = np.zeros((1, PREY_SENSORS + 2))
        self.input_layer[0, PREY_SENSORS] = self.speed
        self.input_layer[0, PREY_SENSORS + 1] = self.energy

        for j in range(PREY_SENSORS):
            angle = (
                (
                    self.degree
                    - PREY_VIEW_ANGLE
                    + (PREY_VIEW_ANGLE / (PREY_SENSORS // 2)) * j
                )
                / 180
            ) * PI

            end_point_x = self.rect.center[0] + 300 * np.cos(angle)
            end_point_y = self.rect.center[1] + 300 * np.sin(angle)

            collide = False
            mn_distance = 1800
            for cell in predator_cell:
                if cell.rect.collidepoint(end_point_x, end_point_y):
                    mn_distance = min(
                        mn_distance,
                        np.linalg.norm(
                            np.array(cell.rect.center)
                            - np.array(self.rect.center)
                        ),
                    )
                    collide = True

            if collide:
                self.input_layer[0, j] = mn_distance

        return NEAT.activate(NEAT(PREY_SENSORS + 2, 2), self.input_layer)

    def update(self, prey_cell, predator_cell):
        self.output_layer = self.network(predator_cell)
        self.speed += self.output_layer[1]
        self.degree += self.output_layer[0]
        angle_rad = (self.degree / 180) * PI
        self.rect.x += self.speed * np.cos(angle_rad)
        self.rect.y += self.speed * np.sin(angle_rad)

        if self.rect.x + MARGIN >= WIDTH or self.rect.x <= MARGIN:
            self.rect.x -= self.speed * np.cos(angle_rad)
            self.degree = 180 - self.degree
        if self.rect.y + MARGIN >= HEIGHT or self.rect.y <= MARGIN:
            self.rect.y -= self.speed * np.sin(angle_rad)
            self.degree = -self.degree

        self.energy -= PREY_CELL_DECR_ENERGY + (self.speed**2) / 400
        if self.energy <= 0:
            self.kill()

        elif self.energy >= PREY_CELL_ENERGY * 2:
            self.energy = PREY_CELL_ENERGY
            cell = PreyCell(self.rect.x, self.rect.y, self.generation + 1)
            prey_cell.add(cell)

    def eat(self):
        self.energy += NUTRIENT_ENERGY


class PredatorCell(pygame.sprite.Sprite):
    def __init__(self, x, y, generation=0):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE))
        self.image.fill((170, 90, 90))  # red
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.degree = np.random.randint(0, 360)
        self.energy = PREDATOR_CELL_ENERGY
        self.speed = 0
        self.generation = generation

    def network(self, prey_cell):
        self.input_layer = np.zeros((1, PREDATOR_SENSORS + 2))
        self.input_layer[0, PREDATOR_SENSORS] = self.speed
        self.input_layer[0, PREDATOR_SENSORS + 1] = self.energy

        for j in range(PREDATOR_SENSORS):
            angle = (
                (
                    self.degree
                    - PREDATOR_VIEW_ANGLE
                    + (PREDATOR_VIEW_ANGLE / (PREDATOR_SENSORS // 2)) * j
                )
                / 180
            ) * PI

            end_point_x = self.rect.center[0] + 300 * np.cos(angle)
            end_point_y = self.rect.center[1] + 300 * np.sin(angle)

            collide = False
            mn_distance = 1800
            for cell in prey_cell:
                if cell.rect.collidepoint(end_point_x, end_point_y):
                    mn_distance = min(
                        mn_distance,
                        np.linalg.norm(
                            np.array(cell.rect.center)
                            - np.array(self.rect.center)
                        ),
                    )
                    collide = True

            if collide:
                self.input_layer[0, j] = mn_distance

        return NEAT.activate(
            NEAT(PREDATOR_SENSORS + 2, 2), self.input_layer
        )

    def update(self, prey_cell, predator_cell):
        self.output_layer = self.network(prey_cell)
        self.speed += self.output_layer[1]
        self.degree += self.output_layer[0]
        angle_rad = (self.degree / 180) * PI
        self.rect.x += self.speed * np.cos(angle_rad)
        self.rect.y += self.speed * np.sin(angle_rad)

        if self.rect.x + MARGIN >= WIDTH or self.rect.x <= MARGIN:
            self.rect.x -= self.speed * np.cos(angle_rad)
            self.degree = 180 - self.degree
        if self.rect.y + MARGIN >= HEIGHT or self.rect.y <= MARGIN:
            self.rect.y -= self.speed * np.sin(angle_rad)
            self.degree = -self.degree

        self.energy -= PREDATOR_CELL_DECR_ENERGY + (self.speed**2) / 400
        if self.energy <= 0:
            self.kill()

        elif self.energy >= PREDATOR_CELL_ENERGY * 2:
            self.energy = PREDATOR_CELL_ENERGY
            cell = PredatorCell(self.rect.x, self.rect.y, self.generation + 1)
            predator_cell.add(cell)

    def eat(self):
        self.energy += PREY_CELL_ENERGY
        self.degree = (self.degree + 180) % 360


def add_cell(prey_cell, predator_cell):
    for i in range(PREY_POP):
        cell = PreyCell(
            np.random.randint(MARGIN, WIDTH - MARGIN),
            np.random.randint(MARGIN, HEIGHT - MARGIN),
        )
        prey_cell.add(cell)

    for i in range(PREDATOR_POP):
        cell = PredatorCell(
            np.random.randint(MARGIN, WIDTH - MARGIN),
            np.random.randint(MARGIN, HEIGHT - MARGIN),
        )
        predator_cell.add(cell)
