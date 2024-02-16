import numpy as np
import pygame

from constants import (
    CELL_A_DECR_ENERGY,
    CELL_A_ENERGY,
    CELL_B_DECR_ENERGY,
    CELL_B_ENERGY,
    CELL_SIZE,
    HEIGHT,
    MARGIN,
    MUTATE_PROB,
    NUTRIENT_ENERGY,
    PI,
    POP_A,
    POP_B,
    WIDTH,
)
from sensor import Sensor


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mutate(weight):
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            if np.random.rand() < MUTATE_PROB:
                weight[i, j] += np.random.uniform(-5.0, 5.0)
    return weight


class Cell_A(pygame.sprite.Sprite):
    def __init__(self, x, y, weight1, weight2):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE))
        self.image.fill((90, 170, 90))  # green
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.weight1 = weight1
        self.weight2 = weight2
        self.degree = np.random.randint(0, 360)
        self.energy = CELL_A_ENERGY

    def network(self, cell_B):
        self.input_layer = np.zeros((1, 25))

        for j in range(25):
            angle = ((self.degree - 90 + 7.5 * j) / 180) * PI
            for i in range(5):
                sensor_front = Sensor(
                    self.rect.center[0] + (15 + 20 * i) * np.cos(angle),
                    self.rect.center[1] + (15 + 20 * i) * np.sin(angle),
                )

                if any(
                    pygame.Rect(sensor_front.rect).colliderect(cell.rect)
                    for cell in cell_B
                ):
                    self.input_layer[0, j] = 1 + i
                    sensor_front.Kill()
                    break
                sensor_front.Kill()

        return np.tanh(
            sigmoid(self.input_layer.dot(self.weight1)).dot(self.weight2)
        )

    def update(self, cell_A, cell_B):

        # show sensor
        # for j in range(25):
        #     for i in range(5):
        #         self.obstacle_hitbox_front = pygame.Rect(0, 0, 10 , 10)
        #         self.obstacle_hitbox_front.center = (self.rect.center[0] + (15 + 20 * i) * np.cos(((self.degree - 90 + 7.5 * j) / 180) * PI),
        #                                              self.rect.center[1] + (15 + 20 * i) * np.sin(((self.degree - 90 + 7.5 * j) / 180) * PI))
        #         pygame.draw.rect(screen, (255, 150, 150), self.obstacle_hitbox_front)

        self.output_layer = self.network(cell_B)
        self.speed = (self.output_layer[0, 1]) * 4
        self.degree += self.output_layer[0, 0] * 5
        angle_rad = (self.degree / 180) * PI
        self.rect.x += self.speed * np.cos(angle_rad)
        self.rect.y += self.speed * np.sin(angle_rad)

        if self.rect.x > WIDTH or self.rect.x < 0:
            self.rect.x -= self.speed * np.cos(angle_rad)
            self.degree = 180 - self.degree
        if self.rect.y > HEIGHT or self.rect.y < 0:
            self.rect.y -= self.speed * np.sin(angle_rad)
            self.degree = -self.degree

        self.energy -= CELL_A_DECR_ENERGY + np.abs(self.speed) / 500
        if self.energy <= 0:
            self.kill()

        elif self.energy >= CELL_A_ENERGY * 2:
            self.energy = CELL_A_ENERGY
            weight1 = mutate(self.weight1)
            weight2 = mutate(self.weight2)
            cell = Cell_A(self.rect.x, self.rect.y, weight1, weight2)
            cell_A.add(cell)

    def eat(self):
        self.energy += NUTRIENT_ENERGY


class Cell_B(pygame.sprite.Sprite):
    def __init__(self, x, y, weight1, weight2):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE))
        self.image.fill((170, 90, 90))  # red
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.weight1 = weight1
        self.weight2 = weight2
        self.degree = np.random.randint(0, 360)
        self.energy = CELL_B_ENERGY

    def network(self, cell_A):
        self.input_layer = np.zeros((1, 13))

        for j in range(13):
            angle = ((self.degree - 30 + 5 * j) / 180) * PI
            for i in range(5):
                sensor_front = Sensor(
                    self.rect.center[0] + (15 + 20 * i) * np.cos(angle),
                    self.rect.center[1] + (15 + 20 * i) * np.sin(angle),
                )

                if any(
                    pygame.Rect(sensor_front.rect).colliderect(cell.rect)
                    for cell in cell_A
                ):
                    self.input_layer[0, j] = 1 + i
                    sensor_front.Kill()
                    break
                sensor_front.Kill()

        return np.tanh(
            sigmoid(self.input_layer.dot(self.weight1)).dot(self.weight2)
        )

    def update(self, cell_A, cell_B):

        # for j in range(13):
        #     for i in range(5):
        #         self.obstacle_hitbox_front = pygame.Rect(0, 0, 10 , 10)
        #         self.obstacle_hitbox_front.center = (self.rect.center[0] + (15 + 20 * i) * np.cos(((self.degree - 30 + 5 * j) / 180) * PI),
        #                                              self.rect.center[1] + (15 + 20 * i) * np.sin(((self.degree - 30 + 5 * j) / 180) * PI))
        #         pygame.draw.rect(screen, (255, 150, 150), self.obstacle_hitbox_front)

        self.output_layer = self.network(cell_A)
        self.speed = 3.1 + (self.output_layer[0, 1]) * 2
        self.degree += self.output_layer[0, 0] * 5
        angle_rad = (self.degree / 180) * PI
        self.rect.x += self.speed * np.cos(angle_rad)
        self.rect.y += self.speed * np.sin(angle_rad)

        if self.rect.x > WIDTH or self.rect.x < 0:
            self.rect.x -= self.speed * np.cos(angle_rad)
            self.degree = 180 - self.degree
        if self.rect.y > HEIGHT or self.rect.y < 0:
            self.rect.y -= self.speed * np.sin(angle_rad)
            self.degree = -self.degree

        self.energy -= CELL_B_DECR_ENERGY + np.abs(self.speed) / 300
        if self.energy <= 0:
            self.kill()

        elif self.energy >= CELL_B_ENERGY * 2:
            self.energy = CELL_A_ENERGY
            weight1 = mutate(self.weight1)
            weight2 = mutate(self.weight2)
            cell = Cell_B(self.rect.x, self.rect.y, weight1, weight2)
            cell_B.add(cell)

    def eat(self):
        self.energy += CELL_A_ENERGY
        self.degree = (self.degree + 180) % 360


def add_cell(cell_A, cell_B):
    for i in range(POP_A):
        weight1 = np.random.uniform(-5, 5, size=(25, 8))
        weight2 = np.random.uniform(-5, 5, size=(8, 2))
        cell = Cell_A(
            np.random.randint(MARGIN, WIDTH - MARGIN),
            np.random.randint(MARGIN, HEIGHT - MARGIN),
            weight1,
            weight2,
        )
        cell_A.add(cell)

    for i in range(POP_B):
        weight1 = np.random.uniform(-5, 5, size=(13, 8))
        weight2 = np.random.uniform(-5, 5, size=(8, 2))
        cell = Cell_B(
            np.random.randint(MARGIN, WIDTH - MARGIN),
            np.random.randint(MARGIN, HEIGHT - MARGIN),
            weight1,
            weight2,
        )
        cell_B.add(cell)
