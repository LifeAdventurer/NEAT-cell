import numpy as np
import pygame

from constants import (
    CELL_SIZE,
    HEIGHT,
    MARGIN,
    MUTATE_PROB,
    NUTRIENT_ENERGY,
    PI,
    PREDATOR_CELL_DECR_ENERGY,
    PREDATOR_CELL_ENERGY,
    PREDATOR_POP,
    PREY_CELL_DECR_ENERGY,
    PREY_CELL_ENERGY,
    PREY_POP,
    WIDTH,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def weight_mutate(weight):
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            if np.random.rand() < MUTATE_PROB:
                weight[i, j] += np.random.uniform(-5.0, 5.0)
    return weight

def speed_mutate(speed_coefficient):
    if np.random.rand() < 0.05:
        speed_coefficient += np.random.uniform(-1, 1)
        if speed_coefficient <= 0:
            speed_coefficient = 0
        elif speed_coefficient >= 10:
            speed_coefficient = 10
    return speed_coefficient


class PreyCell(pygame.sprite.Sprite):
    def __init__(self, x, y, weight1, weight2, speed_coefficient):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE))
        self.image.fill((speed_coefficient * 20, 250, 90))  # green
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.weight1 = weight1
        self.weight2 = weight2
        self.speed_coefficient = speed_coefficient
        self.degree = np.random.randint(0, 360)
        self.energy = PREY_CELL_ENERGY

    def network(self, predator_cell):
        self.input_layer = np.zeros((1, 49))

        for j in range(49):
            angle = ((self.degree - 120 + 5 * j) / 180) * PI

            end_point_x = self.rect.center[0] + 300 * np.cos(angle)
            end_point_y = self.rect.center[1] + 300 * np.sin(angle)

            collide = False
            mn_distance = 1800
            for cell in predator_cell:
                if cell.rect.collidepoint(end_point_x, end_point_y):
                    mn_distance = min(mn_distance, np.linalg.norm(
                        np.array(cell.rect.center) - np.array(self.rect.center)
                    ))
                    collide = True
            
            if collide:
                self.input_layer[0, j] = mn_distance

        return np.tanh(
            sigmoid(self.input_layer.dot(self.weight1)).dot(self.weight2)
        )

    def update(self, prey_cell, predator_cell):
        self.output_layer = self.network(predator_cell)
        self.speed = (self.output_layer[0, 1]) * self.speed_coefficient
        self.degree += self.output_layer[0, 0] * 2
        angle_rad = (self.degree / 180) * PI
        self.rect.x += self.speed * np.cos(angle_rad)
        self.rect.y += self.speed * np.sin(angle_rad)

        if self.rect.x + MARGIN >= WIDTH or self.rect.x <= MARGIN:
            self.rect.x -= self.speed * np.cos(angle_rad)
            self.degree = 180 - self.degree
        if self.rect.y + MARGIN >= HEIGHT or self.rect.y <= MARGIN:
            self.rect.y -= self.speed * np.sin(angle_rad)
            self.degree = -self.degree

        self.energy -= PREY_CELL_DECR_ENERGY + (self.speed ** 2) / 400
        if self.energy <= 0:
            self.kill()

        elif self.energy >= PREY_CELL_ENERGY * 2:
            self.energy = PREY_CELL_ENERGY
            weight1 = weight_mutate(self.weight1)
            weight2 = weight_mutate(self.weight2)
            speed_coefficient = speed_mutate(self.speed_coefficient)
            cell = PreyCell(self.rect.x, self.rect.y, weight1, weight2, speed_coefficient)
            prey_cell.add(cell)

    def eat(self):
        self.energy += NUTRIENT_ENERGY


class PredatorCell(pygame.sprite.Sprite):
    def __init__(self, x, y, weight1, weight2, speed_coefficient):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((CELL_SIZE, CELL_SIZE))
        self.image.fill((255, speed_coefficient * 20, 0))  # red
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.weight1 = weight1
        self.weight2 = weight2
        self.speed_coefficient = speed_coefficient
        self.degree = np.random.randint(0, 360)
        self.energy = PREDATOR_CELL_ENERGY

    def network(self, prey_cell):
        self.input_layer = np.zeros((1, 25))

        for j in range(25):
            angle = ((self.degree - 60 + 5 * j) / 180) * PI

            end_point_x = self.rect.center[0] + 300 * np.cos(angle)
            end_point_y = self.rect.center[1] + 300 * np.sin(angle)

            collide = False
            mn_distance = 1800
            for cell in prey_cell:
                if cell.rect.collidepoint(end_point_x, end_point_y):
                    mn_distance = min(mn_distance, np.linalg.norm(
                        np.array(cell.rect.center) - np.array(self.rect.center)
                    ))
                    collide = True
            
            if collide:
                self.input_layer[0, j] = mn_distance

        return np.tanh(
            sigmoid(self.input_layer.dot(self.weight1)).dot(self.weight2)
        )

    def update(self, prey_cell, predator_cell):
        self.output_layer = self.network(prey_cell) * 2
        self.speed = (self.output_layer[0, 1]) * self.speed_coefficient
        self.degree += self.output_layer[0, 0] * 2
        angle_rad = (self.degree / 180) * PI
        self.rect.x += self.speed * np.cos(angle_rad)
        self.rect.y += self.speed * np.sin(angle_rad)

        if self.rect.x + MARGIN >= WIDTH or self.rect.x <= MARGIN:
            self.rect.x -= self.speed * np.cos(angle_rad)
            self.degree = 180 - self.degree
        if self.rect.y + MARGIN >= HEIGHT or self.rect.y <= MARGIN:
            self.rect.y -= self.speed * np.sin(angle_rad)
            self.degree = -self.degree

        self.energy -= PREDATOR_CELL_DECR_ENERGY + (self.speed ** 2) / 400
        if self.energy <= 0:
            self.kill()

        elif self.energy >= PREDATOR_CELL_ENERGY * 2:
            self.energy = PREDATOR_CELL_ENERGY
            weight1 = weight_mutate(self.weight1)
            weight2 = weight_mutate(self.weight2)
            speed_coefficient = speed_mutate(self.speed_coefficient)
            cell = PredatorCell(self.rect.x, self.rect.y, weight1, weight2, speed_coefficient)
            predator_cell.add(cell)

    def eat(self):
        self.energy += PREY_CELL_ENERGY
        self.degree = (self.degree + 180) % 360


def add_cell(prey_cell, predator_cell):
    for i in range(PREY_POP):
        weight1 = np.random.uniform(-5, 5, size=(49, 8))
        weight2 = np.random.uniform(-5, 5, size=(8, 2))
        cell = PreyCell(
            np.random.randint(MARGIN, WIDTH - MARGIN),
            np.random.randint(MARGIN, HEIGHT - MARGIN),
            weight1,
            weight2,
            4
        )
        prey_cell.add(cell)

    for i in range(PREDATOR_POP):
        weight1 = np.random.uniform(-5, 5, size=(25, 8))
        weight2 = np.random.uniform(-5, 5, size=(8, 2))
        cell = PredatorCell(
            np.random.randint(MARGIN, WIDTH - MARGIN),
            np.random.randint(MARGIN, HEIGHT - MARGIN),
            weight1,
            weight2,
            4
        )
        predator_cell.add(cell)
