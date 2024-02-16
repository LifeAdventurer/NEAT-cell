import pygame

from constants import SENSOR_SIZE


class Sensor(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((SENSOR_SIZE, SENSOR_SIZE))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

    def Kill(self):
        self.kill()
