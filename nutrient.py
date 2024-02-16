import pygame

from constants import NUTRIENT_SIZE


class Nutrient(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((NUTRIENT_SIZE, NUTRIENT_SIZE))
        self.image.fill((170, 170, 90))  # yellow
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
