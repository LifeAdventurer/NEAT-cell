import imageio
import matplotlib.pyplot as plt
import numpy as np
import pygame

from cell import PredatorCell, PreyCell, add_cell
from constants import CLOCK_TICK, FPS, HEIGHT, MARGIN, NUTRIENT_PER_TICK, PI, WIDTH
from nutrient import Nutrient

# def RELU(z):
#     return (z + np.abs(-z)) / 2

save_video = True

nutrient = pygame.sprite.Group()
prey_cell = pygame.sprite.Group()
predator_cell = pygame.sprite.Group()

add_cell(prey_cell, predator_cell)

# for generating video.mp4
frames = []

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Generation Algorithm")
clock = pygame.time.Clock()

running = True
time = 0
list1 = []
list2 = []
list3 = []

while running:
    clock.tick(CLOCK_TICK)
    time += 1

    for i in range(NUTRIENT_PER_TICK):
        nutrient.add(
            Nutrient(
                np.random.randint(MARGIN, WIDTH - MARGIN),
                np.random.randint(MARGIN, HEIGHT - MARGIN),
            )
        )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # draw background
    screen.fill((198, 198, 200))

    nutrient.draw(screen)
    prey_cell.draw(screen)
    predator_cell.draw(screen)
    prey_cell.update(prey_cell, predator_cell)
    predator_cell.update(prey_cell, predator_cell)

    hits = pygame.sprite.groupcollide(predator_cell, prey_cell, False, True)
    for hit in hits:
        hit.eat()

    hits = pygame.sprite.groupcollide(prey_cell, nutrient, False, True)
    for hit in hits:
        hit.eat()

    list1.append(time / CLOCK_TICK)
    list2.append(len(prey_cell))
    list3.append(len(predator_cell))

    print(
        "Time:",
        f"{time / CLOCK_TICK:.2f}",
        "s   Prey Cell population:",
        len(prey_cell),
        "   Predator Cell population:",
        len(predator_cell),
    )

    if len(predator_cell) == 0:
        running = False
    # update display
    pygame.display.update()
    if save_video and time % (CLOCK_TICK / FPS) == 0:
        frames.append(pygame.surfarray.array3d(screen).copy())


pygame.quit()


fig, axs = plt.subplots(2, 1, figsize=(8, 6))

axs[0].stackplot(list1, list2, list3, colors=["m", "c"])
axs[0].set_xlabel("time")
axs[0].set_ylabel("population")
axs[0].legend(["Prey Cell population", "Predator Cell population"])

axs[1].plot(list1, list2, color="m", label="Prey Cell population", linewidth=2)
axs[1].plot(
    list1, list3, color="c", label="Predator Cell population", linewidth=2
)
axs[1].set_xlabel("time")
axs[1].set_ylabel("population")
axs[1].legend()

plt.tight_layout()

plt.savefig("result_plot.png")

plt.show()

# Convert frames to video
if save_video:
    imageio.mimsave('video.mp4', frames, fps=FPS)
