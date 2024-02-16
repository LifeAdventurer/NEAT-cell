import imageio
import matplotlib.pyplot as plt
import numpy as np
import pygame

from cell import Cell_A, Cell_B, add_cell
from constants import HEIGHT, MARGIN, PI, WIDTH
from nutrient import Nutrient
from sensor import Sensor

# def RELU(z):
#     return (z + np.abs(-z)) / 2

save_video = True

nutrient = pygame.sprite.Group()
cell_A = pygame.sprite.Group()
cell_B = pygame.sprite.Group()

add_cell(cell_A, cell_B)

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
    clock.tick(100)
    time += 0.01

    for i in range(6):
        nutrient.add(
            Nutrient(
                np.random.randint(MARGIN, WIDTH - MARGIN),
                np.random.randint(5, HEIGHT - MARGIN),
            )
        )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # draw background
    screen.fill((198, 198, 200))

    nutrient.draw(screen)
    cell_A.draw(screen)
    cell_B.draw(screen)
    cell_A.update(cell_A, cell_B)
    cell_B.update(cell_A, cell_B)

    hits = pygame.sprite.groupcollide(cell_B, cell_A, False, True)
    for hit in hits:
        hit.eat()

    hits = pygame.sprite.groupcollide(cell_A, nutrient, False, True)
    for hit in hits:
        hit.eat()

    list1.append(time)
    list2.append(len(cell_A))
    list3.append(len(cell_B))

    print(
        "Time:",
        round(time, 2),
        "s   cell A population:",
        len(cell_A),
        "   cell B population:",
        len(cell_B),
    )

    if len(cell_B) == 0:
        running = False
    # update display
    pygame.display.update()
    if save_video:
        frames.append(pygame.surfarray.array3d(screen).copy())

pygame.quit()

# Convert frames to video
if save_video:
    imageio.mimsave('video.mp4', frames, fps=100)

fig, axs = plt.subplots(2, 1, figsize=(8, 6))

axs[0].stackplot(list1, list2, list3, colors=["m", "c"])
axs[0].set_xlabel("time")
axs[0].set_ylabel("population")
axs[0].legend(["cell A population", "cell B population"])

axs[1].plot(list1, list2, color="m", label="cell A population", linewidth=2)
axs[1].plot(list1, list3, color="c", label="cell B population", linewidth=2)
axs[1].set_xlabel("time")
axs[1].set_ylabel("population")
axs[1].legend()

plt.tight_layout()

plt.savefig("result_plot.png")

plt.show()
