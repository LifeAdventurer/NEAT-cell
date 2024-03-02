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
time_tick = []
prey_cell_count = []
predator_cell_count = []
prey_cell_generation = []
predator_cell_generation = []
prey_cell_time_alive = []
predator_cell_time_alive = []

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

    time_tick.append(time / CLOCK_TICK)
    prey_cell_count.append(len(prey_cell))
    predator_cell_count.append(len(predator_cell))

    mx_prey_cell_gen = max(cell.generation for cell in prey_cell)
    mx_predator_cell_gen = max(cell.generation for cell in predator_cell)
    mx_prey_cell_time_alive = max(f"{cell.time_alive:.2f} " for cell in prey_cell)
    mx_predator_cell_time_alive = max(f"{cell.time_alive:.2f}" for cell in predator_cell)
    prey_cell_generation.append(mx_prey_cell_gen)
    predator_cell_generation.append(mx_predator_cell_gen)
    prey_cell_time_alive.append(mx_prey_cell_time_alive)
    predator_cell_time_alive.append(mx_predator_cell_time_alive)

    print(
        "Time:",
        f"{time / CLOCK_TICK:.2f}",
        "s  PreyPop:",
        len(prey_cell),
        " PredatorPop:",
        len(predator_cell),
        " MaxPreyGen:",
        mx_prey_cell_gen,
        " MaxPredatorGen:",
        mx_predator_cell_gen,
        " MaxPreyTimeAlive:",
        mx_prey_cell_time_alive,
        " MaxPredatorTimeAlive: ",
        mx_predator_cell_time_alive,
    )

    if len(predator_cell) == 0:
        running = False
    # update display
    pygame.display.update()
    if save_video and time % (CLOCK_TICK / FPS) == 0:
        frames.append(pygame.surfarray.array3d(screen).copy())


pygame.quit()


fig, axs = plt.subplots(3, 1, figsize=(8, 6))

# Population
axs[0].plot(time_tick, prey_cell_count, color="m", label="Prey Cell Population", linewidth=2)
axs[0].plot(
    time_tick, predator_cell_count, color="c", label="Predator Cell Population", linewidth=2
)
axs[0].set_xlabel("time")
axs[0].set_ylabel("population")
axs[0].legend()

# Generation
axs[1].plot(time_tick, prey_cell_generation, color="m", label="Prey Cell Generation", linewidth=2)
axs[1].plot(
    time_tick, predator_cell_generation, color="c", label="Predator Cell Generation", linewidth=2
)
axs[1].set_xlabel("time")
axs[1].set_ylabel("generation")
axs[1].legend()

# Time Alive
axs[2].plot(time_tick, prey_cell_time_alive, color="m", label="Prey Cell Time Alive", linewidth=2)
axs[2].plot(time_tick, predator_cell_time_alive, color="c", label="Predator Cell Time Alive", linewidth=2)
axs[2].set_xlabel("time")
axs[2].set_ylabel("time alive(s)")
axs[2].legend()
axs[2].set_yticks(axs[2].get_yticks()[::len(axs[2].get_yticks()) // 4])

plt.tight_layout()

plt.savefig("result_plot.png")

plt.show()

# Convert frames to video
if save_video:
    imageio.mimsave('video.mp4', frames, fps=FPS)
