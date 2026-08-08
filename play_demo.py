"""
play_demo.py
------------
Run this LOCALLY (on your own machine, not in a headless container) to see
the bee environment as a live pygame window -- a "videogame" view with a
textured field, a bee sprite, and flower sprites.

Right now the bee is driven by a placeholder controller (either a simple
gradient-following heuristic, or pure random) just to prove out the
environment + rendering. Day 2 will replace this controller with the
actual neural network, whose weights are evolved by the ES.

Controls:
  SPACE  - pause / resume
  R      - reset episode (new random flower + bee placement)
  1      - switch to RANDOM controller
  2      - switch to GRADIENT-FOLLOWING heuristic controller
  ESC/Q  - quit

Run:
  python play_demo.py
"""

import sys
import numpy as np
import pygame

from bee_env import Environment
from render import (
    generate_field_texture, make_bee_sprite, make_flower_sprite,
    draw_bee, draw_flower,
)

# ---- rendering config ----
WORLD_W, WORLD_H = 100.0, 100.0
SCALE = 7  # pixels per world unit
SCREEN_W, SCREEN_H = int(WORLD_W * SCALE), int(WORLD_H * SCALE)
FPS = 30

COLOR_TEXT = (255, 255, 255)
TEXT_BG = (20, 20, 20, 160)


def random_controller(obs, rng):
    return rng.uniform(-1, 1), rng.uniform(-1, 1)


def gradient_controller(obs):
    """Simple heuristic: turn toward the flower, go forward proportional
    to signal strength. Not learned -- just here to sanity check the
    sensory model direction convention."""
    rel_angle, strength = obs
    turn_rate = np.clip(rel_angle * 3.0, -1, 1)
    forward_speed = 0.8
    return forward_speed, turn_rate


def build_flower_sprites(n_flowers, size_px):
    actives, wilteds = [], []
    for i in range(n_flowers):
        actives.append(make_flower_sprite(size=size_px, variant=i))
        wilteds.append(make_flower_sprite(size=size_px, variant=i, wilted=True))
    return actives, wilteds


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Bee ER Environment - Day 1 Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    field = generate_field_texture(SCREEN_W, SCREEN_H, seed=7)
    bee_sprite = make_bee_sprite(size=int(SCALE * 3.2))

    rng = np.random.default_rng()
    n_flowers = 3
    env = Environment(width=WORLD_W, height=WORLD_H, n_flowers=n_flowers, max_steps=600)
    flower_sprites_active, flower_sprites_wilted = build_flower_sprites(n_flowers, size_px=int(SCALE * 2.2))

    paused = False
    controller_mode = "gradient"  # "random" or "gradient"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    env = Environment(width=WORLD_W, height=WORLD_H, n_flowers=n_flowers, max_steps=600)
                    flower_sprites_active, flower_sprites_wilted = build_flower_sprites(n_flowers, size_px=int(SCALE * 2.2))
                elif event.key == pygame.K_1:
                    controller_mode = "random"
                elif event.key == pygame.K_2:
                    controller_mode = "gradient"

        if not paused and not env.done:
            obs = env.get_observation()
            if controller_mode == "random":
                fwd, turn = random_controller(obs, rng)
            else:
                fwd, turn = gradient_controller(obs)
            env.step(fwd, turn)

        screen.blit(field, (0, 0))
        for i, f in enumerate(env.flowers):
            draw_flower(screen, f, flower_sprites_active[i], flower_sprites_wilted[i], SCALE, SCREEN_H)
        draw_bee(screen, env.bee, bee_sprite, SCALE, SCREEN_H)

        status = f"controller={controller_mode}  step={env.step_count}  done={env.done}  [SPACE pause | R reset | 1/2 controller]"
        text_surf = font.render(status, True, COLOR_TEXT)
        bg_surf = pygame.Surface((text_surf.get_width() + 12, text_surf.get_height() + 8), pygame.SRCALPHA)
        bg_surf.fill(TEXT_BG)
        screen.blit(bg_surf, (4, 4))
        screen.blit(text_surf, (10, 8))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()