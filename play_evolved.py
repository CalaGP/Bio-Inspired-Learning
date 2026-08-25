"""
play_evolved.py
---------------

Code to watch the EVOLVED NN controller from run_evolution.py. The brain steering the bee is whatever the ES discovered

Load results for ES from best_genome.npy, use the weights in the FeedForward NN.

Controls:
  SPACE  - pause / resume
  R      - reset episode (new random flower layout + bee start)
  UP/DOWN- increase / decrease simulation speed (steps per frame)
  ESC/Q  - quit

"""

import sys
import numpy as np
import pygame

from bee_env import Environment
from ann import FeedforwardNN
from render import (
    generate_field_texture, make_bee_sprite,
    draw_bee, draw_flower,
)

from play_demo import WORLD_W, WORLD_H, SCALE, SCREEN_W, SCREEN_H, build_flower_sprites

N_HIDDEN = 4   #must match what run_evolution.py is using
N_FLOWERS = 10
MAX_STEPS = 400

COLOR_TEXT = (255, 255, 255)
TEXT_BG = (20, 20, 20, 160)

def main():
    genome = np.load("best_genome.npy")
    controller = FeedforwardNN(n_inputs=2, n_hidden=N_HIDDEN, n_outputs=2)
    controller.set_flat_weights(genome)
    print(f"Loaded genome with {genome.shape[0]} weights.")

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Evolved Bee Controller")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
 
    field = generate_field_texture(SCREEN_W, SCREEN_H, seed=7)
    bee_sprite = make_bee_sprite(size=int(SCALE * 3.2))
 
    def new_episode():
        env = Environment(width=WORLD_W, height=WORLD_H, n_flowers=N_FLOWERS, max_steps=MAX_STEPS)
        flower_sprites_active, flower_sprites_wilted = build_flower_sprites(N_FLOWERS, size_px=int(SCALE * 2.2))
        return env, flower_sprites_active, flower_sprites_wilted
 
    env, flower_sprites_active, flower_sprites_wilted = new_episode()
 
    paused = False
    steps_per_frame = 1
    episode_count = 1
 
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
                    env, flower_sprites_active, flower_sprites_wilted = new_episode()
                    episode_count += 1
                elif event.key == pygame.K_UP:
                    steps_per_frame = min(steps_per_frame + 1, 20)
                elif event.key == pygame.K_DOWN:
                    steps_per_frame = max(steps_per_frame - 1, 1)
 
        if not paused:
            for _ in range(steps_per_frame):
                if env.done:
                    break
                obs = env.get_observation()
                action = controller.forward(obs)
                env.step(action[0], action[1])
            if env.done:
                # brief pause on completion, then auto-restart a new episode
                pygame.time.wait(600)
                env, flower_sprites_active, flower_sprites_wilted = new_episode()
                episode_count += 1
 
        screen.blit(field, (0, 0))
        for i, f in enumerate(env.flowers):
            draw_flower(screen, f, flower_sprites_active[i], flower_sprites_wilted[i], SCALE, SCREEN_H)
        draw_bee(screen, env.bee, bee_sprite, SCALE, SCREEN_H)
 
        reached = any(not f.active for f in env.flowers)
        dist = env.nearest_flower_distance()
        status = (f"episode={episode_count}  step={env.step_count}  "
                  f"dist={dist:.1f}  reached={reached}  speed={steps_per_frame}x  "
                  f"[SPACE pause | R reset | UP/DOWN speed]")
        text_surf = font.render(status, True, COLOR_TEXT)
        bg_surf = pygame.Surface((text_surf.get_width() + 12, text_surf.get_height() + 8), pygame.SRCALPHA)
        bg_surf.fill(TEXT_BG)
        screen.blit(bg_surf, (4, 4))
        screen.blit(text_surf, (10, 8))
 
        pygame.display.flip()
        clock.tick(30)
 
    pygame.quit()
    sys.exit()
 
 
if __name__ == "__main__":
    main()


