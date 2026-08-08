"""
render.py
---------
Visual layer for the bee environment. All sprites are drawn procedurally
with pygame primitives (no external image assets needed), then cached as
surfaces so per-frame rendering is cheap.

Exposes:
    generate_field_texture(px_w, px_h, seed) -> Surface
    make_bee_sprite(size) -> Surface   (facing along +x, i.e. theta=0)
    make_flower_sprite(size, hue_variant) -> Surface
    draw_bee(screen, bee, sprite, scale, screen_h)
    draw_flower(screen, flower, sprite, scale, screen_h)
"""

import math
import numpy as np
import pygame


# ---------------------------------------------------------------------
# Field texture (grass background)
# ---------------------------------------------------------------------
def generate_field_texture(px_w, px_h, seed=42):
    """Procedural grass field: base gradient + scattered blade/tuft detail
    + a few dirt patches. Cached once, reused every frame."""
    surf = pygame.Surface((px_w, px_h))
    rng = np.random.default_rng(seed)

    base_top = np.array([86, 148, 68])
    base_bottom = np.array([68, 124, 54])
    for y in range(px_h):
        t = y / max(px_h - 1, 1)
        color = (base_top * (1 - t) + base_bottom * t).astype(int)
        pygame.draw.line(surf, tuple(color), (0, y), (px_w, y))

    # dirt / clover patches (soft blobs, drawn first so blades sit on top)
    n_patches = int((px_w * px_h) / 9000)
    for _ in range(n_patches):
        cx, cy = rng.uniform(0, px_w), rng.uniform(0, px_h)
        r = rng.uniform(10, 28)
        patch_kind = rng.random()
        if patch_kind < 0.5:
            color = (74, 132, 58)  # slightly darker grass clump
        else:
            color = (96, 160, 76)  # slightly lighter clump
        pygame.draw.ellipse(
            surf, color,
            (cx - r, cy - r * 0.5, r * 2, r)
        )

    # grass blade tufts
    n_blades = int((px_w * px_h) / 60)
    for _ in range(n_blades):
        x, y = rng.uniform(0, px_w), rng.uniform(0, px_h)
        h = rng.uniform(4, 9)
        lean = rng.uniform(-2, 2)
        shade = rng.integers(0, 2)
        color = (60, 110, 48) if shade == 0 else (110, 176, 88)
        pygame.draw.line(surf, color, (x, y), (x + lean, y - h), 1)

    return surf


# ---------------------------------------------------------------------
# Bee sprite
# ---------------------------------------------------------------------
def make_bee_sprite(size=22):
    """Draws a bee facing along +x (right), centered on a transparent
    surface of side `size*2.4` so rotation doesn't clip it."""
    canvas = int(size * 2.6)
    surf = pygame.Surface((canvas, canvas), pygame.SRCALPHA)
    cx, cy = canvas // 2, canvas // 2

    body_len = size * 1.0
    body_w = size * 0.5

    # --- wings: teardrop polygons, drawn behind the body, angled back ---
    wing_color = (255, 255, 255, 130)
    wing_outline = (210, 220, 230, 180)

    def teardrop(cxw, cyw, length, width, angle_deg):
        pts_local = [(0, 0), (length * 0.35, -width / 2),
                     (length, 0), (length * 0.35, width / 2)]
        a = math.radians(angle_deg)
        pts = []
        for (px, py) in pts_local:
            rx = px * math.cos(a) - py * math.sin(a)
            ry = px * math.sin(a) + py * math.cos(a)
            pts.append((cxw + rx, cyw + ry))
        return pts

    upper_wing = teardrop(cx - size * 0.05, cy - body_w * 0.35, size * 0.85, size * 0.4, -35)
    lower_wing = teardrop(cx - size * 0.15, cy + body_w * 0.35, size * 0.7, size * 0.32, 25)
    pygame.draw.polygon(surf, wing_color, upper_wing)
    pygame.draw.polygon(surf, wing_outline, upper_wing, width=1)
    pygame.draw.polygon(surf, wing_color, lower_wing)
    pygame.draw.polygon(surf, wing_outline, lower_wing, width=1)

    # --- abdomen: yellow ellipse with distinct black stripes, back half ---
    abdomen_rect = (cx - body_len * 0.15, cy - body_w / 2, body_len * 0.65, body_w)
    pygame.draw.ellipse(surf, (255, 195, 35), abdomen_rect)
    pygame.draw.ellipse(surf, (60, 45, 15), abdomen_rect, width=1)

    # stripes as narrow vertical bars clipped to the abdomen ellipse area
    n_stripes = 3
    abdomen_left = cx - body_len * 0.15
    abdomen_span = body_len * 0.65
    for i in range(n_stripes):
        frac = 0.28 + i * 0.28
        sx = abdomen_left + abdomen_span * frac
        # height tapers slightly to feel ellipse-clipped without a mask pass
        pygame.draw.line(surf, (35, 25, 12), (sx, cy - body_w * 0.48), (sx, cy + body_w * 0.48),
                          max(2, int(size * 0.09)))

    # --- thorax: fuzzy tan circle between abdomen and head ---
    thorax_r = size * 0.28
    thorax_x = cx - body_len * 0.15
    pygame.draw.circle(surf, (150, 110, 60), (int(thorax_x), int(cy)), int(thorax_r))

    # --- head: dark circle at the front (+x side) ---
    head_r = size * 0.22
    head_x = cx + body_len / 2 - head_r * 0.3
    pygame.draw.circle(surf, (35, 28, 15), (int(head_x), int(cy)), int(head_r))
    # tiny eye highlight
    pygame.draw.circle(surf, (255, 255, 255), (int(head_x + head_r * 0.3), int(cy - head_r * 0.3)), max(1, int(head_r * 0.18)))

    # antennae
    pygame.draw.line(surf, (35, 28, 15), (head_x, cy - head_r * 0.6),
                      (head_x + size * 0.3, cy - size * 0.55), 1)
    pygame.draw.line(surf, (35, 28, 15), (head_x, cy + head_r * 0.6),
                      (head_x + size * 0.3, cy + size * 0.55), 1)

    return surf


# ---------------------------------------------------------------------
# Flower sprite
# ---------------------------------------------------------------------
_FLOWER_PALETTES = [
    ((235, 90, 140), (255, 210, 90)),   # pink petals, yellow center
    ((240, 130, 60), (255, 225, 110)),  # orange petals
    ((200, 100, 230), (255, 225, 110)), # purple petals
    ((235, 235, 245), (255, 200, 60)),  # white petals, gold center
]


def make_flower_sprite(size=16, variant=0, wilted=False):
    canvas = size * 3
    surf = pygame.Surface((canvas, canvas), pygame.SRCALPHA)
    cx, cy = canvas // 2, canvas // 2

    petal_color, center_color = _FLOWER_PALETTES[variant % len(_FLOWER_PALETTES)]
    if wilted:
        petal_color = (120, 115, 100)
        center_color = (150, 140, 110)

    # small stem/leaf for grounding
    pygame.draw.line(surf, (60, 120, 50), (cx, cy), (cx, cy + size * 0.9), 2)
    pygame.draw.ellipse(surf, (70, 140, 60),
                         (cx - size * 0.35, cy + size * 0.35, size * 0.5, size * 0.25))

    n_petals = 5
    petal_w, petal_h = size * 0.42, size * 0.62
    petal_offset = size * 0.42  # distance from center to petal center
    outline_color = tuple(max(0, c - 60) for c in petal_color)
    for i in range(n_petals):
        angle = 2 * math.pi * i / n_petals
        petal_surf = pygame.Surface((petal_w, petal_h), pygame.SRCALPHA)
        pygame.draw.ellipse(petal_surf, petal_color, (0, 0, petal_w, petal_h))
        pygame.draw.ellipse(petal_surf, outline_color, (0, 0, petal_w, petal_h), width=1)
        # rotate so the petal points outward from the flower center
        rotated = pygame.transform.rotate(petal_surf, -math.degrees(angle) + 90)
        r_rect = rotated.get_rect(center=(cx + math.sin(angle) * petal_offset,
                                           cy - math.cos(angle) * petal_offset))
        surf.blit(rotated, r_rect)

    pygame.draw.circle(surf, center_color, (cx, cy), int(size * 0.3))
    pygame.draw.circle(surf, (180, 140, 20) if not wilted else (110, 100, 80),
                        (cx, cy), int(size * 0.3), width=1)

    return surf


# ---------------------------------------------------------------------
# Draw helpers (world -> screen)
# ---------------------------------------------------------------------
def world_to_screen(x, y, scale, screen_h):
    return int(x * scale), int(screen_h - y * scale)


def draw_bee(screen, bee, sprite, scale, screen_h, trail_color=(255, 210, 60)):
    if len(bee.trail) > 1:
        pts = [world_to_screen(x, y, scale, screen_h) for x, y in bee.trail[-250:]]
        if len(pts) > 1:
            trail_surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            pygame.draw.lines(trail_surf, (*trail_color, 120), False, pts, 2)
            screen.blit(trail_surf, (0, 0))

    bx, by = world_to_screen(bee.x, bee.y, scale, screen_h)
    # screen y is flipped relative to world y, so on-screen rotation angle
    # is the negative of the world heading (in degrees).
    rotated = pygame.transform.rotate(sprite, math.degrees(bee.theta))
    rect = rotated.get_rect(center=(bx, by))
    screen.blit(rotated, rect)


def draw_flower(screen, flower, sprite_active, sprite_wilted, scale, screen_h):
    fx, fy = world_to_screen(flower.x, flower.y, scale, screen_h)
    sprite = sprite_active if flower.active else sprite_wilted
    rect = sprite.get_rect(center=(fx, fy))
    screen.blit(sprite, rect)