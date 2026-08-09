"""
bee_env.py
----------
Core simulation environment for the bee task.
 
Design:
- 2D continuous world, bounded box.
- Bee: unicycle kinematics (x, y, heading theta). Controller outputs
  (forward_speed, turn_rate) each step.
- Flower(s): static points in the world, each with a "scent" strength.
- Sensory model: SIMPLIFIED GRADIENT VECTOR abstraction.
    For the nearest (or combined) flower, the bee senses:
      - relative_angle: angle to the flower in the bee's own reference
        frame (i.e. already rotated by -theta), normalized to [-1, 1]
        via division by pi.
      - strength: scalar signal that decays with distance (bounded in
        (0, 1]), analogous to scent/gradient intensity.
    This is deliberately simple (no need to move to sense direction,
    unlike antenna-based sampling) so a first ANN+ES pipeline can be
    stood up and debugged quickly. Swapping in antenna-based sampling
    later only requires changing Bee.sense(), nothing else.
 
This module has NO dependency on pygame or any NN library -- it is pure
simulation logic, so it can be reused headless (for evolution) and with
rendering (for visualization/demos).
"""

from dataclasses import dataclass, field
import numpy as np



@dataclass
class Flower:
    """Flower in the environment: has position, scent and active (will go inactive after being pollinated)"""

    x: float
    y: float
    strength: float = 1.0 #Scent intensity at the flower's location
    active: bool = True #Whether the flower has been visited (False) by the bee or not yet (True)


    @property
    def pos(self):
        return np.array([self.x, self.y])


@dataclass
class Bee:
    """Bee in the environment: position, heading angle."""
    x: float
    y: float
    theta: float                    # heading angle in radians
    max_speed: float = 20.0         # World units / seconds
    max_turn_rate: float = 4.0      # Radians / seconds
    trail: list = field(default_factory=list)  # For visualization/debugging

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def step(self, forward_speed: float, turn_rate: float, dt: float):
        """Update the bee's position and heading based on control inputs.
        
        forward_speed, turn_rate are expected in [-1, 1] (controller
        output convention) and get scaled by max_speed / max_turn_rate."""

        forward_speed = float(np.clip(forward_speed, -1.0, 1.0)) * self.max_speed
        turn_rate = float(np.clip(turn_rate, -1.0, 1.0)) * self.max_turn_rate

        # Update the heading angle
        self.theta += turn_rate * dt
        self.theta = float((self.theta + np.pi) % (2 * np.pi) - np.pi)  # Normalize to [-pi, pi]

        # Update position
        self.x = float(self.x + forward_speed * np.cos(self.theta) * dt)
        self.y = float(self.y + forward_speed * np.sin(self.theta) * dt)

        self.trail.append((self.x, self.y))  # Record the position for trail visualization

    def sense(self, flowers, decay: float = 0.05):
        """
        Gradient-vector Sensory Model

        Returns a tuple (relative_angle, strength) describing the NEAREST
        active flower:
        - relative_angle_norm in [-1, 1]: angle to the flower from the bee's heading perspective, 
            normalized by pi. 0 = straight ahead, +/-1 = directly behind.
        - strength in (0, 1]: scent intensity, exponentially decaying with distance.
            i.e. strength = flowert.strength * exp(-decay * distance_to_flower)
        
        If there are no active flowers, returns (0.0, 0.0).
        """

        active = [f for f in flowers if f.active]
        if not active:
            return 0.0, 0.0 # No active flowers

        dists = [np.linalg.norm(self.pos - f.pos) for f in active] #distance between bee and each flower
        nearest_idx = int(np.argmin(dists))
        nearest = active[nearest_idx] #closest flower to the bee
        dist = dists[nearest_idx] #distance to closest flower

        dx,dy = nearest.x - self.x, nearest.y - self.y #vector from bee to flower
        angle_to_flower = np.arctan2(dy, dx) #angle to flower in world frame
        relative_angle = angle_to_flower - self.theta #relative angle from bee's heading
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi #normalize to [-pi, pi]
        relative_angle_norm = relative_angle / np.pi #normalize to [-1, 1]

        strength = nearest.strength * np.exp(-decay * dist) #exponentially decaying scent strength

        return float(relative_angle_norm), float(strength)


class Environment:
    """ Bounded 2D map holding one bee and multiple flowers. """

    def __init__(self, width = 100.0, height = 100.0, n_flowers = 1, 
                dt = 0.1, max_steps = 300, seed = None, 
                flower_positions = None, bee_start = None):
        self.width = width
        self.height = height
        self.dt = dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # Place Flowers
        if flower_positions is not None:
            self.flowers = [Flower(x,y) for (x,y) in flower_positions]
        else:
            self.flowers = [
                Flower(
                    self.rng.uniform(0.2, 0.8) * self.width,
                    self.rng.uniform(0.2, 0.8) * self.height,
                )
                for _ in range(n_flowers)
            ]

        # Place Bee
        if bee_start is not None:
            bx, by, btheta = bee_start
        else:
            bx, by = width / 2, height / 2
            btheta = self.rng.uniform(-np.pi, np.pi)
        self.bee = Bee(bx, by, btheta)

        self.step_count = 0
        self.done = False

    def reset(self):
        self.step_count = 0
        self.done = False
        self.bee.trail = []

    def get_observation(self):
        """Return the bee's sensory input vector the controller (NN) will see."""
        rel_angle, strength = self.bee.sense(self.flowers)
        return np.array([rel_angle, strength], dtype=np.float32)

    def step(self, forward_speed: float, turn_rate: float):
        """Advance the simulation by one time step, given the bee's control inputs."""

        self.bee.step(forward_speed, turn_rate, self.dt)
        self.step_count += 1

        # Clamp bee inside the map boundaries
        self.bee.x = float(np.clip(self.bee.x, 0.0, self.width))
        self.bee.y = float(np.clip(self.bee.y, 0.0, self.height))

        #Check for the closest flower within a radius
        reach_radius = 2.0 
        for f in self.flowers:
            if f.active and np.linalg.norm(self.bee.pos - f.pos) <= reach_radius:
                f.active = False #Pollinate the flower
                break #Only pollinate one flower per step

        if self.step_count >= self.max_steps or all(not f.active for f in self.flowers):
            self.done = True

        return self.get_observation(), self.done

    def nearest_flower_distance(self):
        """Return the distance to the nearest active flower, or None if there are no active flowers."""
        dists = [np.linalg.norm(self.bee.pos - f.pos) for f in self.flowers if f.active]
        return float(min(dists)) if dists else None

    
                                 

