import streamlit as st
import pickle
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


class ParkingGrid:
    def __init__(self, size=10, start=(0,0), parking_spots=[(9,9)],
                 obstacles=None, moving_humans=None,
                 move_penalty=-2, collision_penalty=-50, park_reward=200,
                 boundary_penalty=-20, reward_shaping=True, shaping_coeff=0.1,
                 slip_prob=0.0):

        self.size = size
        self.start = start
        self.parking_spots = set(parking_spots)
        self.static_obstacles = set(obstacles) if obstacles else set()
        self.moving_humans = moving_humans if moving_humans else []
        self.obstacles = self.static_obstacles | {h["pos"] for h in self.moving_humans}

        self.move_penalty = move_penalty
        self.collision_penalty = collision_penalty
        self.park_reward = park_reward
        self.boundary_penalty = boundary_penalty
        self.reward_shaping = reward_shaping
        self.shaping_coeff = shaping_coeff
        self.slip_prob = slip_prob
        
        if not hasattr(self, "goal_candidates"):
            self.goal_candidates = []

        self.reset()

    # --- NEW HELPER METHOD ---
    def _get_state(self):
        """Returns the 4-tuple state: (Row, Col, Goal_Idx, Last_Action)"""
        return (self.state[0], self.state[1], self.goal_idx, self.prev_action)

    def reset(self):
        # 1. Random Start
        if hasattr(self, "start_candidates") and self.start_candidates:
            self.start = random.choice(self.start_candidates)
        
        # 2. Random Goal
        self.goal_idx = 0
        if self.goal_candidates:
            self.goal_idx = random.randint(0, len(self.goal_candidates) - 1)
            self.parking_spots = {self.goal_candidates[self.goal_idx]}
        
        self.state = self.start
        self.steps_taken = 0
        self.prev_action = 4  # 4 = "Start/No Action"
        self.visit_count = {}
        
        # Call the helper
        return self._get_state()

    def _in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def _nearest_goal_distance(self, pos):
        if not self.parking_spots: return 0
        return min(abs(pos[0]-g[0]) + abs(pos[1]-g[1]) for g in self.parking_spots)

    def _update_moving_humans(self):
        new_positions = set()
        for h in self.moving_humans:
            x, y = h["pos"]
            if h["axis"] == "h":
                ny = y + h["dir"]
                if ny < h["min"] or ny > h["max"]:
                    h["dir"] *= -1
                    ny = y + h["dir"]
                h["pos"] = (x, ny)
            else:
                nx = x + h["dir"]
                if nx < h["min"] or nx > h["max"]:
                    h["dir"] *= -1
                    nx = x + h["dir"]
                h["pos"] = (nx, y)
            new_positions.add(h["pos"])
        
        self.obstacles = self.static_obstacles | new_positions
        if hasattr(self, "visual_objects"):
            self.visual_objects["human"] = new_positions

    def step(self, action):
        self._update_moving_humans()
        self.steps_taken += 1

        if self.slip_prob > 0 and np.random.rand() < self.slip_prob:
            action = np.random.randint(4)

        x, y = self.state
        if action == 0: nx, ny = x-1, y
        elif action == 1: nx, ny = x+1, y
        elif action == 2: nx, ny = x, y-1
        elif action == 3: nx, ny = x, y+1
        else: nx, ny = x, y

        info = {"is_collision": False, "is_boundary": False, "is_parked": False}
        done = False
        
        # 1. Boundary Check
        if not self._in_bounds(nx, ny):
            info["is_boundary"] = True
            self.prev_action = action
            return self._get_state(), self.boundary_penalty, done, info

        next_state = (nx, ny)

        # 2. Collision Check
        if next_state in self.obstacles:
            info["is_collision"] = True
            self.prev_action = action
            return self._get_state(), self.collision_penalty, done, info

        # 3. Parked Check
        if next_state in self.parking_spots:
            self.state = next_state
            self.prev_action = action
            info["is_parked"] = True
            return self._get_state(), self.park_reward, True, info

        # 4. Normal Move
        reward = self.move_penalty
        
        # ZIG-ZAG PENALTY
        # If not start (4) and action changed, penalize!
        if self.prev_action != 4 and action != self.prev_action:
            reward -= 10.0 
        
        self.prev_action = action
        
        # Revisit Penalty
        self.visit_count[next_state] = self.visit_count.get(next_state, 0) + 1
        if self.visit_count[next_state] > 1:
            reward -= 1.5

        # Anti-wandering
        if self.steps_taken > 20: reward -= 1
        if self.steps_taken > 50: reward -= 2

        if hasattr(self, "visual_objects") and next_state in self.visual_objects.get("human_walkway", set()):
            reward -= 0.5

        if self.reward_shaping:
            d0 = self._nearest_goal_distance(self.state)
            d1 = self._nearest_goal_distance(next_state)
            reward += self.shaping_coeff * (d0 - d1)

        self.state = next_state
        return self._get_state(), reward, done, info

    def get_state_space(self):
        states = []
        num_goals = len(self.goal_candidates) if self.goal_candidates else 1
        
        # 4D State Space: Row x Col x Goals x PreviousActions(5)
        for i in range(self.size):
            for j in range(self.size):
                for g in range(num_goals):
                    for a in range(5): 
                        states.append((i, j, g, a))
        return states
    
    def render_map(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        for p in self.obstacles: grid[p] = -1
        for p in self.parking_spots: grid[p] = 2
        grid[self.start] = 3
        return grid

# ==========================================
# 2. BUILDERS
# ==========================================
def env_builder_easy():
    obstacle_map = {
        "parking_slots": {
            (2,0),(3,0),(4,0),(5,0),(7,0),
            (2,9),(4,9),(5,9),(6,9),(7,9),(8,9),
            (2,4),(2,5),
            (3,3),(4,3),(5,3),(6,3),
            (3,6),(4,6),(5,6),(6,6),
            (9,9)
        },

        "storage": {(0,8), (0,9)},

        "pillar": {
            (6,0), (3,9), (9,0),
            (7,3), (7,6),
            (2,3), (2,6)
        },

        "bush": {
            (3,4),(4,4),(5,4),(6,4),
            (3,5),(4,5),(5,5),(6,5)
        },

        "guard": {(9,2), (9,3)},

        "parked_car": {
            (2,0),(3,0),(4,0),(7,0),
            (2,9),(4,9),(5,9),(6,9),(8,9),
            (2,5),
            (3,3),(4,3),(5,3),(6,3),
            (3,6),(4,6),(5,6),(6,6)
        },

        "female": {(7,4), (7,5)},
        "waiting": {(4,1)},
        "exiting": {(1,4),(7,8)},
        "empty_soon": {(2,4), (5,0),(7,9)}
    }

    obstacles = set().union(
        *[v for k, v in obstacle_map.items() if k != "parking_slots"]
    )

    env = ParkingGrid(
        size=10,
        start=(0,0),
        parking_spots=[(9,9)],
        obstacles=obstacles,
        move_penalty=-3,
        collision_penalty=-50,
        park_reward=200,
        boundary_penalty=-20,
        reward_shaping=True,
        shaping_coeff=0.1,
        slip_prob=0.1
    )

    env.visual_objects = obstacle_map
    return env

# more obstacles, slight randomness
def env_builder_medium():
    obstacle_map = {

        # ---------------- PARKING SLOTS (VISUAL ONLY) ----------------
        "parking_slots": {
            # left vertical slots
            (3,2),(4,2),(5,2),(6,2),(8,2),(9,2),(10,2),(11,2),(12,2),(13,2),(14,2),
            (3,3),(4,3),(5,3),(6,3),(7,3),(8,3),(9,3),(10,3),(11,3),(12,3),(13,3),(14,3),

            # top horizontal slots
            (2,6),(2,7),(2,8),
            (2,11),(2,12),(2,13),(2,14),

            # second row slots
            (4,6),(4,7),(4,8),
            (4,11),(4,12),(4,13),(4,14),

            # mid vertical
            (8,7),(9,7),(10,7),
            (8,8),(9,8),(10,8),

            # lower mid
            (12,7),(13,7),
            (12,8),(13,8),

            # right clusters
            (9,12),(9,13),(9,14),
            (9,17),(9,18),(9,19),

            (10,12),(10,13),(10,14),
            (10,17),(10,18),(10,19),

            # female area
            (6,12),(6,13),(6,15),(6,16),(6,18),(6,19),
            (7,12),(7,13),(7,15),(7,16),(7,18),(7,19)
        },

        # ---------------- TICKET MACHINE AREA ----------------
        "ticket_machine": {
            (0,17),(0,18),(0,19),
            (1,17),(1,18),(1,19),
            (2,17),(2,18),(2,19),
            (3,17),(3,18),(3,19)
        },

        # ---------------- WATER LEAK / PIPE BOCOR ----------------
        "water_leak": {
            (13,14),(13,15),(13,16),(13,17),(13,18),
            (14,14),(14,15),(14,16),(14,17),(14,18),
            (15,17),(15,18)
        },

        # ---------------- BARRIER CONES ----------------
        "barrier_cone": {
            (15,13),(15,14),(15,15),(15,16),
            (16,17),(16,18),(16,19),
            (12,13),(12,14),(12,15),(12,16),(12,17),(12,18),(12,19),
            (13,13),(14,13),
            (13,19),(14,19),(15,19)
        },

        # ---------------- WALL ----------------
        "wall": {
            (15,2),(15,3),(15,4),(15,5),(15,6),(15,7),(15,8),(15,9),(15,10),(15,11),(15,12),
            (6,14),(6,17),
            (7,14),(7,17)
        },

        # ---------------- RAMP ----------------
        "ramp": {
            (17,18),(18,18),(19,18),
            (17,19),(18,19),(19,19),
            (17,17)
        },

        # ---------------- BUSH ----------------
        "bush": {
            (16,2),(16,3),(16,4),(16,5),(16,6),(16,7),(16,8),(16,9),(16,10),(16,11),(16,12),
            (17,2),(17,3),(17,4),(17,5),(17,6),(17,7),(17,8),(17,9),(17,10),(17,11),(17,12),
            (7,4),(7,5),(7,6),(7,7),
            (3,6),(3,7),(3,8),
            (3,11),(3,12),(3,13),(3,14)
        },

        # ---------------- PARKED CARS ----------------
        "parked_car": {
            (3,2),(6,2),(7,2),(8,2),(9,2),(10,2),(11,2),(12,2),(13,2),(14,2),
            (3,3),(4,3),(5,2),(6,3),(7,3),(8,3),(9,3),(11,3),(12,3),(13,3),(14,3),

            (2,6),(2,7),(2,8),
            (2,11),(2,13),(2,14),
            (4,6),(4,7),
            (4,11),(4,12),(4,13),(4,14),

            (9,12),(9,13),(9,14),
            (9,17),(9,18),(9,19),

            (10,12),(10,13),(10,14),
            (10,18),(10,19),

            (8,7),(10,7),(13,7),
            (8,8),(9,8),(10,8),
            (12,8),(13,8)
        },

        # ---------------- FEMALE PARKING (BLOCKED) ----------------
        "female": {
            (6,12),(6,13),(6,15),(6,16),(6,18),(6,19),
            (7,12),(7,13),(7,15),(7,16),(7,18),(7,19)
        },

        # ---------------- DYNAMIC OBJECTS ----------------
        "waiting": {(3,1)},
        "exiting": {(10,4),(5,4),(1,12),(5,8)},
        "empty_soon": {(4,2),(10,3),(5,3),(2,12),(4,8)}
    }

    # exclude parking slots from obstacles
    obstacles = set().union(
        *[v for k, v in obstacle_map.items() if k != "parking_slots"]
    )

    env = ParkingGrid(
        size=20,
        start=(17,16),
        parking_spots=[(10,17), (9,7), (12,7)],
        obstacles=obstacles,
        move_penalty=-3,
        collision_penalty=-50,
        park_reward=200,
        boundary_penalty=-20,
        reward_shaping=True,
        shaping_coeff=0.35,
        slip_prob=0.1
    )
    
    # 🔹 RANDOM GOAL PER EPISODE
    env.goal_candidates = [(10,17), (9,7), (12,7)] 
    env.visual_objects = obstacle_map
    return env

# hardest layout with moving obstacles
def env_builder_hard():
    moving_humans = [
        # ---------- HUMANS ----------
        {"pos": (25,29), "axis": "h", "min": 23, "max": 29, "dir": -1},
        {"pos": (26,29), "axis": "h", "min": 23, "max": 29, "dir": -1},

        {"pos": (24,23), "axis": "v", "min": 17, "max": 24, "dir":  1},
        {"pos": (24,24), "axis": "v", "min": 17, "max": 24, "dir": -1},

        {"pos": (16,23), "axis": "v", "min": 9,  "max": 16, "dir": -1},
        {"pos": (16,24), "axis": "v", "min": 9,  "max": 16, "dir":  1},

        {"pos": (26,7),  "axis": "v", "min": 18, "max": 26, "dir": -1},

        {"pos": (9,7),   "axis": "v", "min": 9,  "max": 15, "dir":  1},
        {"pos": (9,8),   "axis": "v", "min": 9,  "max": 15, "dir": -1},

        # ---------- TROLLEYS ----------
        {"pos": (29,20), "axis": "v", "min": 22, "max": 29, "dir": -1},
        {"pos": (29,21), "axis": "v", "min": 22, "max": 29, "dir":  1},
        {"pos": (29,22), "axis": "v", "min": 22, "max": 29, "dir": -1},
    ]

    human_walkway = {
        (25,23),(25,24),(25,25),(25,26),(25,27),(25,28),(25,29),
        (26,23),(26,24),(26,25),(26,26),(26,27),(26,28),(26,29),

        (17,23),(18,23),(19,23),(20,23),(21,23),(22,23),(23,23),(24,23),
        (17,24),(18,24),(19,24),(20,24),(21,24),(22,24),(23,24),(24,24),

        (9,23),(10,23),(11,23),(12,23),(13,23),(14,23),(15,23),(16,23),
        (9,24),(10,24),(11,24),(12,24),(13,24),(14,24),(15,24),(16,24),

        (18,7),(19,7),(20,7),(21,7),(22,7),(23,7),(24,7),(25,7),(26,7),

        (9,7),(10,7),(11,7),(12,7),(13,7),(14,7),(15,7),
        (9,8),(10,8),(11,8),(12,8),(13,8),(14,8),(15,8)
    }

    trolley_road = {
        (22,20),(23,20),(24,20),(25,20),(26,20),(27,20),(28,20),(29,20),
        (22,21),(23,21),(24,21),(25,21),(26,21),(27,21),(28,21),(29,21),
        (22,22),(23,22),(24,22),(25,22),(26,22),(27,22),(28,22),(29,22)
    }
    
    obstacle_map = {

        # ---------------- PARKING SLOTS (VISUAL ONLY) ----------------
        "parking_slots": {
            # rows 3–4, cols 2–16
            (3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),(3,12),(3,13),(3,14),(3,15),(3,16),
            (4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(4,15),(4,16),

            # rows 7–8
            (7,5),(7,6),(7,7),(7,8),(7,9),(7,10),(7,11),(7,12),(7,13),(7,14),
            (7,17),(7,18),(7,19),(7,20),(7,21),(7,22),(7,23),
            (8,5),(8,6),(8,7),(8,8),(8,9),(8,10),(8,11),(8,12),(8,13),(8,14),
            (8,17),(8,18),(8,19),(8,20),(8,21),(8,22),(8,23),

            # rows 12–13
            (12,11),(12,12),(12,13),(12,14),(12,15),(12,16),
            (12,18),(12,19),(12,20),(12,21),(12,22),
            (13,11),(13,12),(13,13),(13,14),(13,15),(13,16),
            (13,18),(13,19),(13,20),(13,21),(13,22),

            # rows 16–17
            (16,8),(16,9),(16,10),(16,11),(16,12),(16,13),
            (16,15),(16,16),(16,17),(16,18),(16,19),
            (17,8),(17,9),(17,10),(17,11),(17,12),(17,13),
            (17,15),(17,16),(17,17),(17,18),(17,19),

            # rows 20–21
            (20,11),(20,12),(20,13),(20,14),(20,15),(20,16),
            (20,18),(20,19),(20,20),(20,21),(20,22),
            (21,11),(21,12),(21,13),(21,14),(21,15),(21,16),
            (21,18),(21,19),(21,20),(21,21),(21,22),

            # rightmost columns
            (7,27),(8,27),(9,27),(11,27),(12,27),(13,27),(15,27),(16,27),(17,27),(19,27),(20,27),(21,27),
            (7,28),(8,28),(9,28),(11,28),(12,28),(13,28),(15,28),(16,28),(17,28),(19,28),(20,28),(21,28)
        },

        # ---------------- TICKET MACHINE ----------------
        "ticket_machine": {
            (27,4),(28,4),(29,4),
            (27,5),(28,5),(29,5),
            (27,6),(28,6),(29,6)
        },

        # ---------------- GUARD ----------------
        "guard": {
            (27,0),(28,0),(29,0),
            (27,1),(28,1),(29,1),
            (27,2),(28,2),(29,2)
        },

        # ---------------- OKU ----------------
        "oku": {
            (27,8),(28,8),(27,9),(28,9),
            (27,11),(28,11),(27,12),(28,12),
            (27,14),(28,14),(27,15),(28,15),
            (27,17),(28,17),(27,18),(28,18)
        },

        # ---------------- LIFT & ESCALATOR ----------------
        "lift&escalor": {
            (27,23),(28,23),(29,23),
            (27,24),(28,24),(29,24),
            (27,25),(28,25),(29,25),
            (27,26),(28,26),(29,26),
            (27,27),(28,27),(29,27),
            (27,28),(28,28),(29,28),
            (27,29),(28,29),(29,29)
        },

        # ---------------- CONSTRUCTION ZONE ----------------
        "construction zone": {
            (2,21),(3,21),(4,21),
            (2,22),(3,22),(4,22),
            (2,23),(3,23),(4,23),
            (2,24),(3,24),(4,24)
        },

        # ---------------- BARRIER CONE ----------------
        "barrier_cone": {
            (2,20),(3,20),(4,20),
            (5,21),(5,22),(5,23),(5,24),
            (2,25),(3,25)
        },

        "pillar": {
            (8,0)
        },

        # ---------------- WALL ----------------
        "wall": {
            (0,0),(1,0),
            (16,7),(17,7),(16,14),(17,14),
            (12,10),(13,10),(12,17),(13,17),
            (20,10),(21,10),(20,17),(21,17),
            (10,27),(10,28),(14,27),(14,28),(18,27),(18,28),
            (29,8),(29,9),(29,10),(29,11),(29,12),(29,13),(29,14),(29,15),(29,16),(29,17),(29,18),
            (13,2),(14,2),(15,2),(16,2),(17,2),(18,2),(19,2),(20,2),(21,2),(22,2)
        },

        # ---------------- RAMP ----------------
        "ramp": {
            (0,5),(0,6),(0,7),(0,8),(0,9),(0,10),(0,11),(0,12),(0,13),(0,14),
            (1,5),(1,6),(1,7),(1,8),(1,9),(1,10),(1,11),(1,12),(1,13),(1,14),

            (0,20),(0,21),(0,22),(0,23),(0,24),(0,25),(0,26),(0,27),(0,28),(0,29),
            (1,20),(1,21),(1,22),(1,23),(1,24),(1,25),(1,26),(1,27),(1,28),(1,29),

            (13,0),(14,0),(15,0),(16,0),(17,0),(18,0),(19,0),(20,0),(21,0),(22,0),
            (13,1),(14,1),(15,1),(16,1),(17,1),(18,1),(19,1),(20,1),(21,1),(22,1),

            (13,3),(14,3),(15,3),(16,3),(17,3),(18,3),(19,3),(20,3),(21,3),(22,3),
            (13,4),(14,4),(15,4),(16,4),(17,4),(18,4),(19,4),(20,4),(21,4),(22,4)
        },

        # ---------------- BUSH ----------------
        "bush": {
            (24,8),(24,9),(24,10),(24,11),(24,12),(24,13),(24,14),(24,15),(24,16),(24,17),(24,18),
            (25,8),(25,9),(25,10),(25,11),(25,12),(25,13),(25,14),(25,15),(25,16),(25,17),(25,18),
            (4,25),(4,26),(4,27),(4,28),(4,29),
            (5,25),(5,26),(5,27),(5,28),(5,29)
        },

        "parked_car": {
            # row 3
            (3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10),(3,11),
            (3,13),(3,14),(3,15),(3,16),
        
            # row 4
            (4,2),(4,3),(4,4),(4,5),(4,7),(4,8),(4,9),(4,10),(4,11),(4,12),(4,13),(4,14),(4,15),(4,16),(4,17),
        
            # row 7
            (7,5),(7,6),(7,7),(7,8),(7,9),(7,10),
            (7,11),(7,12),(7,13),(7,14),
            (7,5),(7,6),(7,7),(7,8),(7,9),(7,10),
            (7,12),(7,13),(7,14),
            (7,17),(7,18),(7,19),(7,20),(7,21),
            (7,23),
        
            # row 8
            (8,5),(8,6),(8,7),(8,8),(8,9),(8,10),(8,11),(8,12),(8,13),(8,14),
            (8,17),(8,18),
            (8,20),(8,21),(8,22),(8,23),
        
            # row 12
            (12,11),(12,12),(12,13),(12,14),(12,15),(12,16),
            (12,18),(12,19),(12,20),(12,21),(12,22),
        
            # row 13
            (13,11),(13,12),(13,13),(13,14),(13,15),(13,16),
            (13,18),(13,19),
            (13,21),(13,22),
        
            # row 16
            (16,8),(16,9),(16,10),(16,11),(16,12),(16,13),
            (16,15),
            (16,17),
            (16,19),
        
            # row 17
            (17,8),
            (17,10),(17,11),(17,12),(17,13),
            (17,15),(17,16),(17,17),(17,18),(17,19),
        
            # row 20
            (20,11),
            (20,13),(20,14),(20,15),(20,16),
            (20,18),(20,19),(20,20),(20,21),(20,22),
        
            # row 21
            (21,11),(21,12),(21,13),(21,14),(21,15),(21,16),
            (21,18),(21,19),(21,20),(21,21),(21,22),
        
            # column 27
            (7,27),(8,27),(9,27),
            (11,27),
            (13,27),
            (15,27),(16,27),(17,27),
            (19,27),(20,27),(21,27),
        
            # column 28
            (7,28),(8,28),(9,28),
            (11,28),(12,28),(13,28),
            (15,28),(16,28),(17,28),
            (20,28),(21,28)
        },

        # ---------------- DYNAMIC OBJECTS ----------------
        "waiting": {(5,7),(6,12),(6,23),(9,18),(14,19),(15,15),(19,11)},
        "exiting": {(2,12),(18,9),(22,18),(12,26)},
        "empty_soon": {(4,6),(3,12),(8,19),(7,22),(13,20),(16,16),(17,9),(20,12),(21,18),(12,27)},

        "human_walkway": human_walkway,   # ✅ VISUAL / SEMANTIC
        "trolley_road": trolley_road,
        "human": {h["pos"] for h in moving_humans},  # ✅ DYNAMIC OBSTACLE
    }

    # only blocking objects go into obstacles
    # BUILD BLOCKING OBSTACLES (EXCLUDE VISUALS)
    # ==========================================================
    obstacles = set().union(
        *[v for k, v in obstacle_map.items()
          if k not in ("parking_slots", "human", "human_walkway", "trolley_road")]
    )

    # Safety check
    for h in moving_humans:
        assert h["pos"] not in obstacles, f"Moving entity starts inside obstacle at {h['pos']}"

    # ==========================================================
    # ENVIRONMENT
    # ==========================================================
    env = ParkingGrid(
        size=30,
        start=(1,4),   # temporary, will be overridden
        parking_spots=[(16,18), (19,28)],
        obstacles=obstacles,
        moving_humans=moving_humans,
        move_penalty=-3,
        collision_penalty=-50,
        park_reward=200,
        boundary_penalty=-20,
        reward_shaping=True,
        shaping_coeff=0.45,
        slip_prob=0.1
    )

    env.text_labels = [
        ((25, 2), ">> Ramp to Ground >>", 8),
        ((25, 4), "<< Ramp from Ground <<", 8),
        ((25, 12), "bushes / deco", 10),
        ((28, 26), "lift & escalator", 9),
        ((25, 26), "Humans walking", 8),
        ((16, 26), "Humans walking", 8),
        ((25, 20), "shopping cart / trolley\nmoving", 8),
        ((29, 0), "Guard\nhouse", 6),
        ((29, 9), "OKU", 7), ((29, 12), "OKU", 7), 
        ((29, 15), "OKU", 7), ((29, 18), "OKU", 7),
        ((20, 11), "parking", 10), ((20, 18), "slots", 10),
        ((15, 11), "parking", 10), ((15, 18), "slots", 10),
        ((10, 11), "parking", 10), ((10, 18), "slots", 10),
        ((5, 9), "parking slots", 10),
        ((5, 19), "parking slots", 10),
        ((4, 21), "Construction\nzone", 7),
        ((0, 9), "<< Ramp from P2 <<", 8),
        ((0, 24), "<< Ramp from Ground <<", 8),
        ((23, 12), "oil spill", 6),
    ]
    
    
    # 🔹 RANDOM START + GOAL PER EPISODE
    env.start_candidates = [(12,3), (1,4), (1,19)]
    env.goal_candidates = [(19,28), (16,18)]
    env.visual_objects = obstacle_map
    return env
# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_models():
    filename = "parking_models.pkl"
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data["q_tables"], data["dq_tables"]
    except FileNotFoundError:
        return None, None

q_tables, dq_tables = load_models()

def count_turns(path):
    if len(path) < 3: return 0
    turns = 0
    coords = [p[:2] for p in path]
    for i in range(len(coords) - 2):
        y1, x1 = coords[i]
        y2, x2 = coords[i+1]
        y3, x3 = coords[i+2]
        if (y2-y1, x2-x1) != (y3-y2, x3-x2):
            turns += 1
    return turns

def get_greedy_action(env, table, state_tuple):
    # 1. If state is not in the table, take a random action
    if state_tuple not in table:
        return np.random.randint(0, 4)

    # 2. Get Q-values
    q_values = table[state_tuple]

    # 3. Handle different data types
    if isinstance(q_values, dict):
        # If it's a dictionary (Action -> Value)
        return max(q_values, key=q_values.get)
    
    elif isinstance(q_values, (np.ndarray, list)):
        # If it's a NumPy array or List (Index is Action)
        return int(np.argmax(q_values))
        
    # Default fallback
    return np.random.randint(0, 4)

# --- ROTATED RENDER FUNCTION ---
def render_grid(env, path, title_color="black"):
    rows, cols = env.size, env.size
    
    # 1. Setup the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('white')
    
    # 3. Coordinate Transformer (Row->Y, Col->X)
    def transform(r, c):
        return c, r  # x=Col, y=Row

    # 4. Define Color Palette
    palette = {
        "wall":           "#000000",  # Updated to pure Black
        "bush":           "#2E7D32",  # Dark Green
        "guard_house":    "#37474F",  # Slate Grey
        "ticket_machine": "#1976D2",  # Blue
        "oku":            "#88C3E3",  # Updated to Lighter Blue
        "lift":           "#7B1FA2",  # Purple
        "construction":   "#FFA04D",  # Updated to Lighter Orange
        "ramp":           "#D7CCC8",  # Tan/Beige
        "water_leak":     "#795548",  # Brown
        "cone":           "#FF6F00",  # Amber
        "parking_slots":  "#BDBDBD",  # Light Grey
        "human":          "#F48FB1",  # Updated to Lighter Pink
        "human_walkway":  "#FFCDD2",  # Light Pink Area
        "trolley_road":   "#FFF9C4",  # Light Yellow
    }

    icon_map = {
        "bush":           ("✿", "white", 10),      # Green flower logo
        "oku":            ("♿", "white", 10),      # Blue wheelchair logo
        "wall":           ("WALL", "white", 5),     # Word "WALL"
        "guard_house":    ("⛉", "white", 12),      # Guard logo
        "ticket_machine": ("⌂", "white", 10),      # Ticket logo
        "lift":           ("⍗", "white", 10),      # Lift logo
        "construction":   ("xxx", "blue", 10),       # User asked for Water logo here
        "ramp":           ("r", "white", 5),     # Word "RAMP"
        "water_leak":     (":", "cyan", 10),       # Water logo
        "cone":           ("Δ", "white", 10),       # Cone logo (Warning sign)
        "parked_car":     ("℗", "black", 10),   # Car logo for obstacles
        "trolley_road":   ("♥︎", "black", 6),       # Trolley logo
    }

    # 5. DRAW VISUAL OBJECTS (With Black Edges)
    if hasattr(env, "visual_objects"):
        for obj_type, coords in env.visual_objects.items():
            # A. Draw Background Block
            color = palette.get(obj_type, "#EEEEEE")
            # Special case: Make parking slots transparent/light
            alpha = 0.5 if obj_type == "parking_slots" else 1.0
            
            for r, c in coords:
                x, y = transform(r, c)
                
                # Draw colored rectangle (base)
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                     facecolor=color, alpha=alpha,
                                     edgecolor="none", linewidth=0)
                ax.add_patch(rect)

                # B. Draw Icon/Logo (Overlay)
                if obj_type in icon_map:
                    # Check if the icon is text or image-based
                    if isinstance(icon_map[obj_type], tuple):  # This means it's a text-based icon
                        symbol, txt_color, f_size = icon_map[obj_type]
                        ax.text(x, y, symbol, 
                                ha='center', va='center', 
                                fontsize=f_size, color=txt_color, 
                                fontweight='bold', zorder=2)
                    else:  # Image-based icon
                        img_path = icon_map[obj_type]  # Get image path
                        img = plt.imread(img_path)  # Read the imag
                        cell_size = 1  # Each cell in the grid is 1 unit (can be adjusted based on your grid size)

                        # Set image size relative to the grid cell size
                        zoom_factor = cell_size * 0.025  # Adjust multiplier as necessary (e.g., 0.5 for half the grid size)
                        
                        imagebox = OffsetImage(img, zoom=zoom_factor)  # Adjust zoom for icon size
                        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                        ax.add_artist(ab)

    # 6. Draw Moving Humans
    if hasattr(env, "moving_humans"):
        for h in env.moving_humans:
            hr, hc = h["pos"]
            hx, hy = transform(hr, hc)
            # Draw Human Icon
            ax.text(hx, hy, "⌘", ha='center', va='center', fontsize=12, zorder=6)

    # 7. Draw Goals (Stars)
    if hasattr(env, 'parking_spots'):
        for gr, gc in env.parking_spots:
            gx, gy = transform(gr, gc)
            ax.plot(gx, gy, marker='*', color='#D50000', markersize=20, 
                    markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    # 8. Draw Path
    if len(path) > 1:
        raw_coords = [p[:2] for p in path]
        plot_coords = [transform(r, c) for r, c in raw_coords]
        path_x, path_y = zip(*plot_coords)
        ax.plot(path_x, path_y, color=title_color, linewidth=2, 
                alpha=0.7, marker='.', markersize=4, zorder=6)

    # 9. Draw Agent (Car Logo) 
    # agent grid position (row, col)
    ar, ac = env.state

    # convert grid coord -> plot coord
    ax_p, ay_p = transform(ar, ac)

    # draw the car icon
    img = plt.imread("b.png")
    imagebox = OffsetImage(img, zoom=0.035)
    ab = AnnotationBbox(imagebox, (ax_p, ay_p), frameon=False)
    ax.add_artist(ab)

    # Start Position
    if hasattr(env, "start"):
        sr, sc = env.start
        sx, sy = transform(sr, sc)
        ax.text(sx, sy, "", ha='center', va='center', color="white", fontweight='bold', zorder=5)
        ax.add_patch(plt.Circle((sx, sy), 0.4, color='green', alpha=0.3, zorder=4))

    # Current Agent (The Car)
    # 0=Up, 1=Down, 2=Left, 3=Right
    direction = env.prev_action if hasattr(env, "prev_action") else 4
    
    # Rotate car icon based on direction (using arrows or just a static car)
    # Since we can't easily rotate emoji text, we use a generic Front-Facing Car
    ax.text(ax_p, ay_p, "🏎️", ha='center', va='center', fontsize=16, zorder=10)

    # 10. Grid Lines
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect('equal')
    
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    return fig
    
# --- LEGEND DISPLAY FUNCTION ---
def display_color_legend_python():
    # Define data matching the render_grid palette
    legend_data = [
        {"Object": "Wall",            "Color": "#000000"},
        {"Object": "Bush",            "Color": "#2E7D32"},
        {"Object": "Guard House",     "Color": "#37474F"},
        {"Object": "Ticket Machine",  "Color": "#1976D2"},
        {"Object": "OKU Parking",     "Color": "#88C3E3"},
        {"Object": "Lift/Escalator",  "Color": "#7B1FA2"},
        {"Object": "Construction",    "Color": "#FFA04D"},
        {"Object": "Ramp",            "Color": "#D7CCC8"},
        {"Object": "Water Leak",      "Color": "#795548"},
        {"Object": "Cone",            "Color": "#FF6F00"},
        {"Object": "Parking Slots",   "Color": "#BDBDBD"},
        {"Object": "Moving Human",    "Color": "#F48FB1"},
        {"Object": "Walkway",         "Color": "#FFCDD2"},
        {"Object": "Trolley Road",    "Color": "#FFF9C4"},
    ]

    # Create DataFrame
    df = pd.DataFrame(legend_data)

    def color_background(val):
        return f'background-color: {val}; color: {val};'

    # Apply style
    styled_df = df.style.map(color_background, subset=['Color'])

    st.sidebar.dataframe(styled_df, hide_index=True, use_container_width=True)
    
# ==========================================
# 4. STREAMLIT APP LAYOUT
# ==========================================
st.set_page_config(page_title="RL Parking Comparison", page_icon="🏎️", layout="wide")

st.title("🏎️ Autonomous Parking: Q-Learning vs Double-Q")
st.markdown("### 🚦 Real-Time Comparison Dashboard")

# --- 1. SETUP SIDEBAR (CONTROLS) ---
st.sidebar.header("⚙️ Simulation Controls")

# A. Action Buttons (Top of Sidebar for easy access)
col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
start_btn = col_btn1.button("▶ Start", type="primary") # Primary makes it red/highlighted
pause_btn = col_btn2.button("⏸ Pause")
reset_btn = col_btn3.button("🔄 Reset")

# B. Settings
st.sidebar.divider()
selected_level = st.sidebar.selectbox("Select Map Level", ["easy", "medium", "hard"])
seed_input = st.sidebar.number_input("Map Seed (ID)", min_value=0, value=2925, step=1)
max_steps_input = st.sidebar.slider("Max Steps", 50, 500, 200)
speed = st.sidebar.slider("Speed (Delay)", 0.0, 0.5, 0.01)

# C. Logic for Buttons
if start_btn:
    st.session_state.run_active = True
if pause_btn:
    st.session_state.run_active = False
if reset_btn:
    # Reset helper logic
    st.session_state.current_seed = -1 # Force re-init
    st.session_state.run_active = False
    st.rerun()

# --- 2. INITIALIZATION ---
# Map string to builder functions
env_builders = {
    "easy": env_builder_easy,
    "medium": env_builder_medium,
    "hard": env_builder_hard
}

# Check init
if 'env_q' not in st.session_state or \
   st.session_state.get('current_level') != selected_level or \
   st.session_state.get('current_seed') != seed_input:
    
    # Init Q-Learning
    random.seed(seed_input)
    np.random.seed(seed_input)
    st.session_state.env_q = env_builders[selected_level]()
    st.session_state.env_q.reset()

    # Init Double-Q (Same Seed)
    random.seed(seed_input)
    np.random.seed(seed_input)
    st.session_state.env_dq = env_builders[selected_level]()
    st.session_state.env_dq.reset()

    # Store Data
    st.session_state.path_q = [st.session_state.env_q.state]
    st.session_state.path_dq = [st.session_state.env_dq.state]
    st.session_state.info_q = {}
    st.session_state.info_dq = {}
    st.session_state.done_q = False
    st.session_state.done_dq = False
    st.session_state.step_count = 0
    st.session_state.run_active = False
    
    st.session_state.current_level = selected_level
    st.session_state.current_seed = seed_input
       
# --- 3. DISPLAY LEGEND (IN EXPANDER) ---
                          
st.sidebar.divider()  # Optional separator
st.sidebar.markdown("### 🗺️ Object Legend")

if 'display_color_legend_python' in globals():
    display_color_legend_python()
else:
    st.warning("Legend function not found.")
        
# --- 4. MAIN DASHBOARD ---
col1, col2 = st.columns(2)

# Load Models
table_q = q_tables.get(selected_level, {})
table_dq = dq_tables.get(selected_level, {})

# --- HELPER: Display Metrics Table (Pandas) ---
def render_metrics_table(placeholder, path, info, steps):
    status = "🏎️ Driving..."
    if info.get("is_parked"): status = "✅ Parked"
    elif info.get("is_collision"): status = "💥 Crashed"
    elif info.get("is_boundary"): status = "🚧 Out of Bounds"
    elif steps >= max_steps_input: status = "⌛ Timeout"
    
    # Create Dataframe
    df = pd.DataFrame([
        {"Metric": "Status", "Value": status},
        {"Metric": "Steps",  "Value": steps},
        {"Metric": "Turns",  "Value": count_turns(path)}
    ])
    # Display clean table
    placeholder.dataframe(df, hide_index=True, use_container_width=True)

# --- LAYOUT COLUMNS ---

_, col1, col2, _ = st.columns([0.3, 2.5, 2.5, 0.3])

with col1:
    st.subheader("🤖 Q-Learning")
    plot_q = st.empty()
    st.caption("Live Metrics")
    metrics_q = st.empty()

with col2:
    st.subheader("🧠 Double-Q")
    plot_dq = st.empty()
    st.caption("Live Metrics")
    metrics_dq = st.empty()

# --- 5. ANIMATION LOOP ---
if st.session_state.run_active:
    while st.session_state.step_count < max_steps_input and st.session_state.run_active:
        
        # Step Q-Learning
        if not st.session_state.done_q:
            s = st.session_state.env_q._get_state()
            a = get_greedy_action(st.session_state.env_q, table_q, s)
            _, _, d, i = st.session_state.env_q.step(a)
            st.session_state.path_q.append(st.session_state.env_q.state)
            st.session_state.done_q = d
            st.session_state.info_q = i

        # Step Double-Q
        if not st.session_state.done_dq:
            s = st.session_state.env_dq._get_state()
            a = get_greedy_action(st.session_state.env_dq, table_dq, s)
            _, _, d, i = st.session_state.env_dq.step(a)
            st.session_state.path_dq.append(st.session_state.env_dq.state)
            st.session_state.done_dq = d
            st.session_state.info_dq = i

        st.session_state.step_count += 1

        # RENDER MAPS
        fig1 = render_grid(st.session_state.env_q, st.session_state.path_q, "#2979FF") # Blue title
        plot_q.pyplot(fig1)
        plt.close(fig1)

        fig2 = render_grid(st.session_state.env_dq, st.session_state.path_dq, "#D50000") # Red title
        plot_dq.pyplot(fig2)
        plt.close(fig2)

        # RENDER METRICS
        render_metrics_table(metrics_q, st.session_state.path_q, st.session_state.info_q, st.session_state.step_count)
        render_metrics_table(metrics_dq, st.session_state.path_dq, st.session_state.info_dq, st.session_state.step_count)

        # CHECK DONE
        if st.session_state.done_q and st.session_state.done_dq:
            st.session_state.run_active = False
            st.success("🏁 Race Finished!")
            
        time.sleep(speed)

else:
    # STATIC RENDER (When paused)
    fig1 = render_grid(st.session_state.env_q, st.session_state.path_q, "#2979FF")
    plot_q.pyplot(fig1)
    plt.close(fig1)

    fig2 = render_grid(st.session_state.env_dq, st.session_state.path_dq, "#D50000")
    plot_dq.pyplot(fig2)
    plt.close(fig2)
    
    # Final Metrics
    step_display_q = len(st.session_state.path_q)-1 if st.session_state.done_q else st.session_state.step_count
    step_display_dq = len(st.session_state.path_dq)-1 if st.session_state.done_dq else st.session_state.step_count
    
    render_metrics_table(metrics_q, st.session_state.path_q, st.session_state.info_q, step_display_q)
    render_metrics_table(metrics_dq, st.session_state.path_dq, st.session_state.info_dq, step_display_dq)
