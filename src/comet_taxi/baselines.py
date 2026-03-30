from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .constants import (
    ACTION_ACCEPT_ORDER,
    ACTION_GO_CHARGE,
    ACTION_MOVE_EAST,
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_WEST,
    ACTION_STAY,
)


class GreedyDispatchPolicy:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.cell_count = config.data.grid_rows * config.data.grid_cols

    def _zone_to_row_col(self, zone: int) -> tuple[int, int]:
        return divmod(zone, self.config.data.grid_cols)

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        actions = np.zeros(self.config.env.nmax, dtype=np.int64)
        demand = observation["fleet_signature"][
            self.cell_count * 4 * 3 : self.cell_count * 4 * 3 + self.cell_count
        ]

        for slot in range(self.config.env.nmax):
            if observation["agent_mask"][slot] <= 0:
                continue
            action_mask = observation["action_mask"][slot]
            zone = int(observation["local_obs"][slot, 0]) - 1
            soc = float(observation["local_obs"][slot, 4])
            if action_mask[ACTION_ACCEPT_ORDER] > 0:
                actions[slot] = ACTION_ACCEPT_ORDER
                continue
            if soc < self.config.env.battery_low_threshold and action_mask[ACTION_GO_CHARGE] > 0:
                actions[slot] = ACTION_GO_CHARGE
                continue

            row, col = self._zone_to_row_col(max(zone, 0))
            neighbor_candidates = [
                (ACTION_MOVE_NORTH, row - 1, col),
                (ACTION_MOVE_SOUTH, row + 1, col),
                (ACTION_MOVE_EAST, row, col + 1),
                (ACTION_MOVE_WEST, row, col - 1),
            ]
            best_action = ACTION_STAY
            best_demand = demand[zone] if zone >= 0 else 0.0
            for action, next_row, next_col in neighbor_candidates:
                if action_mask[action] <= 0:
                    continue
                next_zone = next_row * self.config.data.grid_cols + next_col
                if 0 <= next_zone < self.cell_count and demand[next_zone] > best_demand:
                    best_demand = demand[next_zone]
                    best_action = action
            actions[slot] = best_action
        return actions
