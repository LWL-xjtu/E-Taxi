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

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        actions = np.zeros(self.config.env.nmax, dtype=np.int64)
        demand = observation.get("demand_vector")
        if demand is None:
            cell_count = int(self.config.data.cell_count)
            demand = observation["fleet_signature"][
                cell_count * 4 * 3 : cell_count * 4 * 3 + cell_count
            ]
        move_targets = observation.get("move_target_zones")

        for slot in range(self.config.env.nmax):
            if observation["agent_mask"][slot] <= 0:
                continue
            action_mask = observation["action_mask"][slot]
            zone = int(observation["local_obs"][slot, 0]) - 1
            soc = float(observation["local_obs"][slot, 4])
            if action_mask[ACTION_ACCEPT_ORDER] > 0:
                actions[slot] = ACTION_ACCEPT_ORDER
                continue
            charger_queue = observation.get("charger_queue")
            if (
                soc < self.config.planner.risk_trigger_soc
                and action_mask[ACTION_GO_CHARGE] > 0
                and (charger_queue is None or np.mean(charger_queue) <= self.config.env.max_queue_length)
            ):
                actions[slot] = ACTION_GO_CHARGE
                continue

            neighbor_candidates = [
                (ACTION_MOVE_NORTH, int(move_targets[slot, 0]) if move_targets is not None else zone),
                (ACTION_MOVE_SOUTH, int(move_targets[slot, 1]) if move_targets is not None else zone),
                (ACTION_MOVE_EAST, int(move_targets[slot, 2]) if move_targets is not None else zone),
                (ACTION_MOVE_WEST, int(move_targets[slot, 3]) if move_targets is not None else zone),
            ]
            best_action = ACTION_STAY
            best_demand = demand[zone] if zone >= 0 else 0.0
            for action, next_zone in neighbor_candidates:
                if action_mask[action] <= 0:
                    continue
                if 0 <= next_zone < demand.shape[0] and demand[next_zone] > best_demand:
                    best_demand = demand[next_zone]
                    best_action = action
            actions[slot] = best_action
        return actions
