from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .constants import (
    ACTION_ACCEPT_ORDER,
    ACTION_DIM,
    ACTION_GO_CHARGE,
    ACTION_MOVE_EAST,
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_WEST,
    ACTION_STAY,
    COST_NAMES,
    LOCAL_OBS_DIM,
    MODE_CHARGING,
    MODE_IDLE,
    MODE_REPOSITIONING,
    MODE_SERVING,
)
from .data import PreparedDataset
from .utils import as_float32

MOVE_ACTIONS = (
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_EAST,
    ACTION_MOVE_WEST,
)
MOVE_TO_INDEX = {
    ACTION_MOVE_NORTH: 0,
    ACTION_MOVE_SOUTH: 1,
    ACTION_MOVE_EAST: 2,
    ACTION_MOVE_WEST: 3,
}


@dataclass(slots=True)
class VehicleState:
    vehicle_id: int
    zone: int
    battery_kwh: float
    mode: int = MODE_IDLE
    remaining_steps: int = 0
    destination_zone: int | None = None
    idle_steps: int = 0
    requested_charge: bool = False
    assigned_station: int | None = None
    alive: bool = True


class CometTaxiEnv:
    def __init__(
        self,
        dataset: PreparedDataset,
        config: ExperimentConfig,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.base_seed = seed if seed is not None else config.train.seed
        self.rng = np.random.default_rng(self.base_seed)
        self.cell_count = int(self.dataset.metadata["data"]["cell_count"])
        self.episode_steps = int(self.dataset.metadata["data"]["episode_steps"])
        self.time_bins_per_day = int(self.dataset.metadata["data"].get("time_bins_per_day", 144))
        self.start_bin = int(self.dataset.metadata["data"]["start_hour"] * 60 // self.config.data.step_minutes)
        self.charge_stations = [int(zone) for zone in self.dataset.metadata["charge_stations"]]
        self.charge_station_count = len(self.charge_stations)
        self.zone_demand_priors = np.asarray(
            self.dataset.metadata["zone_demand_priors"], dtype=np.float32
        )
        self.global_mean_trip_minutes = float(
            self.dataset.metadata["global_defaults"]["mean_trip_minutes"]
        )
        self.global_mean_fare = float(self.dataset.metadata["global_defaults"]["mean_fare"])
        self.zone_neighbors = self._load_zone_neighbors()
        self.next_hop_matrix, self.distance_matrix = self._build_shortest_hops()
        self.nearest_charge_station, self.next_hop_to_station = self._build_charger_routes()
        self._split_index = self._build_split_index()
        self.current_split = "train"
        self.current_day = ""
        self.current_step = 0
        self.current_bin_index = self.start_bin
        self.current_demand_vector = np.zeros(self.cell_count, dtype=np.float32)
        self.current_charge_price = self.config.env.default_charge_price
        self.current_demand_scale = 1.0
        self.current_travel_noise = 1.0
        self.current_charge_price_scale = 1.0
        self.current_event_scale = 1.0
        self.current_charger_capacity_scale = 1.0
        self.current_delay_summary = 0.0
        self.current_peak_shock = 1.0
        self.current_orders_available = 0.0
        self.remaining_orders: dict[int, list[dict[str, float]]] = {}
        self.vehicles: list[VehicleState] = []
        self.slot_to_vehicle: list[int] = []
        self.agent_mask = np.zeros(self.config.env.nmax, dtype=np.float32)
        self.metrics: dict[str, float] = {}
        self.last_info: dict[str, Any] = {}
        self.pending_runtime_feedback = {
            "fallback_count": 0.0,
            "uncertainty_count": 0.0,
            "planner_steps": 0.0,
        }
        self.done = False
        self.history_len = self.config.temporal.history_len
        self.temporal_history = np.zeros(
            (
                self.history_len,
                self.cell_count + 1 + self.charge_station_count * 2 + 2,
            ),
            dtype=np.float32,
        )
        self.charger_occupancy = np.zeros(self.charge_station_count, dtype=np.float32)
        self.charger_queue = np.zeros(self.charge_station_count, dtype=np.float32)

    def _load_zone_neighbors(self) -> np.ndarray:
        raw_neighbors = self.dataset.metadata.get("mapping", {}).get("zone_neighbors", {})
        matrix = np.zeros((self.cell_count, 4), dtype=np.int64)
        for zone in range(self.cell_count):
            values = raw_neighbors.get(str(zone), [])
            cleaned = [
                int(value)
                for value in values
                if 0 <= int(value) < self.cell_count and int(value) != zone
            ]
            while len(cleaned) < 4:
                cleaned.append(zone)
            matrix[zone] = np.asarray(cleaned[:4], dtype=np.int64)
        return matrix

    def _build_shortest_hops(self) -> tuple[np.ndarray, np.ndarray]:
        adjacency = [
            [
                int(neighbor)
                for neighbor in neighbors
                if 0 <= int(neighbor) < self.cell_count and int(neighbor) != zone
            ]
            for zone, neighbors in enumerate(self.zone_neighbors)
        ]
        next_hop = np.full((self.cell_count, self.cell_count), -1, dtype=np.int64)
        distance = np.full((self.cell_count, self.cell_count), self.cell_count + 1, dtype=np.int64)
        for source in range(self.cell_count):
            queue: deque[int] = deque([source])
            distance[source, source] = 0
            next_hop[source, source] = source
            while queue:
                zone = queue.popleft()
                for neighbor in adjacency[zone]:
                    if distance[source, neighbor] <= distance[source, zone] + 1:
                        continue
                    distance[source, neighbor] = distance[source, zone] + 1
                    next_hop[source, neighbor] = neighbor if zone == source else next_hop[source, zone]
                    queue.append(neighbor)
            fallback_neighbors = adjacency[source] or [source]
            for target in range(self.cell_count):
                if next_hop[source, target] < 0:
                    next_hop[source, target] = fallback_neighbors[0]
                    distance[source, target] = 999
        return next_hop, distance

    def _build_charger_routes(self) -> tuple[np.ndarray, np.ndarray]:
        nearest_station = np.zeros(self.cell_count, dtype=np.int64)
        next_hop = np.zeros(self.cell_count, dtype=np.int64)
        for zone in range(self.cell_count):
            best_station = self.charge_stations[0]
            best_distance = self.distance_matrix[zone, best_station]
            for station in self.charge_stations[1:]:
                candidate_distance = self.distance_matrix[zone, station]
                if candidate_distance < best_distance:
                    best_station = station
                    best_distance = candidate_distance
            nearest_station[zone] = best_station
            next_hop[zone] = self.next_hop_matrix[zone, best_station]
        return nearest_station, next_hop

    def _build_split_index(self) -> dict[str, dict[str, dict[int, pd.DataFrame]]]:
        index: dict[str, dict[str, dict[int, pd.DataFrame]]] = {}
        for split, frame in self.dataset.splits.items():
            split_days: dict[str, dict[int, pd.DataFrame]] = {}
            for day, day_frame in frame.groupby("service_date"):
                split_days[str(day)] = {
                    int(time_bin): time_frame.reset_index(drop=True)
                    for time_bin, time_frame in day_frame.groupby("time_bin_index")
                }
            index[split] = split_days
        return index

    def _sample_day(self, split: str) -> str:
        available_days = sorted(self._split_index[split].keys())
        if split == "train":
            return str(self.rng.choice(available_days))
        return available_days[0]

    def _sample_vehicle_count(self, scenario: dict[str, float] | None = None) -> int:
        if scenario and "active_agents" in scenario:
            return int(scenario["active_agents"])
        return int(
            self.rng.integers(
                self.config.env.min_active_agents,
                self.config.env.max_active_agents + 1,
            )
        )

    def _build_scenario(self, split: str, overrides: dict[str, float] | None = None) -> dict[str, float]:
        overrides = overrides or {}
        dr = self.config.domain_randomization
        enabled = split == "train" and dr.enabled
        if split != "train" and dr.apply_to_eval:
            enabled = True
        if not enabled:
            scenario = {
                "demand_scale": 1.0,
                "travel_noise": 1.0,
                "charge_price_scale": 1.0,
                "charger_capacity_scale": 1.0,
                "peak_shock_scale": 1.0,
                "event_scale": 1.0,
            }
        else:
            peak_scale = 1.0
            if self.rng.random() < dr.peak_shock_probability:
                peak_scale = float(self.rng.uniform(dr.peak_shock_scale_min, dr.peak_shock_scale_max))
            event_scale = 1.0
            if self.rng.random() < dr.event_day_probability:
                event_scale = float(self.rng.uniform(dr.event_day_scale_min, dr.event_day_scale_max))
            scenario = {
                "demand_scale": float(self.rng.uniform(dr.demand_scale_min, dr.demand_scale_max)),
                "travel_noise": float(self.rng.uniform(dr.travel_time_noise_min, dr.travel_time_noise_max)),
                "charge_price_scale": float(self.rng.uniform(dr.charge_price_scale_min, dr.charge_price_scale_max)),
                "charger_capacity_scale": float(self.rng.uniform(dr.charger_capacity_scale_min, dr.charger_capacity_scale_max)),
                "peak_shock_scale": peak_scale,
                "event_scale": event_scale,
            }
        scenario.update(overrides)
        return scenario

    def _apply_scenario(self, scenario: dict[str, float]) -> None:
        self.current_demand_scale = float(scenario.get("demand_scale", 1.0)) * float(
            scenario.get("peak_shock_scale", 1.0)
        ) * float(scenario.get("event_scale", 1.0))
        self.current_travel_noise = float(scenario.get("travel_noise", 1.0))
        self.current_charge_price_scale = float(scenario.get("charge_price_scale", 1.0))
        self.current_charger_capacity_scale = float(scenario.get("charger_capacity_scale", 1.0))
        self.current_peak_shock = float(scenario.get("peak_shock_scale", 1.0))
        self.current_event_scale = float(scenario.get("event_scale", 1.0))

    def register_runtime_feedback(
        self,
        fallback_count: int = 0,
        uncertainty_count: int = 0,
        planner_steps: int = 0,
    ) -> None:
        self.pending_runtime_feedback = {
            "fallback_count": float(fallback_count),
            "uncertainty_count": float(uncertainty_count),
            "planner_steps": float(planner_steps),
        }

    def _sample_zone(self) -> int:
        return int(self.rng.choice(np.arange(self.cell_count), p=self.zone_demand_priors))

    def _sample_battery_kwh(self) -> float:
        minimum = self.config.env.init_battery_min_ratio * self.config.env.battery_capacity_kwh
        maximum = self.config.env.init_battery_max_ratio * self.config.env.battery_capacity_kwh
        return float(self.rng.uniform(minimum, maximum))

    def _default_charge_price_for_bin(self, bin_index: int) -> float:
        fraction = float(bin_index / max(self.time_bins_per_day, 1))
        price = 0.92 + 0.10 * np.sin(2.0 * np.pi * (fraction - 0.15))
        if 42 <= bin_index <= 60 or 96 <= bin_index <= 114:
            price += 0.12
        return float(price)

    def reset(
        self,
        split: str = "train",
        day: str | None = None,
        scenario: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        self.current_split = split
        self.current_day = day or self._sample_day(split)
        self.current_step = 0
        self.current_bin_index = self.start_bin
        effective_scenario = self._build_scenario(split, overrides=scenario)
        self._apply_scenario(effective_scenario)
        active_count = self._sample_vehicle_count(effective_scenario)
        self.vehicles = [
            VehicleState(
                vehicle_id=vehicle_id,
                zone=self._sample_zone(),
                battery_kwh=self._sample_battery_kwh(),
            )
            for vehicle_id in range(active_count)
        ]
        self.done = False
        self.charger_occupancy.fill(0.0)
        self.charger_queue.fill(0.0)
        self.temporal_history.fill(0.0)
        self.pending_runtime_feedback = {
            "fallback_count": 0.0,
            "uncertainty_count": 0.0,
            "planner_steps": 0.0,
        }
        self.metrics = {
            "orders_available": 0.0,
            "orders_served": 0.0,
            "profit_total": 0.0,
            "empty_moves": 0.0,
            "low_battery_violations": 0.0,
            "dead_vehicles": 0.0,
            "charging_steps": 0.0,
            "charge_need_steps": 0.0,
            "charge_response_steps": 0.0,
            "real_agent_steps": 0.0,
            "team_reward_sum": 0.0,
            "episode_steps": 0.0,
            "battery_violation_cost_sum": 0.0,
            "charger_overflow_cost_sum": 0.0,
            "service_violation_cost_sum": 0.0,
            "uncertainty_count": 0.0,
            "fallback_count": 0.0,
            "planner_steps": 0.0,
            "initial_agents": float(active_count),
        }
        self._refresh_orders()
        self._update_temporal_history()
        observation = self._build_observation()
        self.last_info = {
            "costs": {name: 0.0 for name in COST_NAMES},
            "aux_targets": self._build_aux_targets(done=False),
        }
        return observation
    def _resample_order_rows(self, rows: pd.DataFrame, scale: float) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        target_count = int(round(len(rows) * scale))
        if target_count <= 0:
            return rows.iloc[0:0].copy().reset_index(drop=True)
        indices = self.rng.choice(
            len(rows),
            size=target_count,
            replace=target_count > len(rows),
        )
        return rows.iloc[np.asarray(indices, dtype=np.int64)].reset_index(drop=True)

    def _refresh_orders(self) -> None:
        rows = self._split_index[self.current_split].get(self.current_day, {}).get(
            int(self.current_bin_index), pd.DataFrame()
        )
        sampled_rows = self._resample_order_rows(rows, self.current_demand_scale) if not rows.empty else rows
        remaining_orders: dict[int, list[dict[str, float]]] = {}
        demand_vector = np.zeros(self.cell_count, dtype=np.float32)
        delay_values: list[float] = []
        for record in sampled_rows.to_dict("records"):
            pickup_cell = int(record["pickup_cell"])
            trip_minutes = float(record["trip_minutes"]) * self.current_travel_noise
            order = {
                "dropoff_cell": int(record["dropoff_cell"]),
                "trip_steps": max(1, int(np.ceil(trip_minutes / self.config.data.step_minutes))),
                "trip_minutes": trip_minutes,
                "trip_distance_km": float(record["trip_distance_km"]),
                "revenue_amount": float(record.get("revenue_amount", record.get("fare_amount", self.global_mean_fare))),
            }
            remaining_orders.setdefault(pickup_cell, []).append(order)
            demand_vector[pickup_cell] += 1.0
            delay_values.append(float(record.get("travel_time_residual", 0.0)) * self.current_travel_noise)
        self.current_orders_available = float(len(sampled_rows))
        self.metrics["orders_available"] += self.current_orders_available
        self.current_delay_summary = float(np.mean(delay_values) if delay_values else 0.0)
        if sampled_rows.empty:
            base_price = self._default_charge_price_for_bin(self.current_bin_index)
        else:
            base_price = float(sampled_rows["charge_price"].mean())
        self.current_charge_price = base_price * self.current_charge_price_scale
        self.current_demand_vector = demand_vector
        self.remaining_orders = remaining_orders

    def _update_temporal_history(self) -> None:
        demand_total = max(float(self.current_demand_vector.sum()), 1.0)
        normalized_demand = self.current_demand_vector / demand_total
        row = np.concatenate(
            [
                normalized_demand,
                np.asarray([self.current_charge_price], dtype=np.float32),
                self.charger_occupancy / max(float(self.config.env.charger_capacity), 1.0),
                self.charger_queue / max(float(self.config.env.max_queue_length), 1.0),
                np.asarray(
                    [
                        self.current_delay_summary,
                        self.current_travel_noise - 1.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        self.temporal_history = np.roll(self.temporal_history, shift=-1, axis=0)
        self.temporal_history[-1] = row

    def _mode_one_hot(self, mode: int) -> np.ndarray:
        vector = np.zeros(4, dtype=np.float32)
        vector[mode] = 1.0
        return vector

    def _battery_ratio(self, battery_kwh: float) -> float:
        return float(np.clip(battery_kwh / max(self.config.env.battery_capacity_kwh, 1e-6), 0.0, 1.0))

    def _battery_bucket(self, battery_kwh: float) -> int:
        ratio = self._battery_ratio(battery_kwh)
        if ratio < self.config.env.battery_low_threshold:
            return 0
        if ratio < self.config.env.battery_mid_threshold:
            return 1
        return 2

    def _battery_one_hot(self, battery_kwh: float) -> np.ndarray:
        bucket = self._battery_bucket(battery_kwh)
        vector = np.zeros(3, dtype=np.float32)
        vector[bucket] = 1.0
        return vector

    def _move_zone(self, zone: int, action: int) -> int:
        if action not in MOVE_TO_INDEX:
            return zone
        return int(self.zone_neighbors[zone, MOVE_TO_INDEX[action]])

    def _nearest_charge_station(self, zone: int) -> int:
        return int(self.nearest_charge_station[zone])

    def _step_toward_zone(self, zone: int, target_zone: int) -> int:
        if target_zone == zone:
            return zone
        candidate = int(self.next_hop_matrix[zone, target_zone])
        if candidate == zone:
            neighbor_zones = self.zone_neighbors[zone]
            best_neighbor = zone
            best_demand = -1.0
            for neighbor in neighbor_zones:
                neighbor = int(neighbor)
                demand = float(self.current_demand_vector[neighbor])
                if demand > best_demand:
                    best_neighbor = neighbor
                    best_demand = demand
            return best_neighbor
        return candidate

    def _take_order(self, pickup_zone: int) -> dict[str, float] | None:
        options = self.remaining_orders.get(pickup_zone, [])
        if not options:
            return None
        order = options.pop(0)
        if not options:
            self.remaining_orders.pop(pickup_zone, None)
        self.current_demand_vector[pickup_zone] = max(0.0, self.current_demand_vector[pickup_zone] - 1.0)
        return order

    def _action_mask_for_vehicle(self, vehicle: VehicleState) -> np.ndarray:
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        if not vehicle.alive:
            return mask
        if vehicle.mode in (MODE_SERVING, MODE_REPOSITIONING, MODE_CHARGING):
            mask[ACTION_STAY] = 1.0
            return mask
        mask[ACTION_STAY] = 1.0
        if self.current_demand_vector[vehicle.zone] > 0:
            mask[ACTION_ACCEPT_ORDER] = 1.0
        mask[ACTION_GO_CHARGE] = 1.0
        for action in MOVE_ACTIONS:
            next_zone = self._move_zone(vehicle.zone, action)
            if next_zone != vehicle.zone:
                mask[action] = 1.0
        return mask

    def _shuffle_slots(self) -> None:
        indices = np.arange(len(self.vehicles))
        if self.current_split == "train":
            indices = self.rng.permutation(indices)
        self.slot_to_vehicle = indices.tolist()

    def _build_fleet_signature(self) -> np.ndarray:
        histogram = np.zeros((self.cell_count, 4, 3), dtype=np.float32)
        for vehicle in self.vehicles:
            if not vehicle.alive:
                continue
            histogram[vehicle.zone, vehicle.mode, self._battery_bucket(vehicle.battery_kwh)] += 1.0
        fleet_hist = histogram.reshape(-1)
        if len(self.vehicles) > 0:
            fleet_hist = fleet_hist / float(len(self.vehicles))
        demand_total = max(float(self.current_demand_vector.sum()), 1.0)
        demand_vector = self.current_demand_vector / demand_total
        progress = self.current_step / max(self.episode_steps - 1, 1)
        scalars = np.asarray(
            [
                len(self.vehicles) / float(self.config.env.nmax),
                self.current_charge_price,
                self.current_delay_summary,
                np.sin(progress * 2.0 * np.pi),
                np.cos(progress * 2.0 * np.pi),
            ],
            dtype=np.float32,
        )
        signature = np.concatenate(
            [
                fleet_hist,
                demand_vector.astype(np.float32),
                self.charger_occupancy.astype(np.float32),
                self.charger_queue.astype(np.float32),
                scalars,
            ]
        ).astype(np.float32)
        return signature

    def _build_candidate_context(
        self,
        nearest_station_zones: np.ndarray,
        move_target_zones: np.ndarray,
    ) -> dict[str, np.ndarray]:
        ranked_demand_zones = np.argsort(-self.current_demand_vector).astype(np.int64)
        return {
            "demand_vector": as_float32(self.current_demand_vector.copy()),
            "ranked_demand_zones": ranked_demand_zones,
            "charger_zones": np.asarray(self.charge_stations, dtype=np.int64),
            "charger_occupancy": as_float32(self.charger_occupancy.copy()),
            "charger_queue": as_float32(self.charger_queue.copy()),
            "nearest_charger_zone": nearest_station_zones.astype(np.int64),
            "move_target_zones": move_target_zones.astype(np.int64),
        }

    def _build_observation(self) -> dict[str, np.ndarray]:
        self._shuffle_slots()
        local_obs = np.zeros((self.config.env.nmax, LOCAL_OBS_DIM), dtype=np.float32)
        action_mask = np.zeros((self.config.env.nmax, ACTION_DIM), dtype=np.float32)
        agent_mask = np.zeros(self.config.env.nmax, dtype=np.float32)
        nearest_station_zones = np.zeros(self.config.env.nmax, dtype=np.int64)
        move_target_zones = np.zeros((self.config.env.nmax, 4), dtype=np.int64)

        for slot_index, vehicle_index in enumerate(self.slot_to_vehicle):
            vehicle = self.vehicles[vehicle_index]
            agent_mask[slot_index] = 1.0
            local_obs[slot_index, 0] = vehicle.zone + 1
            local_obs[slot_index, 1:4] = self._battery_one_hot(vehicle.battery_kwh)
            local_obs[slot_index, 4] = self._battery_ratio(vehicle.battery_kwh)
            local_obs[slot_index, 5:9] = self._mode_one_hot(vehicle.mode)
            local_obs[slot_index, 9] = vehicle.remaining_steps / max(self.episode_steps, 1)
            action_mask[slot_index] = self._action_mask_for_vehicle(vehicle)
            nearest_station_zones[slot_index] = self._nearest_charge_station(vehicle.zone)
            move_target_zones[slot_index] = self.zone_neighbors[vehicle.zone]

        self.agent_mask = agent_mask
        candidate_context = self._build_candidate_context(nearest_station_zones, move_target_zones)
        return {
            "local_obs": as_float32(local_obs),
            "fleet_signature": as_float32(self._build_fleet_signature()),
            "agent_mask": as_float32(agent_mask),
            "active_masks": as_float32(agent_mask),
            "action_mask": as_float32(action_mask),
            "temporal_history": as_float32(self.temporal_history.copy()),
            **candidate_context,
        }

    def _compose_reward(self, profit: float, efficiency_penalty: float, battery_penalty: float) -> float:
        reward_cfg = self.config.reward
        return (
            reward_cfg.profit_weight * profit
            - reward_cfg.efficiency_weight * efficiency_penalty
            - reward_cfg.battery_weight * battery_penalty
        )

    def _battery_penalty(self, vehicle: VehicleState, include_charge_price: bool = False) -> float:
        penalty = 0.0
        ratio = self._battery_ratio(vehicle.battery_kwh)
        if ratio < self.config.env.battery_low_threshold:
            penalty += self.config.env.low_battery_penalty_weight * (
                self.config.env.battery_low_threshold - ratio
            )
            self.metrics["low_battery_violations"] += 1.0
        if include_charge_price:
            penalty += 0.02 * self.current_charge_price
        return penalty

    def _kill_vehicle_if_needed(self, vehicle: VehicleState) -> float:
        if vehicle.battery_kwh >= self.config.env.battery_min_kwh:
            return 0.0
        vehicle.alive = False
        vehicle.mode = MODE_IDLE
        vehicle.remaining_steps = 0
        vehicle.destination_zone = None
        self.metrics["dead_vehicles"] += 1.0
        return self.config.env.death_penalty

    def _consume_reposition_energy(self, vehicle: VehicleState) -> None:
        vehicle.battery_kwh = max(
            0.0,
            vehicle.battery_kwh - self.config.env.reposition_distance_km * self.config.env.energy_per_km,
        )
    def _update_busy_vehicle(self, vehicle: VehicleState) -> float:
        battery_penalty = 0.0
        vehicle.requested_charge = False
        if vehicle.mode == MODE_SERVING:
            vehicle.remaining_steps = max(0, vehicle.remaining_steps - 1)
            if vehicle.remaining_steps == 0:
                if vehicle.destination_zone is not None:
                    vehicle.zone = vehicle.destination_zone
                vehicle.destination_zone = None
                vehicle.mode = MODE_IDLE
        elif vehicle.mode == MODE_REPOSITIONING:
            vehicle.remaining_steps = max(0, vehicle.remaining_steps - 1)
            if vehicle.remaining_steps == 0:
                vehicle.mode = MODE_IDLE
        elif vehicle.mode == MODE_CHARGING:
            vehicle.requested_charge = True
            vehicle.assigned_station = vehicle.zone
        battery_penalty += self._battery_penalty(vehicle)
        death_penalty = self._kill_vehicle_if_needed(vehicle)
        return self._compose_reward(0.0, 0.0, battery_penalty) - death_penalty

    def _step_idle_vehicle(self, vehicle: VehicleState, action: int) -> float:
        profit = 0.0
        efficiency_penalty = 0.0
        battery_penalty = 0.0
        vehicle.requested_charge = False

        if self._battery_ratio(vehicle.battery_kwh) < self.config.env.battery_mid_threshold:
            self.metrics["charge_need_steps"] += 1.0
            if action == ACTION_GO_CHARGE or vehicle.mode == MODE_CHARGING:
                self.metrics["charge_response_steps"] += 1.0

        if action == ACTION_ACCEPT_ORDER:
            order = self._take_order(vehicle.zone)
            if order is not None:
                vehicle.mode = MODE_SERVING
                vehicle.destination_zone = int(order["dropoff_cell"])
                vehicle.remaining_steps = int(order["trip_steps"])
                vehicle.idle_steps = 0
                profit += float(order["revenue_amount"])
                vehicle.battery_kwh = max(
                    0.0,
                    vehicle.battery_kwh
                    - float(order["trip_distance_km"]) * self.config.env.energy_per_km,
                )
                self.metrics["orders_served"] += 1.0
                self.metrics["profit_total"] += float(order["revenue_amount"])
            else:
                efficiency_penalty += self.config.env.idle_penalty
                vehicle.idle_steps += 1
        elif action == ACTION_GO_CHARGE:
            target_station = self._nearest_charge_station(vehicle.zone)
            if vehicle.zone == target_station:
                vehicle.mode = MODE_CHARGING
                vehicle.requested_charge = True
                vehicle.assigned_station = target_station
                battery_penalty += 0.02 * self.current_charge_price
            else:
                next_zone = int(self.next_hop_to_station[vehicle.zone])
                if next_zone != vehicle.zone:
                    vehicle.zone = next_zone
                    self.metrics["empty_moves"] += 1.0
                vehicle.mode = MODE_REPOSITIONING
                vehicle.remaining_steps = 1
                efficiency_penalty += self.config.env.empty_move_penalty
                self._consume_reposition_energy(vehicle)
                vehicle.assigned_station = target_station
        elif action in MOVE_ACTIONS:
            next_zone = self._move_zone(vehicle.zone, action)
            if next_zone != vehicle.zone:
                vehicle.zone = next_zone
                self.metrics["empty_moves"] += 1.0
            vehicle.mode = MODE_REPOSITIONING
            vehicle.remaining_steps = 1
            self._consume_reposition_energy(vehicle)
            efficiency_penalty += self.config.env.empty_move_penalty
        else:
            vehicle.idle_steps += 1
            efficiency_penalty += self.config.env.idle_penalty

        battery_penalty += self._battery_penalty(vehicle)
        death_penalty = self._kill_vehicle_if_needed(vehicle)
        return self._compose_reward(profit, efficiency_penalty, battery_penalty) - death_penalty

    def _effective_charger_capacity(self) -> int:
        scaled = int(round(self.config.env.charger_capacity * self.current_charger_capacity_scale))
        return max(1, scaled)

    def _apply_charger_allocation(self) -> float:
        capacity = self._effective_charger_capacity()
        overflow_cost = 0.0
        self.charger_occupancy.fill(0.0)
        self.charger_queue.fill(0.0)

        station_to_vehicles: dict[int, list[VehicleState]] = {zone: [] for zone in self.charge_stations}
        for vehicle in self.vehicles:
            if vehicle.alive and vehicle.requested_charge and vehicle.zone in station_to_vehicles:
                station_to_vehicles[vehicle.zone].append(vehicle)

        charge_gain = self.config.env.charge_power_kw * (self.config.data.step_minutes / 60.0)
        for station_index, station_zone in enumerate(self.charge_stations):
            candidates = station_to_vehicles.get(station_zone, [])
            occupancy = min(len(candidates), capacity)
            queue = max(len(candidates) - capacity, 0)
            self.charger_occupancy[station_index] = float(occupancy)
            self.charger_queue[station_index] = float(queue)
            overflow_cost += float(queue) / max(float(capacity), 1.0)
            for candidate_index, vehicle in enumerate(candidates):
                if candidate_index < capacity:
                    vehicle.mode = MODE_CHARGING
                    vehicle.battery_kwh = min(
                        self.config.env.battery_capacity_kwh,
                        vehicle.battery_kwh + charge_gain,
                    )
                    self.metrics["charging_steps"] += 1.0
                    if self._battery_ratio(vehicle.battery_kwh) >= 0.95:
                        vehicle.mode = MODE_IDLE
                else:
                    vehicle.mode = MODE_IDLE
                    vehicle.idle_steps += 1
        return overflow_cost

    def _remove_dead_vehicles(self) -> None:
        self.vehicles = [vehicle for vehicle in self.vehicles if vehicle.alive]

    def _build_costs(self) -> dict[str, float]:
        active_agents = max(float(len(self.vehicles)), 1.0)
        battery_cost = float(
            sum(
                self._battery_ratio(vehicle.battery_kwh) < self.config.env.battery_low_threshold
                for vehicle in self.vehicles
            )
            + self.metrics["dead_vehicles"]
        ) / max(active_agents + self.metrics["dead_vehicles"], 1.0)
        charger_cost = float(self.charger_queue.sum()) / max(
            float(self._effective_charger_capacity() * max(len(self.charge_stations), 1)),
            1.0,
        )
        service_cost = float(self.current_demand_vector.sum()) / max(self.current_orders_available, 1.0)
        costs = {
            "battery_violation_cost": battery_cost,
            "charger_overflow_cost": charger_cost,
            "service_violation_cost": service_cost,
        }
        self.metrics["battery_violation_cost_sum"] += costs["battery_violation_cost"]
        self.metrics["charger_overflow_cost_sum"] += costs["charger_overflow_cost"]
        self.metrics["service_violation_cost_sum"] += costs["service_violation_cost"]
        return costs

    def _build_aux_targets(self, done: bool) -> dict[str, np.ndarray | float]:
        demand_total = max(float(self.current_demand_vector.sum()), 1.0)
        next_demand = np.zeros_like(self.current_demand_vector) if done else self.current_demand_vector / demand_total
        return {
            "next_demand": as_float32(next_demand),
            "charger_occupancy": as_float32(
                self.charger_occupancy / max(float(self._effective_charger_capacity()), 1.0)
            ),
            "travel_time_residual": float(self.current_delay_summary),
        }

    def generate_candidate_actions_for_slot(
        self,
        observation: dict[str, np.ndarray],
        slot: int,
        top_k: int,
    ) -> list[int]:
        if observation["agent_mask"][slot] <= 0:
            return []
        candidates = {ACTION_STAY}
        action_mask = observation["action_mask"][slot]
        if action_mask[ACTION_ACCEPT_ORDER] > 0:
            candidates.add(ACTION_ACCEPT_ORDER)
        if action_mask[ACTION_GO_CHARGE] > 0:
            candidates.add(ACTION_GO_CHARGE)
        zone = int(observation["local_obs"][slot, 0]) - 1
        for action in MOVE_ACTIONS:
            if action_mask[action] > 0:
                candidates.add(action)
        ranked_zones = observation["ranked_demand_zones"][:top_k]
        for target_zone in ranked_zones:
            target_zone = int(target_zone)
            if target_zone == zone:
                candidates.add(ACTION_STAY)
                continue
            next_zone = self._step_toward_zone(zone, target_zone)
            for action in MOVE_ACTIONS:
                if action_mask[action] > 0 and self._move_zone(zone, action) == next_zone:
                    candidates.add(action)
                    break
        return sorted(candidates)

    def generate_candidate_actions(
        self,
        observation: dict[str, np.ndarray],
        top_k: int,
    ) -> list[list[int]]:
        return [
            self.generate_candidate_actions_for_slot(observation, slot, top_k)
            for slot in range(self.config.env.nmax)
        ]

    def step(
        self, actions: np.ndarray | list[int]
    ) -> tuple[dict[str, np.ndarray], np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is finished. Call reset() before step().")

        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.config.env.nmax,):
            raise ValueError(
                f"Expected actions to have shape ({self.config.env.nmax},), got {action_array.shape}."
            )

        per_agent_rewards = np.zeros(self.config.env.nmax, dtype=np.float32)
        for slot_index, vehicle_index in enumerate(self.slot_to_vehicle):
            if vehicle_index >= len(self.vehicles):
                continue
            vehicle = self.vehicles[vehicle_index]
            valid_mask = self._action_mask_for_vehicle(vehicle)
            chosen_action = int(action_array[slot_index])
            if valid_mask[chosen_action] <= 0:
                chosen_action = ACTION_STAY
            reward = (
                self._update_busy_vehicle(vehicle)
                if vehicle.mode in (MODE_SERVING, MODE_REPOSITIONING, MODE_CHARGING)
                else self._step_idle_vehicle(vehicle, chosen_action)
            )
            per_agent_rewards[slot_index] = reward

        overflow_penalty = self._apply_charger_allocation()
        self._remove_dead_vehicles()
        costs = self._build_costs()
        costs["charger_overflow_cost"] += overflow_penalty
        self.metrics["charger_overflow_cost_sum"] += overflow_penalty

        active_rewards = per_agent_rewards[self.agent_mask > 0]
        team_reward = float(active_rewards.mean()) if active_rewards.size > 0 else 0.0
        self.metrics["team_reward_sum"] += team_reward
        self.metrics["real_agent_steps"] += float(self.agent_mask.sum())
        self.metrics["episode_steps"] += 1.0
        self.metrics["fallback_count"] += self.pending_runtime_feedback["fallback_count"]
        self.metrics["uncertainty_count"] += self.pending_runtime_feedback["uncertainty_count"]
        self.metrics["planner_steps"] += self.pending_runtime_feedback["planner_steps"]
        self.pending_runtime_feedback = {
            "fallback_count": 0.0,
            "uncertainty_count": 0.0,
            "planner_steps": 0.0,
        }

        self.current_step += 1
        self.current_bin_index = self.start_bin + self.current_step
        self.done = self.current_step >= self.episode_steps or len(self.vehicles) == 0

        if self.done:
            self.current_demand_vector = np.zeros(self.cell_count, dtype=np.float32)
            self.current_orders_available = 0.0
            self._update_temporal_history()
            next_observation = self._build_observation()
        else:
            self._refresh_orders()
            self._update_temporal_history()
            next_observation = self._build_observation()

        info = self._build_info(costs)
        info["aux_targets"] = self._build_aux_targets(done=self.done)
        self.last_info = info
        return next_observation, per_agent_rewards, team_reward, self.done, info

    def _build_info(self, costs: dict[str, float]) -> dict[str, Any]:
        orders_available = max(self.metrics["orders_available"], 1.0)
        initial_vehicle_count = max(float(self.metrics["initial_agents"]), 1.0)
        charge_need_steps = max(self.metrics["charge_need_steps"], 1.0)
        real_agent_steps = max(self.metrics["real_agent_steps"], 1.0)
        planner_steps = max(self.metrics["planner_steps"], 1.0)
        battery_violation_events = self.metrics["low_battery_violations"] + self.metrics["dead_vehicles"]
        return {
            "order_completion_rate": self.metrics["orders_served"] / orders_available,
            "average_profit_per_vehicle": self.metrics["profit_total"] / initial_vehicle_count,
            "empty_travel_ratio": self.metrics["empty_moves"] / max(
                self.metrics["empty_moves"] + self.metrics["orders_served"], 1.0
            ),
            "battery_safety_rate": 1.0 - battery_violation_events / real_agent_steps,
            "charging_efficiency": self.metrics["charge_response_steps"] / charge_need_steps,
            "mean_team_reward": self.metrics["team_reward_sum"] / max(self.metrics["episode_steps"], 1.0),
            "orders_available": self.metrics["orders_available"],
            "orders_served": self.metrics["orders_served"],
            "battery_violation_rate": self.metrics["battery_violation_cost_sum"] / max(self.metrics["episode_steps"], 1.0),
            "charger_overflow_rate": self.metrics["charger_overflow_cost_sum"] / max(self.metrics["episode_steps"], 1.0),
            "service_violation_rate": self.metrics["service_violation_cost_sum"] / max(self.metrics["episode_steps"], 1.0),
            "fallback_rate": self.metrics["fallback_count"] / planner_steps,
            "uncertainty_trigger_rate": self.metrics["uncertainty_count"] / planner_steps,
            "costs": costs,
        }
