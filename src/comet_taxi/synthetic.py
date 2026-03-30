from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_synthetic_tlc_frame(
    seed: int = 0,
    start: str = "2023-01-01 08:00:00",
    days: int = 7,
    rows_per_step: int = 12,
    step_minutes: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    records: list[dict[str, float | str]] = []

    for day in range(days):
        day_start = start_ts + pd.Timedelta(days=day)
        for step in range((12 * 60) // step_minutes):
            current = day_start + pd.Timedelta(minutes=step * step_minutes)
            hour_profile = 1.0 + 0.3 * np.sin(step / 5.0)
            row_count = max(4, int(rows_per_step * hour_profile))
            for _ in range(row_count):
                pickup_lon = -74.05 + rng.random() * 0.25
                pickup_lat = 40.60 + rng.random() * 0.25
                dropoff_lon = -74.05 + rng.random() * 0.25
                dropoff_lat = 40.60 + rng.random() * 0.25
                trip_minutes = int(rng.integers(8, 35))
                total_amount = float(rng.uniform(8.0, 42.0))
                records.append(
                    {
                        "tpep_pickup_datetime": current,
                        "tpep_dropoff_datetime": current + pd.Timedelta(minutes=trip_minutes),
                        "pickup_longitude": pickup_lon,
                        "pickup_latitude": pickup_lat,
                        "dropoff_longitude": dropoff_lon,
                        "dropoff_latitude": dropoff_lat,
                        "total_amount": total_amount,
                    }
                )

    return pd.DataFrame.from_records(records)


def write_synthetic_parquet(path: str | Path, **kwargs: int | str) -> Path:
    target = Path(path)
    frame = build_synthetic_tlc_frame(**kwargs)
    frame.to_parquet(target, index=False)
    return target
