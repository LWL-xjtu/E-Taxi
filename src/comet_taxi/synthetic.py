from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_synthetic_tlc_frame(
    seed: int = 0,
    start: str = "2026-01-01 00:00:00",
    days: int = 7,
    rows_per_step: int = 12,
    step_minutes: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    records: list[dict[str, float | int | str | pd.Timestamp]] = []
    bins_per_day = (24 * 60) // step_minutes
    hubs = np.asarray([48, 79, 138, 161, 186, 230, 239], dtype=np.int64)

    for day in range(days):
        day_start = start_ts + pd.Timedelta(days=day)
        for step in range(bins_per_day):
            current = day_start + pd.Timedelta(minutes=step * step_minutes)
            time_fraction = step / max(bins_per_day, 1)
            peak_multiplier = 1.0 + 0.6 * np.sin(2.0 * np.pi * (time_fraction - 0.2)) ** 2
            row_count = max(4, int(rows_per_step * peak_multiplier))
            for _ in range(row_count):
                if rng.random() < 0.7:
                    pickup_zone = int(rng.choice(hubs))
                else:
                    pickup_zone = int(rng.integers(1, 264))
                if rng.random() < 0.65:
                    dropoff_zone = int(rng.choice(hubs))
                else:
                    dropoff_zone = int(rng.integers(1, 264))
                if dropoff_zone == pickup_zone:
                    dropoff_zone = 1 + (dropoff_zone % 263)
                trip_distance = float(rng.uniform(0.6, 7.0))
                trip_minutes = int(rng.integers(5, 40))
                fare_amount = float(max(4.0, 3.5 + trip_distance * 2.8 + rng.normal(0.0, 1.0)))
                tip_amount = max(0.0, float(rng.normal(1.2, 0.8)))
                total_amount = fare_amount + 1.5 + tip_amount
                records.append(
                    {
                        "VendorID": int(rng.integers(1, 3)),
                        "tpep_pickup_datetime": current,
                        "tpep_dropoff_datetime": current + pd.Timedelta(minutes=trip_minutes),
                        "passenger_count": float(rng.integers(1, 3)),
                        "trip_distance": trip_distance,
                        "RatecodeID": 1.0,
                        "store_and_fwd_flag": "N",
                        "PULocationID": pickup_zone,
                        "DOLocationID": dropoff_zone,
                        "payment_type": int(rng.integers(1, 3)),
                        "fare_amount": fare_amount,
                        "extra": 1.0,
                        "mta_tax": 0.5,
                        "tip_amount": tip_amount,
                        "tolls_amount": 0.0,
                        "improvement_surcharge": 1.0,
                        "total_amount": total_amount,
                        "congestion_surcharge": 2.5,
                        "Airport_fee": 0.0,
                        "cbd_congestion_fee": 0.0,
                    }
                )

    return pd.DataFrame.from_records(records)


def write_synthetic_parquet(path: str | Path, **kwargs: int | str) -> Path:
    target = Path(path)
    frame = build_synthetic_tlc_frame(**kwargs)
    frame.to_parquet(target, index=False)
    return target
