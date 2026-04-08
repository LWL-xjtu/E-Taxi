from __future__ import annotations

from pathlib import Path


def test_readme_mentions_formal_generalist_workflow() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for snippet in (
        "configs/formal_generalist.toml",
        "configs/generalization_eval.toml",
        "data/raw/yellow_tripdata_2025-12.parquet",
        "data/raw/yellow_tripdata_2026-01.parquet",
        "data/raw/yellow_tripdata_2026-02.parquet",
        "outputs/formal_generalist_run",
        "outputs/formal_generalist_eval",
        "--resume-checkpoint",
    ):
        assert snippet in readme
