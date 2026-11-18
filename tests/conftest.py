"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from hyouka.catalog import CatalogManager
from hyouka.config import CatalogConfig, ComputeConfig, DefaultsConfig, HyoukaConfig


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / "transformations").mkdir()
    (tmp_path / "warehouse").mkdir()
    (tmp_path / ".hyouka").mkdir()
    return tmp_path


@pytest.fixture
def hyouka_config(project_dir: Path) -> HyoukaConfig:
    cfg = HyoukaConfig(
        project="test_project",
        version="1.0",
        catalog=CatalogConfig(
            type="sql",
            uri=f"sqlite:///{(project_dir / '.hyouka' / 'catalog.db').as_posix()}",
            warehouse=str(project_dir / "warehouse"),
        ),
        compute=ComputeConfig(),
        defaults=DefaultsConfig(),
        transformations_dir=str(project_dir / "transformations"),
    )
    cfg.project_root = project_dir
    return cfg


@pytest.fixture
def catalog_mgr(hyouka_config: HyoukaConfig) -> CatalogManager:
    return CatalogManager.from_config(
        name=hyouka_config.project,
        catalog_type=hyouka_config.catalog.type,
        uri=hyouka_config.resolved_catalog_uri(),
        warehouse=hyouka_config.resolved_warehouse(),
    )


@pytest.fixture
def sample_orders() -> pa.Table:
    return pa.table(
        {
            "order_id": pa.array([1, 2, 3], type=pa.int64()),
            "customer_id": pa.array([100, 101, 100], type=pa.int64()),
            "amount": pa.array([10.5, 25.0, 7.25], type=pa.float64()),
        }
    )


@pytest.fixture
def sample_customers() -> pa.Table:
    return pa.table(
        {
            "id": pa.array([100, 101], type=pa.int64()),
            "region": pa.array(["us-east", "us-west"], type=pa.string()),
        }
    )
