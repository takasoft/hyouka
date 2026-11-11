"""Pipeline orchestration — top-level Floe entry point."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from floe.catalog import CatalogManager
from floe.config import FloeConfig
from floe.engine import Executor, make_executor
from floe.models import DynamicTable, RefreshResult
from floe.parser import load_dits
from floe.planner import DAGPlanner
from floe.sources import SourceLoader, SourceLoadResult


class Pipeline:
    """A Floe pipeline — config + DITs + planner + executor."""

    def __init__(
        self,
        config: FloeConfig,
        dits: list[DynamicTable],
        catalog_mgr: CatalogManager,
    ):
        self.config = config
        self.dits = {d.name: d for d in dits}
        self.planner = DAGPlanner(dits)
        self.catalog_mgr = catalog_mgr
        self.executor: Executor = make_executor(config, catalog_mgr, self.dits)
        self.sources = list(config.sources)
        for src in self.sources:
            for name in filter(None, (src.table, src.quarantine)):
                if name in self.dits:
                    raise ValueError(
                        f"{name!r} is both a source table and a dynamic table; "
                        "a table must have exactly one writer"
                    )

    @classmethod
    def from_config(cls, config_path: str | Path) -> Pipeline:
        config = FloeConfig.load(config_path)
        dits = load_dits(config.resolved_transformations_dir())
        catalog_mgr = CatalogManager.from_config(
            name=config.project,
            catalog_type=config.catalog.type,
            uri=config.resolved_catalog_uri(),
            warehouse=config.resolved_warehouse(),
            properties=config.catalog.properties,
        )
        return cls(config, dits, catalog_mgr)

    def load_sources(
        self,
        *,
        now: datetime | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> list[SourceLoadResult]:
        """Load new files for every declared source into its bronze table."""
        loader = SourceLoader(self.catalog_mgr, self.config.project_root, progress=progress)
        return [loader.load(src, now=now) for src in self.sources]

    def run(
        self, *, progress: Callable[[str], None] | None = None
    ) -> tuple[list[SourceLoadResult], list[RefreshResult]]:
        """Load sources, then refresh every DIT: the whole pipeline in one call.

        DITs whose upstream snapshots did not change are skipped as usual, so a
        run with no new source files is cheap.
        """
        loads = self.load_sources(progress=progress)
        return loads, self.refresh_all()

    def refresh_all(
        self, *, on_result: Callable[[RefreshResult], None] | None = None
    ) -> list[RefreshResult]:
        """Refresh every DIT in topological order."""
        results: list[RefreshResult] = []
        for name in self.planner.topological_order():
            dit = self.dits[name]
            results.append(self.executor.refresh(dit))
            if on_result:
                on_result(results[-1])
        return results

    def refresh_one(self, name: str) -> RefreshResult:
        if name not in self.dits:
            raise KeyError(f"No DIT named {name!r}")
        return self.executor.refresh(self.dits[name])

    def status(self) -> list[dict]:
        rows = []
        for name in self.planner.topological_order():
            dit = self.dits[name]
            exists = self.catalog_mgr.table_exists(name)
            rows.append(
                {
                    "name": name,
                    "exists": exists,
                    "refresh_mode": dit.refresh_mode.value,
                    "lag": dit.lag,
                    "current_snapshot": self.catalog_mgr.current_snapshot_id(name)
                    if exists
                    else None,
                    "checkpoint": self.catalog_mgr.get_checkpoint(name) if exists else None,
                    "upstream": dit.upstream_tables,
                }
            )
        return rows
