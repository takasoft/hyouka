"""DuckDB refresh executor — runs DIT queries via DuckDB and writes to Iceberg.

This is Floe's default compute engine. It implements the structural ``Executor``
interface (see :mod:`floe.engine`): ``refresh(dit, *, force=False) -> RefreshResult``.
An alternative ``flink`` engine implementing the same interface lives in
:mod:`floe.flink_executor`; the engine in use is selected by ``compute.engine``.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta

import duckdb
import pyarrow as pa
from pyiceberg.expressions import GreaterThanOrEqual
from pyiceberg.io.pyarrow import sanitize_column_names

from floe.catalog import CatalogManager
from floe.lineage import LINEAGE_COLUMNS, inject_lineage, new_job_run_id
from floe.models import DynamicTable, RefreshResult


class DuckDBExecutor:
    """Executes a DIT refresh: read upstream Iceberg → DuckDB SQL → write Iceberg."""

    def __init__(self, catalog_mgr: CatalogManager, all_dits: dict[str, DynamicTable]):
        self.catalog_mgr = catalog_mgr
        self.all_dits = all_dits

    def refresh(self, dit: DynamicTable, *, force: bool = False) -> RefreshResult:
        """Refresh a single DIT.

        With ``force=True`` the executor skips its "is this stale?" short-circuit
        and always re-runs the underlying query. The watcher uses this because it
        has already detected the upstream change that motivated the refresh, so
        re-checking staleness would be redundant work.
        """
        started_at = datetime.now(UTC)
        job_run_id = new_job_run_id()

        if dit.is_partitioned:
            return self._refresh_partitioned(dit, started_at, job_run_id, force=force)
        return self._refresh_unpartitioned(dit, started_at, job_run_id, force=force)

    # --- unpartitioned (table-level) refresh -------------------------------

    def _refresh_unpartitioned(
        self, dit: DynamicTable, started_at: datetime, job_run_id: str, *, force: bool = False
    ) -> RefreshResult:
        primary_upstream = self._primary_upstream(dit)
        upstream_snapshot_id = (
            self.catalog_mgr.current_snapshot_id(primary_upstream)
            if primary_upstream and self.catalog_mgr.table_exists(primary_upstream)
            else None
        )

        table_exists = self.catalog_mgr.table_exists(dit.name)
        current_upstreams = self._current_upstream_snapshots(dit)

        if not force and table_exists:
            last_upstreams = self.catalog_mgr.get_upstream_checkpoints(dit.name)
            if last_upstreams is not None and last_upstreams == current_upstreams:
                return RefreshResult(
                    table=dit.name,
                    mode=dit.refresh_mode,
                    rows_written=0,
                    input_snapshot_id=upstream_snapshot_id,
                    output_snapshot_id=self.catalog_mgr.current_snapshot_id(dit.name),
                    job_run_id=job_run_id,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    skipped=True,
                    skip_reason="upstream snapshot unchanged",
                )

        # We always recompute the full result from the current upstream snapshots
        # and replace the table contents. This is idempotent for every query shape
        # (pass-through, join, or aggregation): re-running can never duplicate or
        # double-count rows. True delta-based incremental append is deferred to the
        # Flink compute layer on the roadmap; DuckDB recompute-and-replace is the
        # correct v0.1 behaviour for both INCREMENTAL and FULL unpartitioned tables.
        result_arrow = self._execute_query(dit)
        result_with_lineage = inject_lineage(
            result_arrow,
            input_snapshot_id=upstream_snapshot_id,
            job_run_id=job_run_id,
            refresh_mode=dit.refresh_mode.value,
        )

        if not table_exists:
            self.catalog_mgr.ensure_namespace(dit.namespace)
            self.catalog_mgr.catalog.create_table(dit.name, schema=result_with_lineage.schema)

        self.catalog_mgr.overwrite(dit.name, result_with_lineage)

        self.catalog_mgr.set_upstream_checkpoints(dit.name, current_upstreams)
        if upstream_snapshot_id is not None:
            self.catalog_mgr.set_checkpoint(dit.name, upstream_snapshot_id)

        return RefreshResult(
            table=dit.name,
            mode=dit.refresh_mode,
            rows_written=result_with_lineage.num_rows,
            input_snapshot_id=upstream_snapshot_id,
            output_snapshot_id=self.catalog_mgr.current_snapshot_id(dit.name),
            job_run_id=job_run_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    # --- partitioned (window-aware) refresh --------------------------------

    def _refresh_partitioned(
        self, dit: DynamicTable, started_at: datetime, job_run_id: str, *, force: bool = False
    ) -> RefreshResult:
        partition_col = dit.partition_by[0]

        primary_upstream = self._primary_upstream(dit)
        upstream_snap = (
            self.catalog_mgr.current_snapshot_id(primary_upstream)
            if primary_upstream and self.catalog_mgr.table_exists(primary_upstream)
            else None
        )

        table_exists = self.catalog_mgr.table_exists(dit.name)
        current_upstreams = self._current_upstream_snapshots(dit)
        last_upstreams = (
            self.catalog_mgr.get_upstream_checkpoints(dit.name) if table_exists else None
        )
        last_window_refresh = (
            self.catalog_mgr.get_window_refreshed_at(dit.name) if table_exists else None
        )

        upstream_changed = last_upstreams is None or last_upstreams != current_upstreams

        window_stale = False
        freshness_secs = dit.partition_freshness_seconds()
        if freshness_secs is not None:
            if last_window_refresh is None:
                window_stale = True
            else:
                age = (datetime.now(UTC) - last_window_refresh).total_seconds()
                window_stale = age > freshness_secs

        if not force and table_exists and not upstream_changed and not window_stale:
            return RefreshResult(
                table=dit.name,
                mode=dit.refresh_mode,
                rows_written=0,
                input_snapshot_id=upstream_snap,
                output_snapshot_id=self.catalog_mgr.current_snapshot_id(dit.name),
                job_run_id=job_run_id,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                skipped=True,
                skip_reason="upstream unchanged and window not stale",
            )

        cutoff = self._compute_window_cutoff(dit)

        # Compute the (possibly windowed) result.
        result_arrow = self._execute_query(dit, window_filter=(partition_col, cutoff))
        result_with_lineage = inject_lineage(
            result_arrow,
            input_snapshot_id=upstream_snap,
            job_run_id=job_run_id,
            refresh_mode=dit.refresh_mode.value,
        )

        if not table_exists:
            self.catalog_mgr.create_partitioned_table(
                dit.name, result_with_lineage.schema, dit.partition_by
            )

        if cutoff is not None:
            self.catalog_mgr.overwrite_with_filter(
                dit.name,
                result_with_lineage,
                GreaterThanOrEqual(partition_col, cutoff.isoformat()),
            )
        else:
            self.catalog_mgr.overwrite(dit.name, result_with_lineage)

        if upstream_snap is not None:
            self.catalog_mgr.set_checkpoint(dit.name, upstream_snap)
        self.catalog_mgr.set_upstream_checkpoints(dit.name, current_upstreams)
        self.catalog_mgr.set_window_refreshed_at(dit.name, datetime.now(UTC))

        return RefreshResult(
            table=dit.name,
            mode=dit.refresh_mode,
            rows_written=result_with_lineage.num_rows,
            input_snapshot_id=upstream_snap,
            output_snapshot_id=self.catalog_mgr.current_snapshot_id(dit.name),
            job_run_id=job_run_id,
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    def _compute_window_cutoff(self, dit: DynamicTable) -> date | None:
        secs = dit.partition_window_seconds()
        if secs is None:
            return None
        return (datetime.now(UTC) - timedelta(seconds=secs)).date()

    def _current_upstream_snapshots(self, dit: DynamicTable) -> dict[str, int]:
        """Snapshot the current snapshot ID of every existing upstream table.

        Upstreams that don't exist yet, or that have no snapshot, are omitted —
        their later appearance is itself a change that flips staleness.
        """
        snaps: dict[str, int] = {}
        for upstream in dit.upstream_tables:
            if not self.catalog_mgr.table_exists(upstream):
                continue
            sid = self.catalog_mgr.current_snapshot_id(upstream)
            if sid is not None:
                snaps[upstream] = sid
        return snaps

    def _primary_upstream(self, dit: DynamicTable) -> str | None:
        if not dit.upstream_tables:
            return None
        # MVP: pick the first upstream as primary for snapshot tracking.
        return dit.upstream_tables[0]

    def _execute_query(
        self,
        dit: DynamicTable,
        *,
        window_filter: tuple[str, date | None] | None = None,
    ) -> pa.Table:
        """Execute the DIT's SQL query in DuckDB against current Iceberg snapshots.

        Every upstream is read at its current snapshot; the full result is then
        recomputed. Callers decide whether to overwrite (unpartitioned) or
        window-overwrite (partitioned) the output — so this method never needs to
        read incremental deltas, which keeps it correct for joins and aggregations.

        If ``window_filter`` is ``(column, cutoff_date)``, the cutoff is applied at
        two layers:

        1. **Upstream pushdown:** for any upstream whose schema contains
           ``column``, the Iceberg scan is given a ``column >= cutoff`` row
           filter so partition/file pruning happens before bytes are read.
        2. **Output guard:** the user's query is wrapped in
           ``SELECT * FROM (...) WHERE column >= cutoff`` to handle the case
           where ``column`` is synthesised by a projection (e.g.
           ``date_trunc('day', ts) AS column``) and is therefore not present in
           any upstream schema.
        """
        con = duckdb.connect(":memory:")
        con.execute("SET TimeZone='UTC'")

        try:
            for upstream in dit.upstream_tables:
                if not self.catalog_mgr.table_exists(upstream):
                    raise RuntimeError(
                        f"Upstream table {upstream!r} required by {dit.name!r} does not exist"
                    )

                row_filter = self._upstream_row_filter(upstream, window_filter)
                alias = _view_alias(upstream)
                parquet_view = self._parquet_view_sql(upstream, row_filter, window_filter)
                if parquet_view is not None:
                    con.execute(f"CREATE VIEW {alias} AS {parquet_view}")
                else:
                    arrow_table = self._read_upstream(upstream, row_filter=row_filter)
                    con.register(alias, arrow_table)

            rewritten = _rewrite_table_names(dit.query, dit.upstream_tables)

            if window_filter is not None and window_filter[1] is not None:
                col, cutoff = window_filter
                rewritten = (
                    f"SELECT * FROM ({rewritten.rstrip(';').strip()}) AS _floe_inner "
                    f"WHERE {col} >= DATE '{cutoff.isoformat()}'"
                )

            result = con.sql(rewritten)
            # DuckDB 2 removed fetch_arrow_table. Prefer its replacement while
            # retaining support for older 1.x versions allowed by our dependency.
            if hasattr(result, "to_arrow_table"):
                return result.to_arrow_table()
            return result.fetch_arrow_table()
        finally:
            con.close()

    def _upstream_row_filter(
        self,
        upstream: str,
        window_filter: tuple[str, date | None] | None,
    ):
        """Build an Iceberg row filter for the windowed slice of `upstream`.

        Returns ``None`` if the upstream's schema does not contain the partition
        column — in that case the column is synthesised by the user's query and
        the outer SQL guard takes over.
        """
        if window_filter is None or window_filter[1] is None:
            return None
        col, cutoff = window_filter
        table = self.catalog_mgr.load_table(upstream)
        if col not in table.schema().column_names:
            return None
        return GreaterThanOrEqual(col, cutoff.isoformat())

    def _parquet_view_sql(
        self,
        name: str,
        row_filter,
        window_filter: tuple[str, date | None] | None,
    ) -> str | None:
        """SQL that lets DuckDB scan the upstream's Parquet files directly.

        Reading an upstream through ``scan().to_arrow()`` copies the whole table
        into memory before DuckDB sees it, which caps the table size at RAM.
        Pointing ``read_parquet`` at the data files Iceberg planned for the
        current snapshot lets DuckDB stream, project, and spill instead.

        This is only equivalent to the Iceberg read when every planned file is
        local, has no delete files, and was written with the table's only
        schema using unescaped column names (Parquet columns are matched by
        name, Iceberg's by field ID).
        Anything else returns ``None`` and the caller uses the Arrow path.
        Iceberg's ``plan_files`` prunes whole files by the row filter, and the
        window predicate is repeated in SQL to drop the remaining rows.
        """
        table = self.catalog_mgr.load_table(name)
        if table.current_snapshot() is None or len(table.metadata.schemas) != 1:
            return None
        # PyIceberg writes names that are not valid Avro names (spaces, dashes,
        # `count_star()`, ...) under an escaped name in Parquet.
        if sanitize_column_names(table.schema()) != table.schema():
            return None
        scan = table.scan(row_filter=row_filter) if row_filter is not None else table.scan()
        paths: list[str] = []
        for task in scan.plan_files():
            if task.delete_files:
                return None
            local = _local_file_path(task.file.file_path)
            if local is None:
                return None
            paths.append(local)
        if not paths:
            return None

        columns = ", ".join(f'"{f.name}"' for f in table.schema().fields)
        files = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
        sql = f"SELECT {columns} FROM read_parquet([{files}])"
        if row_filter is not None and window_filter is not None and window_filter[1] is not None:
            col, cutoff = window_filter
            sql += f" WHERE \"{col}\" >= DATE '{cutoff.isoformat()}'"
        return sql

    def _read_upstream(self, name: str, *, row_filter=None) -> pa.Table:
        """Read an upstream Iceberg table at its current snapshot, optionally row-filtered."""
        table = self.catalog_mgr.load_table(name)
        # A table without a snapshot still has a schema. Let the Iceberg scan
        # preserve it so projections and downstream empty tables remain valid.
        scan_kwargs = {"row_filter": row_filter} if row_filter is not None else {}
        return table.scan(**scan_kwargs).to_arrow()


def _view_alias(table_name: str) -> str:
    """DuckDB view alias for a fully-qualified Iceberg table."""
    return table_name.replace(".", "__")


def _local_file_path(location: str) -> str | None:
    """Turn an Iceberg data file location into a local path, or None if remote.

    Handles ``file:///abs``, ``file://C:/abs`` (the form Floe writes on
    Windows), and bare paths. Returns None if the file is not on this machine.
    """
    if location.startswith("file:"):
        path = location[len("file:") :]
        if path.startswith("//"):
            path = path[2:]
        # file:///C:/x -> /C:/x -> C:/x
        if len(path) > 2 and path[0] == "/" and path[2] == ":":
            path = path[1:]
    elif "://" in location:
        return None
    else:
        path = location
    return path if os.path.isfile(path) else None


def _rewrite_table_names(query: str, upstreams: list[str]) -> str:
    """Replace dotted upstream names in the SQL with DuckDB view aliases."""
    import re

    rewritten = query
    # Sort by length desc so longer names match first (avoid partial matches).
    for name in sorted(upstreams, key=len, reverse=True):
        alias = _view_alias(name)
        # Match the dotted name as a whole word (not surrounded by alphanumerics or _)
        pattern = r"(?<![\w.])" + re.escape(name) + r"(?![\w.])"
        rewritten = re.sub(pattern, alias, rewritten)
    return rewritten


def strip_lineage_columns(arrow: pa.Table) -> pa.Table:
    """Remove lineage cols (used by tests to compare logical content)."""
    cols = [c for c in arrow.column_names if c not in LINEAGE_COLUMNS]
    return arrow.select(cols)


# Back-compat alias: the DuckDB executor was historically the only executor and
# was named ``RefreshExecutor``. Keep the old name importable.
RefreshExecutor = DuckDBExecutor
