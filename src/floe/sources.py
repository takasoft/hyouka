"""Declarative sources: load files into bronze Iceberg tables, incrementally.

A source (see :class:`floe.config.SourceConfig`) names a set of files, local or
remote, and the bronze table they feed. Each ``floe load`` / ``floe run``:

1. **Expands** the source into an ordered list of files (one URL per time step
   for a templated URL, or every match of a local glob).
2. **Skips** files already loaded. The loaded set is read from the target
   table's snapshot summaries (``floe.source.files``), which are written in the
   same atomic commit as the rows. A crash can therefore never record a file as
   loaded without its rows, or the reverse.
3. **Fetches and transforms** new files in a small thread pool. Each worker
   downloads the file (remote only), reads it with DuckDB as the relation
   ``raw``, applies the source's ``select``, and splits rows by ``checks``.
4. **Commits** results in file order, batching several files per Iceberg
   append so the table gets reasonably sized data files.

**Schema evolution.** When a batch has columns the table does not, they are
added (``union_by_name``) in the same transaction as the append, so the new
schema, the rows, and the loaded-file record land in one commit. Table columns
missing from a batch are written as NULL. Several sources may feed one table,
for example one entry per upstream format with its own window and ``select``.
A column whose type changes incompatibly fails the load.

Quarantined rows are committed to the quarantine table before the passing rows
go to the target. Each quarantine commit also records its files, so a retry
after a crash between the two commits does not duplicate quarantined rows.

Limits worth knowing: files are identified by URL or relative path only, so a
local file edited after loading is not reloaded. Changing ``select`` affects new
files only. Two concurrent loads of the same source can both append a file.
"""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import duckdb
import pyarrow as pa

from floe.catalog import CatalogManager
from floe.config import SourceConfig
from floe.models import _parse_lag

SOURCE_FILES_PROPERTY = "floe.source.files"
# On quarantine commits: the target table the rejected rows were meant for, so
# sources with different targets can share one quarantine table.
SOURCE_TARGET_PROPERTY = "floe.source.target"
SOURCE_FILE_COL = "_floe_source_file"
LOADED_AT_COL = "_floe_loaded_at"
DQ_FAILURE_COL = "_floe_dq_failure_reason"

# Flush a commit once this many rows or files are pending. Larger batches give
# fewer, bigger Iceberg data files. Smaller ones lose less work on a crash.
_BATCH_ROWS = 1_000_000
_BATCH_FILES = 100

_MAX_FILES = 100_000
_DOWNLOAD_TIMEOUT_SECONDS = 120
_DOWNLOAD_ATTEMPTS = 3
_USER_AGENT = "floe-source-loader"

_READERS = {"json": "read_json", "csv": "read_csv", "parquet": "read_parquet"}
_EXTENSIONS = {
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
}
_COMPRESSION_SUFFIXES = {".gz", ".zst", ".zstd"}


@dataclass(frozen=True)
class SourceFile:
    """One loadable unit. ``key`` is what gets recorded as loaded."""

    key: str
    location: str
    remote: bool


@dataclass
class SourceLoadResult:
    """Outcome of loading one source."""

    table: str
    files_total: int = 0
    files_loaded: int = 0
    files_already_loaded: int = 0
    files_missing: list[str] = field(default_factory=list)
    rows_loaded: int = 0
    rows_quarantined: int = 0
    commits: int = 0
    duration_ms: int = 0


@dataclass
class _FileResult:
    file: SourceFile
    passed: pa.Table | None = None
    failed: pa.Table | None = None
    missing: bool = False
    local_path: Path | None = None


# --- file expansion -----------------------------------------------------------


def expand_source_files(
    source: SourceConfig, project_root: Path, *, now: datetime | None = None
) -> list[SourceFile]:
    """Return every file the source covers, in load order."""
    if source.path is not None:
        return _expand_glob(source.path, project_root)
    assert source.url is not None
    if not source.is_time_templated:
        return [SourceFile(key=source.url, location=source.url, remote=_is_remote(source.url))]
    return _expand_url_template(source, now or datetime.now(UTC))


def _expand_glob(pattern: str, project_root: Path) -> list[SourceFile]:
    base = Path(pattern)
    full = base if base.is_absolute() else project_root / base
    files = []
    for match in sorted(glob.glob(str(full), recursive=True)):
        p = Path(match)
        if not p.is_file():
            continue
        try:
            key = p.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            key = p.resolve().as_posix()
        files.append(SourceFile(key=key, location=str(p), remote=False))
    return files


def _expand_url_template(source: SourceConfig, now: datetime) -> list[SourceFile]:
    assert source.url and source.every and source.start
    step = timedelta(seconds=_parse_lag(source.every))
    end = source.end or _floor(now, source.start, step)
    files: list[SourceFile] = []
    t = source.start
    while t <= end:
        url = render_template(source.url, t)
        files.append(SourceFile(key=url, location=url, remote=_is_remote(url)))
        if len(files) > _MAX_FILES:
            raise ValueError(
                f"source {source.table!r} expands to more than {_MAX_FILES} files, "
                "so narrow start/end or use a larger 'every'"
            )
        t += step
    return files


def _floor(now: datetime, start: datetime, step: timedelta) -> datetime:
    """Latest step boundary at or before ``now``, aligned to ``start``."""
    if now < start:
        return start - step
    return start + step * ((now - start) // step)


def render_template(template: str, t: datetime) -> str:
    """Fill a URL template for time step ``t``."""
    return template.format(
        time=t,
        date=t.strftime("%Y-%m-%d"),
        year=t.year,
        month=t.month,
        day=t.day,
        hour=t.hour,
        minute=t.minute,
    )


def _is_remote(location: str) -> bool:
    return urllib.parse.urlparse(location).scheme in {"http", "https"}


def infer_format(location: str) -> str:
    """Infer json/csv/parquet from a file name, ignoring compression suffixes."""
    suffixes = [s.lower() for s in PurePosixPath(urllib.parse.urlparse(location).path).suffixes]
    while suffixes and suffixes[-1] in _COMPRESSION_SUFFIXES:
        suffixes.pop()
    if suffixes and suffixes[-1] in _EXTENSIONS:
        return _EXTENSIONS[suffixes[-1]]
    raise ValueError(f"cannot infer the format of {location!r}, so set 'format' on the source")


# --- loaded-file bookkeeping --------------------------------------------------


def loaded_files(catalog_mgr: CatalogManager, table: str, *, target: str | None = None) -> set[str]:
    """Files recorded as loaded across every snapshot of ``table``.

    With ``target``, only count quarantine commits made for that target table.
    Expiring old snapshots also forgets which files they loaded, so keep the
    snapshots of source tables (or reload into a fresh table).
    """
    if not catalog_mgr.table_exists(table):
        return set()
    keys: set[str] = set()
    for snap in catalog_mgr.load_table(table).snapshots():
        summary = snap.summary
        raw = summary.get(SOURCE_FILES_PROPERTY) if summary else None
        if not raw:
            continue
        if target is not None and summary.get(SOURCE_TARGET_PROPERTY, target) != target:
            continue
        keys.update(json.loads(raw))
    return keys


# --- the loader ---------------------------------------------------------------


class SourceLoader:
    """Loads declarative sources into Iceberg. One instance per pipeline."""

    def __init__(
        self,
        catalog_mgr: CatalogManager,
        project_root: Path,
        *,
        progress: Callable[[str], None] | None = None,
    ):
        self.catalog_mgr = catalog_mgr
        self.project_root = project_root
        self.download_dir = project_root / ".floe" / "downloads"
        self._progress = progress or (lambda _msg: None)

    def load(self, source: SourceConfig, *, now: datetime | None = None) -> SourceLoadResult:
        started = time.monotonic()
        result = SourceLoadResult(table=source.table)
        files = expand_source_files(source, self.project_root, now=now)
        done = loaded_files(self.catalog_mgr, source.table)
        quarantine_done = (
            loaded_files(self.catalog_mgr, source.quarantine, target=source.table)
            if source.quarantine
            else set()
        )
        todo = [f for f in files if f.key not in done]
        result.files_total = len(files)
        result.files_already_loaded = len(files) - len(todo)
        self._progress(
            f"{source.table}: {len(todo)} new of {len(files)} file(s) "
            f"({result.files_already_loaded} already loaded)"
        )

        pending: list[_FileResult] = []
        try:
            for item in self._fetch_all(source, todo):
                if item.missing:
                    result.files_missing.append(item.file.key)
                    continue
                pending.append(item)
                pending_rows = sum(r.passed.num_rows + r.failed.num_rows for r in pending)
                if pending_rows >= _BATCH_ROWS or len(pending) >= _BATCH_FILES:
                    # Detach the batch before committing: if anything fails after
                    # the target append, `finally` must not append it again.
                    batch, pending = pending, []
                    self._commit(source, batch, quarantine_done, result)
        finally:
            # Keep finished work even if a later file failed.
            if pending:
                batch, pending = pending, []
                self._commit(source, batch, quarantine_done, result)
            result.duration_ms = int((time.monotonic() - started) * 1000)
        return result

    # -- fetching / transforming (runs in worker threads) --

    def _fetch_all(self, source: SourceConfig, files: list[SourceFile]) -> Iterator[_FileResult]:
        """Yield transformed files in order while up to ``parallel`` run ahead."""
        if not files:
            return
        window = source.parallel * 2
        with ThreadPoolExecutor(max_workers=source.parallel) as pool:
            futures: list[Future[_FileResult]] = []
            next_index = 0
            try:
                while next_index < len(files) or futures:
                    while next_index < len(files) and len(futures) < window:
                        futures.append(pool.submit(self._process, source, files[next_index]))
                        next_index += 1
                    yield futures.pop(0).result()
            finally:
                for f in futures:
                    f.cancel()

    def _process(self, source: SourceConfig, file: SourceFile) -> _FileResult:
        if file.remote:
            local = self._download(file.location)
            if local is None:
                return _FileResult(file=file, missing=True)
        else:
            local = Path(file.location)
        passed, failed = transform_file(source, file.key, local)
        return _FileResult(
            file=file,
            passed=passed,
            failed=failed,
            local_path=local if file.remote else None,
        )

    def _download(self, url: str) -> Path | None:
        """Download ``url`` into the download cache. Returns None on 404/410.

        Files are written to ``<name>.part`` and renamed when complete, so a
        file left in the cache by an interrupted run is always whole and is
        reused instead of downloaded again.
        """
        name = PurePosixPath(urllib.parse.urlparse(url).path).name or "download"
        digest = hashlib.sha1(url.encode()).hexdigest()[:12]
        target = self.download_dir / f"{digest}-{name}"
        if target.exists():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                with (
                    urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp,
                    open(part, "wb") as out,
                ):
                    shutil.copyfileobj(resp, out, length=1 << 20)
                part.replace(target)
                return target
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 410):
                    return None
                if exc.code < 500 or attempt == _DOWNLOAD_ATTEMPTS:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == _DOWNLOAD_ATTEMPTS:
                    raise
            time.sleep(2**attempt)
        return None  # unreachable: the loop either returns or raises

    # -- committing (main thread, file order) --

    def _commit(
        self,
        source: SourceConfig,
        batch: list[_FileResult],
        quarantine_done: set[str],
        result: SourceLoadResult,
    ) -> None:
        keys = [r.file.key for r in batch]
        passed = pa.concat_tables([r.passed for r in batch], promote_options="default")

        if source.quarantine:
            # Skip files whose failed rows a previous, interrupted run already
            # quarantined (that run committed quarantine but not the target).
            new_failed = [r.failed for r in batch if r.file.key not in quarantine_done]
            failed = pa.concat_tables(
                new_failed or [batch[0].failed.slice(0, 0)], promote_options="default"
            )
            # Create (and evolve) the quarantine table even when nothing failed,
            # so downstream SQL can always reference it and its columns.
            if failed.num_rows:
                self._append(
                    source.quarantine, failed, keys, extra={SOURCE_TARGET_PROPERTY: source.table}
                )
                quarantine_done.update(keys)
                result.rows_quarantined += failed.num_rows
            else:
                self._ensure_table(source.quarantine, failed.schema)
                self._evolve(source.quarantine, failed.schema)

        self._append(source.table, passed, keys)
        result.commits += 1
        result.files_loaded += len(batch)
        result.rows_loaded += passed.num_rows
        self._progress(
            f"{source.table}: committed {len(batch)} file(s), {passed.num_rows:,} row(s) "
            f"[{result.files_loaded} file(s), {result.rows_loaded:,} row(s) so far]"
        )

        for r in batch:
            if r.local_path is not None:
                with contextlib.suppress(OSError):
                    r.local_path.unlink(missing_ok=True)

    def _append(
        self, name: str, data: pa.Table, keys: list[str], *, extra: dict[str, str] | None = None
    ) -> None:
        """Append ``data`` and record ``keys``, adding any new columns first.

        The schema update and the append share one Iceberg transaction, so a
        failed append never leaves behind a half-applied schema change.
        ``union_by_name`` also adds new nested struct fields, and is a no-op
        (no new schema version) when nothing is new.
        """
        self._ensure_table(name, data.schema)
        table = self.catalog_mgr.load_table(name)
        _reject_case_collisions(name, table, data.schema)
        props = {SOURCE_FILES_PROPERTY: json.dumps(keys), **(extra or {})}
        with table.transaction() as txn:
            with txn.update_schema() as update:
                update.union_by_name(data.schema)
            txn.append(data, snapshot_properties=props)

    def _evolve(self, name: str, schema: pa.Schema) -> None:
        table = self.catalog_mgr.load_table(name)
        _reject_case_collisions(name, table, schema)
        with table.update_schema() as update:
            update.union_by_name(schema)

    def _ensure_table(self, name: str, schema: pa.Schema) -> None:
        if not self.catalog_mgr.table_exists(name):
            self.catalog_mgr.ensure_namespace(name.rsplit(".", 1)[0])
            self.catalog_mgr.catalog.create_table(name, schema=schema)


def _reject_case_collisions(name: str, table: Any, schema: pa.Schema) -> None:
    """Refuse a new column that differs from an existing one only by case.

    Iceberg would add it as a separate column, but SQL engines such as DuckDB
    match column names case-insensitively and would merge the two.
    """
    existing = {c.casefold(): c for c in table.schema().column_names}
    for col in schema.names:
        match = existing.get(col.casefold())
        if match is not None and match != col:
            raise ValueError(
                f"column {col!r} differs only by case from {match!r} in {name!r}; "
                "rename it in the source's select"
            )


# --- per-file transform -------------------------------------------------------


def transform_file(source: SourceConfig, key: str, path: Path) -> tuple[pa.Table, pa.Table]:
    """Read one file with DuckDB and return ``(passed_rows, failed_rows)``.

    Both tables carry ``_floe_source_file`` and ``_floe_loaded_at``. Failed rows
    also carry ``_floe_dq_failure_reason``, a comma-separated list of the checks
    they failed (a check that evaluates to NULL counts as failed).

    Raises ``ValueError`` if rows fail checks and the source has no quarantine.
    """
    fmt = source.format or infer_format(key)
    reader = _READERS[fmt]
    args = [_sql_literal(path.as_posix())]
    args += [f"{name} = {_sql_literal(value)}" for name, value in source.options.items()]
    loaded_at = datetime.now(UTC).isoformat()
    lineage = (
        f"{_sql_literal(key)} AS {SOURCE_FILE_COL}, "
        f"TIMESTAMPTZ {_sql_literal(loaded_at)} AS {LOADED_AT_COL}"
    )

    con = duckdb.connect(":memory:")
    try:
        con.execute("SET TimeZone='UTC'")
        con.execute(f"CREATE TEMP VIEW raw AS SELECT * FROM {reader}({', '.join(args)})")
        con.execute(f"CREATE TEMP TABLE _floe_rows AS {source.select.strip().rstrip(';')}")
        if not source.checks:
            passed = _to_arrow(con.sql(f"SELECT *, {lineage} FROM _floe_rows"))
            # Same shape as a checked failure, so a quarantine table created now
            # still accepts rows once checks are added.
            failed = _to_arrow(
                con.sql(
                    f"SELECT *, ''::VARCHAR AS {DQ_FAILURE_COL}, {lineage} FROM _floe_rows LIMIT 0"
                )
            )
            return passed, failed

        failures = ", ".join(
            f"CASE WHEN NOT coalesce(({expr}), false) THEN {_sql_literal(name)} END"
            for name, expr in source.checks.items()
        )
        con.execute(
            "CREATE TEMP TABLE _floe_checked AS "
            f"SELECT *, concat_ws(', ', {failures}) AS {DQ_FAILURE_COL} FROM _floe_rows"
        )
        passed = _to_arrow(
            con.sql(
                f"SELECT * EXCLUDE ({DQ_FAILURE_COL}), {lineage} "
                f"FROM _floe_checked WHERE {DQ_FAILURE_COL} = ''"
            )
        )
        failed = _to_arrow(
            con.sql(f"SELECT *, {lineage} FROM _floe_checked WHERE {DQ_FAILURE_COL} <> ''")
        )
    finally:
        con.close()

    if failed.num_rows and not source.quarantine:
        reasons = sorted(set(failed[DQ_FAILURE_COL].to_pylist()))
        raise ValueError(
            f"{failed.num_rows} row(s) in {key} failed checks ({'; '.join(reasons)}) and "
            f"source {source.table!r} has no quarantine table"
        )
    return passed, failed


def _to_arrow(relation: Any) -> pa.Table:
    # DuckDB 2 renamed fetch_arrow_table to to_arrow_table. Support both.
    if hasattr(relation, "to_arrow_table"):
        return relation.to_arrow_table()
    return relation.fetch_arrow_table()


def _sql_literal(value: Any) -> str:
    """Render a YAML value as a DuckDB literal (strings, numbers, lists, structs)."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_sql_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{_sql_literal(str(k))}: {_sql_literal(v)}" for k, v in value.items())
        return "{" + items + "}"
    raise TypeError(f"unsupported reader option value {value!r}")
