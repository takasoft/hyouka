"""Declarative sources: expansion, incremental loading, quarantine, and `floe run`."""

from __future__ import annotations

import gzip
import http.server
import json
import threading
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pyarrow as pa
import pytest
from typer.testing import CliRunner

from floe.config import FloeConfig, SourceConfig
from floe.pipeline import Pipeline
from floe.sources import (
    DQ_FAILURE_COL,
    SOURCE_FILE_COL,
    SourceLoader,
    expand_source_files,
    loaded_files,
)


def _write_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["order_id,amount"] + [",".join("" if v is None else str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows(catalog_mgr, table: str) -> pa.Table:
    return catalog_mgr.load_table(table).scan().to_arrow()


# --- expansion ----------------------------------------------------------------


def test_hourly_template_expands_inclusive_with_unpadded_hours(tmp_path):
    src = SourceConfig(
        table="bronze.events",
        url="https://example.test/{time:%Y-%m-%d}-{hour}.json.gz",
        every="1 hour",
        start="2024-01-15 22:00",
        end="2024-01-16 01:00",
    )
    keys = [f.key for f in expand_source_files(src, tmp_path)]
    assert keys == [
        "https://example.test/2024-01-15-22.json.gz",
        "https://example.test/2024-01-15-23.json.gz",
        "https://example.test/2024-01-16-0.json.gz",
        "https://example.test/2024-01-16-1.json.gz",
    ]


def test_open_ended_template_stops_at_the_last_step_before_now(tmp_path):
    src = SourceConfig(
        table="bronze.events",
        url="https://example.test/{date}.csv",
        every="1 day",
        start="2024-01-01",
    )
    now = datetime(2024, 1, 3, 12, tzinfo=UTC)
    keys = [f.key for f in expand_source_files(src, tmp_path, now=now)]
    assert keys == [f"https://example.test/2024-01-0{d}.csv" for d in (1, 2, 3)]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"url": "https://x.test/a.csv", "path": "a.csv"}, "exactly one of"),
        ({"url": "https://x.test/{date}.csv"}, "needs 'every' and 'start'"),
        ({"path": "*.csv", "every": "1 day"}, "only apply to a templated url"),
        ({"path": "*.csv", "quarantine": "bronze.events"}, "different table"),
        ({"path": "*.csv", "format": "xml"}, "unsupported source format"),
        ({"path": "*.csv", "checks": {"bad name": "true"}}, "not a valid identifier"),
    ],
)
def test_source_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SourceConfig(table="bronze.events", **kwargs)


def test_a_table_cannot_be_both_a_source_target_and_a_quarantine():
    with pytest.raises(ValueError, match="also a quarantine table"):
        FloeConfig(
            project="p",
            sources=[
                {"table": "bronze.a", "path": "a/*.csv"},
                {"table": "bronze.b", "path": "b/*.csv", "quarantine": "bronze.a"},
            ],
        )


def test_source_table_cannot_also_be_a_dynamic_table(floe_config, catalog_mgr):
    from floe.parser import parse_dit_sql

    floe_config.sources = [SourceConfig(table="bronze.orders", path="*.csv")]
    dit = parse_dit_sql("CREATE DYNAMIC TABLE bronze.orders AS SELECT * FROM raw.orders")
    with pytest.raises(ValueError, match="exactly one writer"):
        Pipeline(floe_config, [dit], catalog_mgr)


# --- local files ----------------------------------------------------------------


def test_local_files_load_once_and_new_files_load_incrementally(project_dir, catalog_mgr):
    _write_csv(project_dir / "data" / "a.csv", [(1, 10.0), (2, 20.0)])
    _write_csv(project_dir / "data" / "b.csv", [(3, 30.0)])
    src = SourceConfig(table="bronze.orders", path="data/*.csv")
    loader = SourceLoader(catalog_mgr, project_dir)

    first = loader.load(src)
    assert (first.files_loaded, first.rows_loaded, first.commits) == (2, 3, 1)
    assert loaded_files(catalog_mgr, "bronze.orders") == {"data/a.csv", "data/b.csv"}

    again = loader.load(src)
    assert (again.files_loaded, again.files_already_loaded, again.commits) == (0, 2, 0)

    _write_csv(project_dir / "data" / "c.csv", [(4, 40.0)])
    third = loader.load(src)
    assert (third.files_loaded, third.rows_loaded) == (1, 1)

    table = _rows(catalog_mgr, "bronze.orders")
    assert sorted(table["order_id"].to_pylist()) == [1, 2, 3, 4]
    assert set(table[SOURCE_FILE_COL].to_pylist()) == {"data/a.csv", "data/b.csv", "data/c.csv"}


def test_failed_checks_go_to_quarantine_with_reasons(project_dir, catalog_mgr):
    _write_csv(project_dir / "data" / "a.csv", [(1, 10.0), (2, -5.0), (None, 7.0), (4, None)])
    src = SourceConfig(
        table="bronze.orders",
        path="data/*.csv",
        select="SELECT order_id, amount FROM raw",
        checks={"has_id": "order_id IS NOT NULL", "positive": "amount > 0"},
        quarantine="quarantine.orders",
    )
    result = SourceLoader(catalog_mgr, project_dir).load(src)
    assert (result.rows_loaded, result.rows_quarantined) == (1, 3)

    good = _rows(catalog_mgr, "bronze.orders")
    assert good["order_id"].to_pylist() == [1]
    assert DQ_FAILURE_COL not in good.column_names

    bad = _rows(catalog_mgr, "quarantine.orders").sort_by("amount")
    # A check that evaluates to NULL (amount IS NULL) counts as failed.
    reasons = dict(zip(bad["amount"].to_pylist(), bad[DQ_FAILURE_COL].to_pylist(), strict=True))
    assert reasons == {-5.0: "positive", 7.0: "has_id", None: "positive"}


def test_failed_checks_without_quarantine_abort_and_commit_nothing(project_dir, catalog_mgr):
    _write_csv(project_dir / "data" / "a.csv", [(1, -1.0)])
    src = SourceConfig(table="bronze.orders", path="data/*.csv", checks={"positive": "amount > 0"})
    with pytest.raises(ValueError, match="no quarantine table"):
        SourceLoader(catalog_mgr, project_dir).load(src)
    assert not catalog_mgr.table_exists("bronze.orders")


def test_retry_after_crash_between_commits_does_not_duplicate_quarantine(
    project_dir, catalog_mgr, monkeypatch
):
    _write_csv(project_dir / "data" / "a.csv", [(1, 10.0), (2, -5.0)])
    src = SourceConfig(
        table="bronze.orders",
        path="data/*.csv",
        checks={"positive": "amount > 0"},
        quarantine="quarantine.orders",
    )
    loader = SourceLoader(catalog_mgr, project_dir)
    real_ensure = loader._ensure_table

    def crash_on_target(name, schema):
        if name == "bronze.orders":
            raise RuntimeError("simulated crash after the quarantine commit")
        real_ensure(name, schema)

    monkeypatch.setattr(loader, "_ensure_table", crash_on_target)
    with pytest.raises(RuntimeError, match="simulated crash"):
        loader.load(src)
    assert _rows(catalog_mgr, "quarantine.orders").num_rows == 1
    assert loaded_files(catalog_mgr, "bronze.orders") == set()

    monkeypatch.setattr(loader, "_ensure_table", real_ensure)
    result = loader.load(src)
    assert (result.rows_loaded, result.rows_quarantined) == (1, 0)
    assert _rows(catalog_mgr, "quarantine.orders").num_rows == 1


# --- remote files ---------------------------------------------------------------


@pytest.fixture
def http_root(tmp_path):
    """Serve a directory over HTTP on localhost for the duration of a test."""
    root = tmp_path / "served"
    root.mkdir()
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _write_ndjson_gz(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_remote_hourly_files_skip_missing_hours_and_retry_them(http_root, project_dir, catalog_mgr):
    root, base = http_root
    _write_ndjson_gz(root / "2024-01-15-0.json.gz", [{"id": "1", "type": "PushEvent"}])
    _write_ndjson_gz(root / "2024-01-15-2.json.gz", [{"id": "3", "type": "WatchEvent"}])
    src = SourceConfig(
        table="bronze.events",
        url=base + "/{time:%Y-%m-%d}-{hour}.json.gz",
        every="1 hour",
        start="2024-01-15 00:00",
        end="2024-01-15 02:00",
        options={"columns": {"id": "VARCHAR", "type": "VARCHAR"}},
        select="SELECT CAST(id AS BIGINT) AS event_id, type AS event_type FROM raw",
    )
    loader = SourceLoader(catalog_mgr, project_dir)

    first = loader.load(src)
    assert first.files_loaded == 2
    assert first.files_missing == [base + "/2024-01-15-1.json.gz"]
    # Downloads are removed once their rows are committed.
    assert list(loader.download_dir.iterdir()) == []

    _write_ndjson_gz(root / "2024-01-15-1.json.gz", [{"id": "2", "type": "ForkEvent"}])
    second = loader.load(src)
    assert (second.files_loaded, second.files_already_loaded, second.files_missing) == (1, 2, [])
    assert sorted(_rows(catalog_mgr, "bronze.events")["event_id"].to_pylist()) == [1, 2, 3]


# --- floe run -------------------------------------------------------------------


def test_floe_run_loads_sources_then_refreshes_the_dag(project_dir):
    from floe.cli import app

    _write_csv(project_dir / "data" / "a.csv", [(1, 10.0), (2, 20.0)])
    (project_dir / "transformations" / "gold_totals.sql").write_text(
        "CREATE DYNAMIC TABLE gold.totals REFRESH_MODE = 'FULL' AS "
        "SELECT count(*) AS orders, sum(amount) AS revenue FROM bronze.orders",
        encoding="utf-8",
    )
    (project_dir / "floe.yaml").write_text(
        "project: run_test\nsources:\n  - table: bronze.orders\n    path: data/*.csv\n",
        encoding="utf-8",
    )
    config = str(project_dir / "floe.yaml")
    runner = CliRunner()

    first = runner.invoke(app, ["run", "-c", config])
    assert first.exit_code == 0, first.output
    pipeline = Pipeline.from_config(config)
    totals = _rows(pipeline.catalog_mgr, "gold.totals")
    assert totals.select(["orders", "revenue"]).to_pylist() == [{"orders": 2, "revenue": 30.0}]

    second = runner.invoke(app, ["run", "-c", config])
    assert second.exit_code == 0, second.output
    assert "skipped" in second.output

    _write_csv(project_dir / "data" / "b.csv", [(3, 5.0)])
    third = runner.invoke(app, ["run", "-c", config])
    assert third.exit_code == 0, third.output
    totals = _rows(pipeline.catalog_mgr, "gold.totals")
    assert totals.select(["orders", "revenue"]).to_pylist() == [{"orders": 3, "revenue": 35.0}]


def test_failure_after_the_target_commit_does_not_append_the_batch_again(
    project_dir, catalog_mgr, monkeypatch
):
    monkeypatch.setattr("floe.sources._BATCH_FILES", 1)
    _write_csv(project_dir / "data" / "a.csv", [(1, 10.0)])
    _write_csv(project_dir / "data" / "b.csv", [(2, 20.0)])
    src = SourceConfig(table="bronze.orders", path="data/*.csv")

    def broken_pipe(message):
        if "committed" in message:
            raise BrokenPipeError("output closed")

    loader = SourceLoader(catalog_mgr, project_dir, progress=broken_pipe)
    with pytest.raises(BrokenPipeError):
        loader.load(src)
    assert _rows(catalog_mgr, "bronze.orders")["order_id"].to_pylist() == [1]
    assert len(catalog_mgr.load_table("bronze.orders").snapshots()) == 1


def test_quarantine_created_before_checks_accepts_failures_later(project_dir, catalog_mgr):
    _write_csv(project_dir / "data" / "a.csv", [(1, 10.0)])
    loader = SourceLoader(catalog_mgr, project_dir)
    loader.load(SourceConfig(table="bronze.orders", path="data/*.csv", quarantine="q.orders"))
    assert DQ_FAILURE_COL in catalog_mgr.load_table("q.orders").schema().column_names

    _write_csv(project_dir / "data" / "b.csv", [(2, -1.0)])
    checked = SourceConfig(
        table="bronze.orders",
        path="data/*.csv",
        checks={"positive": "amount > 0"},
        quarantine="q.orders",
    )
    assert loader.load(checked).rows_quarantined == 1
    assert _rows(catalog_mgr, "q.orders")[DQ_FAILURE_COL].to_pylist() == ["positive"]


def test_sources_with_different_shapes_evolve_one_table(project_dir, catalog_mgr):
    """Two eras of one feed: a column disappears and another appears."""
    old_dir, new_dir = project_dir / "old", project_dir / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "a.csv").write_text("order_id,amount,coupon\n1,10.0,SAVE\n", encoding="utf-8")
    (new_dir / "b.csv").write_text("order_id,amount,channel\n2,20.0,web\n", encoding="utf-8")
    common = {"table": "sales.orders", "quarantine": "sales.orders_rejected"}
    old = SourceConfig(
        **common,
        path="old/*.csv",
        select="SELECT order_id, amount, coupon FROM raw",
        checks={"positive": "amount > 0"},
    )
    new = SourceConfig(
        **common,
        path="new/*.csv",
        select="SELECT order_id, amount, channel FROM raw",
        checks={"positive": "amount > 0"},
    )
    FloeConfig(project="p", sources=[old, new])  # sharing a target is allowed
    loader = SourceLoader(catalog_mgr, project_dir)

    loader.load(old)
    assert "channel" not in catalog_mgr.load_table("sales.orders").schema().column_names

    loader.load(new)
    table = catalog_mgr.load_table("sales.orders")
    assert {"coupon", "channel"} <= set(table.schema().column_names)
    rows = sorted(
        table.scan().to_arrow().select(["order_id", "coupon", "channel"]).to_pylist(),
        key=lambda r: r["order_id"],
    )
    assert rows == [
        {"order_id": 1, "coupon": "SAVE", "channel": None},
        {"order_id": 2, "coupon": None, "channel": "web"},
    ]
    # Each era's commit references the schema it was written with, and the
    # schema change rode along with the second append rather than its own commit.
    assert [s.schema_id for s in table.snapshots()] == [0, 1]
    assert loaded_files(catalog_mgr, "sales.orders") == {"old/a.csv", "new/b.csv"}
    # The empty quarantine table evolved too, so SQL can select the new column.
    assert "channel" in catalog_mgr.load_table("sales.orders_rejected").schema().column_names
    assert loader.load(old).files_loaded == 0


def test_sources_with_different_targets_can_share_a_quarantine(project_dir, catalog_mgr):
    _write_csv(project_dir / "data" / "a.csv", [(1, 10.0), (2, -5.0)])
    shared = {"path": "data/*.csv", "quarantine": "q.rejects"}
    by_amount = SourceConfig(table="sales.orders", checks={"positive": "amount > 0"}, **shared)
    by_id = SourceConfig(table="sales.order_ids", checks={"small_id": "order_id < 2"}, **shared)
    loader = SourceLoader(catalog_mgr, project_dir)

    assert loader.load(by_amount).rows_quarantined == 1
    assert loader.load(by_id).rows_quarantined == 1
    reasons = sorted(_rows(catalog_mgr, "q.rejects")[DQ_FAILURE_COL].to_pylist())
    assert reasons == ["positive", "small_id"]


def test_new_nested_fields_are_added_to_struct_columns(project_dir, catalog_mgr):
    data = project_dir / "data"
    data.mkdir()
    (data / "1.json").write_text('{"id": 1, "p": {"a": 1}}\n', encoding="utf-8")
    src = SourceConfig(table="raw.events", path="data/*.json")
    loader = SourceLoader(catalog_mgr, project_dir)
    loader.load(src)

    (data / "2.json").write_text('{"id": 2, "p": {"a": 2, "b": "x"}}\n', encoding="utf-8")
    loader.load(src)
    rows = sorted(
        _rows(catalog_mgr, "raw.events").select(["id", "p"]).to_pylist(), key=lambda r: r["id"]
    )
    assert rows == [{"id": 1, "p": {"a": 1, "b": None}}, {"id": 2, "p": {"a": 2, "b": "x"}}]


def test_a_new_column_differing_only_by_case_is_rejected(project_dir, catalog_mgr):
    (project_dir / "d").mkdir()
    (project_dir / "d" / "1.csv").write_text("id,name\n1,old\n", encoding="utf-8")
    src = SourceConfig(table="raw.people", path="d/*.csv")
    loader = SourceLoader(catalog_mgr, project_dir)
    loader.load(src)

    (project_dir / "d" / "2.csv").write_text("id,Name\n2,new\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs only by case"):
        loader.load(src)
    assert _rows(catalog_mgr, "raw.people").num_rows == 1
