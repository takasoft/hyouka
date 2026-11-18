"""Compatibility with the Arrow result APIs across supported DuckDB versions."""

from types import SimpleNamespace

import pyarrow as pa
import pytest

from hyouka.executor import DuckDBExecutor
from hyouka.parser import parse_dit_sql
from hyouka.pipeline import Pipeline


@pytest.mark.parametrize("modern", [True, False])
def test_arrow_result_api(monkeypatch, catalog_mgr, modern):
    expected = pa.table({"value": [1]})
    connection = SimpleNamespace(close=lambda: None)
    connection.execute = lambda _: connection
    connection.sql = lambda _: connection
    if modern:
        connection.to_arrow_table = lambda: expected
    else:
        connection.fetch_arrow_table = lambda: expected
    monkeypatch.setattr("hyouka.executor.duckdb.connect", lambda _: connection)
    executor = DuckDBExecutor(catalog_mgr, {})
    dit = parse_dit_sql(
        "CREATE DYNAMIC TABLE silver.constant REFRESH_MODE = 'FULL' AS SELECT 1 AS value"
    )
    assert executor._execute_query(dit).equals(expected)


def test_never_written_source_keeps_schema_through_two_hops(
    hyouka_config, catalog_mgr, sample_orders
):
    catalog_mgr.ensure_namespace("bronze")
    source = catalog_mgr.catalog.create_table("bronze.orders", schema=sample_orders.schema)
    dits = [
        parse_dit_sql(
            "CREATE DYNAMIC TABLE silver.orders AS SELECT order_id, amount FROM bronze.orders"
        ),
        parse_dit_sql(
            "CREATE DYNAMIC TABLE gold.totals REFRESH_MODE = 'FULL' "
            "AS SELECT COUNT(*) AS n FROM silver.orders"
        ),
    ]
    pipeline = Pipeline(hyouka_config, dits, catalog_mgr)
    results = pipeline.refresh_all()
    assert [r.rows_written for r in results] == [0, 1]
    silver = catalog_mgr.load_table("silver.orders").scan().to_arrow()
    assert silver.schema.field("order_id").type == sample_orders.schema.field("order_id").type
    assert catalog_mgr.load_table("gold.totals").scan().to_arrow()["n"].to_pylist() == [0]
    assert all(r.skipped for r in pipeline.refresh_all())

    source.append(sample_orders)
    assert all(not r.skipped for r in pipeline.refresh_all())
    assert catalog_mgr.load_table("gold.totals").scan().to_arrow()["n"].to_pylist() == [3]


def test_local_upstreams_are_scanned_as_parquet_files(catalog_mgr, sample_orders):
    catalog_mgr.ensure_namespace("bronze")
    catalog_mgr.catalog.create_table("bronze.orders", schema=sample_orders.schema)
    catalog_mgr.append("bronze.orders", sample_orders)
    catalog_mgr.append("bronze.orders", sample_orders)
    executor = DuckDBExecutor(catalog_mgr, {})

    view = executor._parquet_view_sql("bronze.orders", None, None)
    assert view is not None and view[0].count(".parquet'") == 2 and view[1] is None

    dit = parse_dit_sql(
        "CREATE DYNAMIC TABLE gold.totals REFRESH_MODE = 'FULL' "
        "AS SELECT count(*) AS n, sum(amount) AS total FROM bronze.orders"
    )
    assert executor._execute_query(dit).to_pylist() == [{"n": 6, "total": 85.5}]


def test_evolved_schemas_fall_back_to_the_iceberg_reader(catalog_mgr, sample_orders):
    from pyiceberg.types import StringType

    catalog_mgr.ensure_namespace("bronze")
    table = catalog_mgr.catalog.create_table("bronze.orders", schema=sample_orders.schema)
    table.append(sample_orders)
    with table.update_schema() as update:
        update.rename_column("amount", "total")
        update.add_column("note", StringType())
    executor = DuckDBExecutor(catalog_mgr, {})

    # Parquet files still say "amount"; only Iceberg's field IDs map it to "total".
    assert executor._parquet_view_sql("bronze.orders", None, None) is None
    dit = parse_dit_sql(
        "CREATE DYNAMIC TABLE gold.totals REFRESH_MODE = 'FULL' "
        "AS SELECT sum(total) AS total FROM bronze.orders"
    )
    assert executor._execute_query(dit).to_pylist() == [{"total": 42.75}]


def test_escaped_parquet_column_names_fall_back_to_the_iceberg_reader(catalog_mgr):
    catalog_mgr.ensure_namespace("bronze")
    data = pa.table({"Event Type": ["push", "push", "fork"]})
    catalog_mgr.catalog.create_table("bronze.events", schema=data.schema)
    catalog_mgr.append("bronze.events", data)
    executor = DuckDBExecutor(catalog_mgr, {})

    # PyIceberg stores "Event Type" as "Event_x20Type" inside the Parquet files.
    assert executor._parquet_view_sql("bronze.events", None, None) is None
    dit = parse_dit_sql(
        "CREATE DYNAMIC TABLE gold.types REFRESH_MODE = 'FULL' "
        'AS SELECT count(DISTINCT "Event Type") AS n FROM bronze.events'
    )
    assert executor._execute_query(dit).to_pylist() == [{"n": 2}]


def test_added_columns_keep_the_parquet_scan_and_read_as_null(catalog_mgr, sample_orders):
    catalog_mgr.ensure_namespace("bronze")
    table = catalog_mgr.catalog.create_table("bronze.orders", schema=sample_orders.schema)
    table.append(sample_orders)
    newer = pa.table(
        {
            "order_id": pa.array([4], type=pa.int64()),
            "customer_id": pa.array([102], type=pa.int64()),
            "channel": pa.array(["web"], type=pa.string()),
        }
    )
    with table.transaction() as txn:
        with txn.update_schema() as update:
            update.union_by_name(newer.schema)
        txn.append(newer)
    executor = DuckDBExecutor(catalog_mgr, {})

    view = executor._parquet_view_sql("bronze.orders", None, None)
    assert view is not None and view[1] is not None
    dit = parse_dit_sql(
        "CREATE DYNAMIC TABLE gold.by_channel REFRESH_MODE = 'FULL' AS "
        "SELECT channel, count(*) AS n, sum(amount) AS total FROM bronze.orders "
        "GROUP BY channel ORDER BY channel NULLS FIRST"
    )
    assert executor._execute_query(dit).to_pylist() == [
        {"channel": None, "n": 3, "total": 42.75},
        {"channel": "web", "n": 1, "total": None},
    ]


def test_names_differing_only_by_case_fall_back_to_the_iceberg_reader(catalog_mgr):
    catalog_mgr.ensure_namespace("bronze")
    first = pa.table({"id": [1], "name": ["old"]})
    table = catalog_mgr.catalog.create_table("bronze.people", schema=first.schema)
    table.append(first)
    second = pa.table({"id": [2], "Name": ["new"]})
    with table.transaction() as txn:
        with txn.update_schema() as update:
            update.union_by_name(second.schema)
        txn.append(second)
    executor = DuckDBExecutor(catalog_mgr, {})

    # Parquet name matching in DuckDB ignores case and would merge the two
    # columns. (The source loader refuses to create such a pair.)
    assert executor._parquet_view_sql("bronze.people", None, None) is None
