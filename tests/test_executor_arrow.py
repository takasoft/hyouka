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
