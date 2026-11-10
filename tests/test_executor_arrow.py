"""Compatibility with the Arrow result APIs across supported DuckDB versions."""

from types import SimpleNamespace

import pyarrow as pa
import pytest

from floe.executor import DuckDBExecutor
from floe.parser import parse_dit_sql


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
    monkeypatch.setattr("floe.executor.duckdb.connect", lambda _: connection)
    executor = DuckDBExecutor(catalog_mgr, {})
    dit = parse_dit_sql("CREATE DYNAMIC TABLE silver.constant AS SELECT 1 AS value")
    assert executor._execute_query(dit).equals(expected)
