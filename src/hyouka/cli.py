"""Hyouka CLI: `hyouka init`, `hyouka plan`, `hyouka load`, `hyouka run`, `hyouka apply`, `hyouka status`, ..."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table as RichTable

from hyouka.pipeline import Pipeline

app = typer.Typer(
    name="hyouka",
    help="Hyouka — declarative lakehouse engine on Apache Iceberg.",
    no_args_is_help=True,
)
console = Console()

DEFAULT_CONFIG = "hyouka.yaml"


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Project name"),
    path: Path = typer.Option(Path("."), "--path", "-p", help="Where to scaffold"),
):
    """Scaffold a new Hyouka project in the given directory."""
    target = path / project_name
    if target.exists() and any(target.iterdir()):
        console.print(f"[red]Error:[/red] {target} already exists and is not empty")
        raise typer.Exit(1)
    target.mkdir(parents=True, exist_ok=True)
    (target / "transformations").mkdir(exist_ok=True)
    (target / "warehouse").mkdir(exist_ok=True)

    (target / "hyouka.yaml").write_text(
        f"""project: {project_name}
version: "1.0"

catalog:
  type: sql
  uri: sqlite:///./.hyouka/catalog.db
  warehouse: ./warehouse

compute:
  engine: duckdb

defaults:
  lag: "5 minutes"
  refresh_mode: INCREMENTAL
  lineage: true

transformations_dir: transformations
""",
        encoding="utf-8",
    )

    console.print(f"[green]Created Hyouka project at {target}[/green]")
    console.print("Next steps:")
    console.print(f"  cd {target}")
    console.print("  # Add .sql files under transformations/")
    console.print("  hyouka plan")
    console.print("  hyouka apply")


@app.command()
def plan(config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", "-c")):
    """Validate DIT definitions and show the DAG."""
    pipeline = Pipeline.from_config(config)
    console.print(f"[bold]Project:[/bold] {pipeline.config.project}")
    if pipeline.sources:
        console.print(f"[bold]Declared sources:[/bold] {len(pipeline.sources)}")
        for src in pipeline.sources:
            where = src.url or src.path
            if src.is_time_templated:
                end = f"{src.end:%Y-%m-%d %H:%M}" if src.end else "now"
                where += f" every {src.every}, {src.start:%Y-%m-%d %H:%M} to {end} UTC"
            extra = f" (quarantine: {src.quarantine})" if src.quarantine else ""
            console.print(escape(f"  - {src.table} <- {where}{extra}"))
    console.print(f"[bold]Discovered DITs:[/bold] {len(pipeline.dits)}")
    console.print()
    console.print(pipeline.planner.render_ascii())


@app.command()
def apply(config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", "-c")):
    """Refresh every DIT in topological order."""
    pipeline = Pipeline.from_config(config)
    console.print(
        f"[bold]Engine:[/bold] {pipeline.config.compute.engine}    "
        f"[bold]Trigger:[/bold] {pipeline.config.compute.trigger}"
    )
    _print_refresh_results(pipeline.refresh_all())


@app.command()
def load(config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", "-c")):
    """Load new files from every declared source into its bronze table."""
    pipeline = Pipeline.from_config(config)
    if not pipeline.sources:
        console.print("[yellow]No sources declared in the config.[/yellow]")
        raise typer.Exit(1)
    _print_load_results(pipeline.load_sources(progress=_log))


@app.command()
def run(config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", "-c")):
    """Run the whole pipeline: load declared sources, then refresh every DIT."""
    pipeline = Pipeline.from_config(config)
    console.print(
        f"[bold]Engine:[/bold] {pipeline.config.compute.engine}    "
        f"[bold]Sources:[/bold] {len(pipeline.sources)}    "
        f"[bold]DITs:[/bold] {len(pipeline.dits)}"
    )
    loads = pipeline.load_sources(progress=_log)
    if loads:
        _print_load_results(loads)

    def _log_refresh(r):
        state = "skipped" if r.skipped else f"{r.rows_written:,} rows"
        _log(f"{r.table}: {state} ({r.duration_ms:,} ms)")

    _print_refresh_results(pipeline.refresh_all(on_result=_log_refresh))


def _log(message: str) -> None:
    console.print(f"[dim]{escape(message)}[/dim]")


def _print_load_results(results) -> None:
    t = RichTable(title="Source Loads")
    t.add_column("Table")
    t.add_column("New files", justify="right")
    t.add_column("Already loaded", justify="right")
    t.add_column("Missing", justify="right")
    t.add_column("Rows", justify="right")
    t.add_column("Quarantined", justify="right")
    t.add_column("Duration (ms)", justify="right")
    for r in results:
        t.add_row(
            r.table,
            str(r.files_loaded),
            str(r.files_already_loaded),
            str(len(r.files_missing)),
            f"{r.rows_loaded:,}",
            f"{r.rows_quarantined:,}",
            f"{r.duration_ms:,}",
        )
    console.print(t)
    for r in results:
        if r.files_missing:
            shown = ", ".join(r.files_missing[:3])
            more = f" and {len(r.files_missing) - 3} more" if len(r.files_missing) > 3 else ""
            console.print(
                f"[yellow]{r.table}: {len(r.files_missing)} file(s) not found yet "
                f"(retried next run): {escape(shown)}{more}[/yellow]"
            )


def _print_refresh_results(results) -> None:
    t = RichTable(title="Refresh Results")
    t.add_column("Table")
    t.add_column("Mode")
    t.add_column("Rows", justify="right")
    t.add_column("In Snapshot", justify="right")
    t.add_column("Out Snapshot", justify="right")
    t.add_column("Duration (ms)", justify="right")
    t.add_column("Status")

    for r in results:
        status = "[yellow]skipped[/yellow]" if r.skipped else "[green]ok[/green]"
        t.add_row(
            r.table,
            r.mode.value,
            str(r.rows_written),
            str(r.input_snapshot_id) if r.input_snapshot_id else "-",
            str(r.output_snapshot_id) if r.output_snapshot_id else "-",
            str(r.duration_ms),
            status,
        )
    console.print(t)


@app.command()
def refresh(
    table: str = typer.Argument(..., help="Fully-qualified DIT name to refresh"),
    config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", "-c"),
):
    """Refresh a single DIT."""
    pipeline = Pipeline.from_config(config)
    r = pipeline.refresh_one(table)
    if r.skipped:
        console.print(f"[yellow]Skipped[/yellow] {r.table}: {r.skip_reason}")
    else:
        console.print(
            f"[green]Refreshed[/green] {r.table} — {r.rows_written} rows, "
            f"snapshot {r.output_snapshot_id} ({r.duration_ms} ms)"
        )


@app.command()
def status(config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", "-c")):
    """Show DAG status and refresh health."""
    pipeline = Pipeline.from_config(config)
    rows = pipeline.status()

    t = RichTable(title="DIT Status")
    t.add_column("Table")
    t.add_column("Exists")
    t.add_column("Mode")
    t.add_column("Lag")
    t.add_column("Snapshot", justify="right")
    t.add_column("Checkpoint", justify="right")
    t.add_column("Upstream")

    for r in rows:
        t.add_row(
            r["name"],
            "yes" if r["exists"] else "no",
            r["refresh_mode"],
            r["lag"],
            str(r["current_snapshot"]) if r["current_snapshot"] else "-",
            str(r["checkpoint"]) if r["checkpoint"] else "-",
            ", ".join(r["upstream"]) or "-",
        )
    console.print(t)


@app.command()
def dag(config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", "-c")):
    """Print the dependency DAG."""
    pipeline = Pipeline.from_config(config)
    console.print(pipeline.planner.render_ascii())


@app.command()
def watch(
    config: Path = typer.Option(Path(DEFAULT_CONFIG), "--config", "-c"),
    poll_interval: float = typer.Option(10.0, "--poll", help="Seconds between polls"),
    quiet_period: float = typer.Option(
        2.0, "--quiet", help="Debounce seconds after detecting a change"
    ),
    ui: bool = typer.Option(
        True, "--ui/--no-ui", help="Show the Rich live dashboard (default) or plain logs"
    ),
):
    """Watch upstream Iceberg tables and keep dependent DITs fresh.

    With `trigger: poll` (default) this polls the catalog for new snapshots on
    every external source; when one changes, it refreshes all DITs that depend on
    it (transitively) in topological order, using the configured engine. With
    `trigger: push` (requires `engine: flink`) it instead submits one long-running
    Flink streaming job per streamable table that reacts to upstream commits
    continuously. Press Ctrl+C to stop.

    The default UI is a live Rich dashboard showing the DAG, per-table status,
    and a rolling event log. Use --no-ui for plain log-only output (useful
    in headless environments).
    """
    from hyouka.watcher import WatchConfig, Watcher

    pipeline = Pipeline.from_config(config)

    if pipeline.config.compute.trigger == "push":
        _run_push(pipeline)
        return

    sources = pipeline.planner.external_sources()
    if not sources:
        console.print("[yellow]No external upstream sources found in this pipeline.[/yellow]")
        raise typer.Exit(1)

    cfg = WatchConfig(
        poll_interval_seconds=poll_interval,
        quiet_period_seconds=quiet_period,
    )
    watcher = Watcher(pipeline, cfg)

    if ui:
        from hyouka.dashboard import run_dashboard

        run_dashboard(pipeline, watcher, poll_interval)
    else:
        console.print(
            f"[bold]Watching[/bold] {len(sources)} upstream source(s) "
            f"(engine: {pipeline.config.compute.engine}, trigger: poll):"
        )
        for s in sources:
            console.print(f"  - {s}")
        console.print(f"Polling every {poll_interval}s. Press Ctrl+C to stop.")
        console.print()
        watcher.run()


def _run_push(pipeline: Pipeline) -> None:
    """Start one continuous Flink streaming job per streamable DIT (push trigger).

    Submits the jobs to the Flink cluster, reports which tables stream and which
    are skipped (not yet supported for streaming), then stays alive until
    interrupted. The streaming jobs run on the cluster independently of this
    process, so stopping the command leaves them running.
    """
    import signal
    import threading

    from hyouka.flink_executor import StreamHandle, StreamingNotSupportedError

    executor = pipeline.executor
    if not hasattr(executor, "start_stream"):
        console.print(
            "[red]trigger: push requires engine: flink.[/red] "
            "Set compute.engine to flink (and start the `flink` Compose profile)."
        )
        raise typer.Exit(1)

    console.print(
        "[bold]Starting streaming push[/bold] (engine: flink). Submitting one "
        "continuous Flink job per streamable table:"
    )
    started: list[StreamHandle] = []
    for name in pipeline.planner.topological_order():
        dit = pipeline.dits[name]
        try:
            handle = executor.start_stream(dit)
        except StreamingNotSupportedError as exc:
            console.print(f"  [yellow]skip[/yellow] {name}: {exc}")
            continue
        started.append(handle)
        console.print(f"  [green]streaming[/green] {name}  ->  Flink job {handle.job_id}")

    if not started:
        console.print(
            "\n[yellow]No streamable tables found.[/yellow] Streaming push currently "
            "supports single-source, append-only transforms; aggregations and joins use "
            "trigger: poll (or the duckdb/flink batch engine)."
        )
        raise typer.Exit(1)

    console.print(
        f"\n[bold]{len(started)} streaming job(s) running[/bold] on the Flink cluster "
        "(dashboard: http://localhost:8081). They keep running on the cluster; press "
        "Ctrl+C to stop watching (the jobs continue)."
    )
    stop = threading.Event()

    def _handle(_signum, _frame):
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):
            pass
    try:
        while not stop.wait(5):
            pass
    except KeyboardInterrupt:
        pass
    console.print(
        "Stopped watching. Streaming jobs remain on the cluster; cancel them from the "
        "Flink dashboard (http://localhost:8081) to stop them."
    )


@app.command()
def version():
    """Print the Hyouka version."""
    from hyouka import __version__

    console.print(f"hyouka {__version__}")


if __name__ == "__main__":
    app()
