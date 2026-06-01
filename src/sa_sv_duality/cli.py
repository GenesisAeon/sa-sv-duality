"""S_A / S_V Duality CLI — GenesisAeon Package 36."""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .benchmark import run_benchmarks
from .constants import GAMMA_UNIVERSAL, SIGMA_UTAC
from .system import SAVDuality

app = typer.Typer(
    name="sav",
    help="S_A / S_V Entropy Duality — GenesisAeon Package 36",
    rich_markup_mode="rich",
)
console = Console()
err = Console(stderr=True)


@app.command()
def run(
    gamma: Annotated[float, typer.Option(help="CREP Γ value")] = GAMMA_UNIVERSAL,
    sigma: Annotated[float, typer.Option(help="UTAC sigma parameter")] = SIGMA_UTAC,
    duration: Annotated[float, typer.Option(help="Integration duration (ODE time units)")] = 10.0,
    json_out: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """Run a full S_A / S_V duality cycle."""
    system = SAVDuality(sigma=sigma, gamma=gamma)
    result = system.run_cycle(duration_years=duration)

    if json_out:
        console.print_json(json.dumps(result, default=str))
        return

    console.print(
        Panel(
            f"[bold cyan]S_A / S_V Duality[/bold cyan] · GenesisAeon P36  v{__version__}\n"
            f"Γ = {gamma:.4f}  σ = {sigma:.2f}  T = {duration}",
            expand=False,
        )
    )
    table = Table(show_header=True, box=None)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("S_A (action entropy)",   f"{result['S_A']:.6f}")
    table.add_row("S_V (volume entropy)",   f"{result['S_V']:.6f}")
    table.add_row("S_A · S_V",             f"{result['duality_constant']:.6f}")
    table.add_row("λ_opt",                  f"{result['lambda_opt']:.4f}")
    table.add_row("Conserved",              "✓" if result['conserved'] else "✗")
    table.add_row("Phase events",           str(result['n_phase_events']))
    table.add_row("Runtime (s)",            f"{result['duration_s']:.3f}")
    console.print(table)


@app.command(name="q4-map")
def q4_map(
    gamma: Annotated[float, typer.Option(help="CREP Γ value")] = GAMMA_UNIVERSAL,
    json_out: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
) -> None:
    """Show S_A and S_V for all 16 Q4 states."""
    from .q4_sav_map import Q4SAVMap

    q4 = Q4SAVMap()
    rows = q4.summary_table()

    if json_out:
        console.print_json(json.dumps(rows))
        return

    table = Table(title="Q4 State Entropy Map — Package 36", show_lines=True)
    table.add_column("ID",     justify="right", style="dim")
    table.add_column("State",  style="bold cyan")
    table.add_column("S_A",    justify="right")
    table.add_column("S_V",    justify="right")
    table.add_column("S_A·S_V", justify="right")
    table.add_column("Γ",      justify="right")
    table.add_column("Active", justify="center")

    for row in rows:
        table.add_row(
            str(row["id"]),
            str(row["label"]),
            f"{row['S_A']:.4f}",
            f"{row['S_V']:.4f}",
            f"{row['S_A*S_V']:.4f}",
            f"{row['gamma']:.4f}",
            str(row["n_active"]),
        )
    console.print(table)


@app.command()
def route(
    source: Annotated[int, typer.Argument(help="Start Q4 state (0-15)")] = 0,
    target: Annotated[int, typer.Argument(help="Target Q4 state (0-15)")] = 15,
    lambda_: Annotated[float, typer.Option("--lambda", help="Lagrange multiplier")] = 1.0,
) -> None:
    """Find the optimal S_A-minimising path through Q4 state space."""
    if not (0 <= source <= 15 and 0 <= target <= 15):
        err.print("[red]States must be in [0, 15].[/red]")
        raise typer.Exit(1)

    from .network_routing import SAVRouter

    router = SAVRouter.from_q4_adjacency(lambda_=lambda_)
    result = router.route(source, target)

    if result is None:
        err.print(f"[red]No path found from {source} to {target}.[/red]")
        raise typer.Exit(1)

    path_labels = [format(n, "04b") for n in result.path]
    console.print(
        Panel(
            f"Optimal path [bold]{source}[/bold] → [bold]{target}[/bold]  "
            f"(λ = {lambda_})\n"
            f"[cyan]{' → '.join(path_labels)}[/cyan]\n"
            f"S_A = {result.total_S_A:.4f}   "
            f"S_V = {result.total_S_V:.4f}   "
            f"Hops = {result.hops}",
            title="S_A/S_V Router",
            expand=False,
        )
    )

    improvement = router.improvement_vs_dijkstra(source, target)
    if improvement > 0:
        console.print(f"[green]SAV routing improved S_A by {improvement:.1%} vs. Dijkstra.[/green]")
    else:
        console.print("[dim]SAV routing equivalent to Dijkstra for this pair.[/dim]")


@app.command()
def benchmark(
    fast: Annotated[bool, typer.Option("--fast", help="Skip slow tests")] = False,
) -> None:
    """Run all S_A / S_V benchmark tests."""
    console.print("[bold]Running S_A/S_V benchmarks...[/bold]")
    report = run_benchmarks(fast=fast)
    console.print(report.summary())

    for p in report.passed:
        console.print(f"  [green]✔[/green] {p}")
    for f in report.failed:
        console.print(f"  [red]✘[/red] {f}")
    for s in report.skipped:
        console.print(f"  [yellow]⊘[/yellow] {s}")

    if not report.ok:
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"sa-sv-duality [bold]{__version__}[/bold]  (GenesisAeon P36)")


if __name__ == "__main__":
    app()
