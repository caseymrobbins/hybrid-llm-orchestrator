# src/cli.py

import asyncio
import click
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Assuming orchestrator and metrics_collector are accessible
# from.orchestrator import Orchestrator 
# from.api_server import metrics_collector

# This is a mock for demonstration; replace with actual imports
from.orchestrator import Orchestrator, metrics_collector 

console = Console()

@click.group()
def cli():
    """Hybrid LLM Orchestrator CLI"""
    pass

@cli.command()
@click.argument('query')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed execution')
@click.option('--json-output', '-j', is_flag=True, help='Output as JSON')
@click.option('--profile', '-p', default=None, help='User profile ID')
def query(query, verbose, json_output, profile):
    """Process a query through the orchestrator"""
    
    async def run_query():
        orchestrator = Orchestrator(config_path="configs")
        
        if not json_output:
            console.print(f"[bold blue]Processing:[/bold blue] {query}\n")
        
        response = None
        module_outputs = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            if verbose and not json_output:
                # Show module execution in real-time
                modules = orchestrator.config_loader.workflow_config.execution_plan
                task = progress.add_task("Initializing...", total=len(modules))
                
                context = {'query': query, 'user_id': profile or 1}
                for step in modules:
                    module_name = step['module']
                    progress.update(task, description=f"Executing {module_name}...")
                    
                    # Execute module
                    output = await orchestrator._execute_module(module_name, context)
                    context[module_name] = output
                    
                    progress.update(task, advance=1)
                    
                    if verbose:
                        output_str = str(output)
                        console.print(f"  ✓ {module_name}: [dim]{output_str[:70]}...[/dim]")
                response = "Workflow complete. Final synthesis would occur here."
            
            else:
                # Simple execution
                response = await orchestrator.execute_workflow(query)
        
        if json_output:
            result = {
                'query': query,
                'response': response,
                'modules': context
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print("\n[bold green]Response:[/bold green]")
            console.print(Panel(response))

    asyncio.run(run_query())


@cli.command()
def health():
    """Check system health"""
    async def run_health_check():
        orchestrator = Orchestrator(config_path="configs")
        health_status = await orchestrator.health_check()
        
        table = Table(title="System Health Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        
        overall = health_status['overall_status']
        status_color = "green" if overall == "ok" else "yellow" if overall == "degraded" else "red"
        table.add_row("Overall Status", f"[{status_color}]{overall.upper()}[/{status_color}]")
        
        db_status = health_status['database']['status']
        table.add_row("Database", f"[green]✓[/green] {db_status}" if db_status == "ok" else f"[red]✗[/red] {db_status}")
        
        for api, status in health_status['external_apis'].items():
            status_icon = "[green]✓[/green]" if status == "ok" else "[red]✗[/red]"
            table.add_row(f"API: {api}", f"{status_icon} {status}")
        
        for breaker, state in health_status['circuit_breakers'].items():
            state_color = "green" if state == "closed" else "yellow" if state == "half_open" else "red"
            table.add_row(f"Circuit: {breaker}", f"[{state_color}]{state}[/{state_color}]")
        
        console.print(table)

    asyncio.run(run_health_check())

@cli.command()
def metrics():
    """Display system metrics"""
    metrics = metrics_collector.get_report()
    
    console.print("[bold]System Metrics[/bold]\n")
    
    console.print(f"📊 Total Queries: {metrics.get('total_requests', 0)}")
    console.print(f"⚡ Latency P50: {metrics.get('latency_p50_ms', 0):.2f}ms")
    console.print(f"⚡ Latency P99: {metrics.get('latency_p99_ms', 0):.2f}ms")
    console.print(f"💾 Cache Hit Rate: {metrics.get('cache_hit_rate', 0)*100:.1f}%")
    console.print(f"💰 Total Cost: ${metrics.get('total_cost_dollars', 0):.2f}")
    
    local, external = metrics.get('local_vs_external_ratio', '0:0').split(':')
    total = int(local) + int(external)
    local_pct = (int(local) / total * 100) if total > 0 else 0
    external_pct = (int(external) / total * 100) if total > 0 else 0
    
    console.print(f"🖥️  Local Processing: {local_pct:.1f}%")
    console.print(f"☁️  External Processing: {external_pct:.1f}%")

@cli.command()
def modules():
    """List configured modules"""
    orchestrator = Orchestrator(config_path="configs")
    
    table = Table(title="Configured Modules")
    table.add_column("Module", style="cyan")
    table.add_column("Model Strategy")
    table.add_column("Fallback")
    
    for name, config in orchestrator.config_loader.module_configs.items():
        model = config.llm_address or "Router-Managed"
        fallback = "✓" if config.fallback else "-"
        table.add_row(name, model, fallback)
    
    console.print(table)

@cli.command()
@click.option('--host', default='0.0.0.0', help='API host')
@click.option('--port', default=8000, help='API port')
def serve(host, port):
    """Start the API server"""
    import uvicorn
    console.print(f"[bold green]Starting API server on {host}:{port}[/bold green]")
    uvicorn.run("src.api_server:app", host=host, port=port, reload=True)

@cli.command()
def ui():
    """Launch the web UI"""
    import subprocess
    import webbrowser
    import time
    
    console.print("[bold blue]Starting Web UI...[/bold blue]")
    
    api_process = subprocess.Popen(['python', '-m', 'src.cli', 'serve'])
    ui_process = subprocess.Popen(['npm', 'start'], cwd='ui')
    
    time.sleep(3)
    webbrowser.open('http://localhost:3000')
    
    console.print("[bold green]UI launched at http://localhost:3000[/bold green]")
    console.print("Press Ctrl+C to stop all services")
    
    try:
        api_process.wait()
    except KeyboardInterrupt:
        api_process.terminate()
        ui_process.terminate()

if __name__ == "__main__":
    cli()