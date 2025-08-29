#!/usr/bin/env python3
"""
Real-time monitoring dashboard for ML trading system.

This dashboard provides live monitoring of:
- Data ingestion rates
- Feature computation latency
- Model inference performance
- Signal generation metrics
- System health indicators
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.monitoring._config import DashboardConfig

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class SystemMonitor:
    """Monitors system metrics and health."""

    def __init__(self, config: DashboardConfig | None = None) -> None:
        """Initialize system monitor."""
        self.console = Console()
        self.metrics: dict[str, Any] = {}
        self.alerts: list[dict[str, Any]] = []
        self.last_update = datetime.now()
        self.config: DashboardConfig = config or DashboardConfig()

    def update_metrics(self) -> None:
        """Update system metrics."""
        # Data metrics
        self.metrics["data"] = self._get_data_metrics()

        # Feature metrics
        self.metrics["features"] = self._get_feature_metrics()

        # Model metrics
        self.metrics["models"] = self._get_model_metrics()

        # System metrics
        self.metrics["system"] = self._get_system_metrics()

        self.last_update = datetime.now()

    def _get_data_metrics(self) -> dict[str, Any]:
        """Get data ingestion metrics."""
        metrics: dict[str, Any] = {
            "l0_symbols": 0,
            "l1_symbols": 0,
            "l2_symbols": 0,
            "total_size_gb": 0.0,
            "ingestion_rate": 0.0,
        }

        # Check L0 data
        data_dir = Path(self.config.data_dir)
        if data_dir.exists():
            l0_dirs = list(data_dir.glob("*/"))
            metrics["l0_symbols"] = len(l0_dirs)

            # Check L1 data
            l1_dirs = list(data_dir.glob("*/l1/"))
            metrics["l1_symbols"] = len(l1_dirs)

            # Calculate total size
            total_size = 0
            for symbol_dir in l0_dirs:
                for file in symbol_dir.rglob("*.parquet"):
                    total_size += file.stat().st_size
            metrics["total_size_gb"] = total_size / (1024**3)

        # Check ingestion progress
        progress_file = Path(self.config.l1_progress_file)
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
                stats = progress.get("stats", {})

                # Calculate rate
                last_update = progress.get("last_update")
                if last_update:
                    last_dt = datetime.fromisoformat(last_update)
                    elapsed = (datetime.now() - last_dt).total_seconds()
                    if elapsed > 0:
                        completed = stats.get("completed_bbo", 0)
                        metrics["ingestion_rate"] = completed / (elapsed / 3600)  # symbols/hour

        return metrics

    def _get_feature_metrics(self) -> dict[str, Any]:
        """Get feature computation metrics."""
        metrics: dict[str, Any] = {
            "computed_symbols": 0,
            "total_features": 26,
            "avg_latency_ms": 0.0,
            "cache_hit_rate": 0.0,
        }

        # Check feature progress
        progress_file = Path(self.config.feature_progress_file)
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
                metrics["computed_symbols"] = len(progress.get("completed", []))

        # Placeholder for real metrics (would come from Prometheus)
        metrics["avg_latency_ms"] = 2.3
        metrics["cache_hit_rate"] = 0.85

        return metrics

    def _get_model_metrics(self) -> dict[str, Any]:
        """Get model inference metrics."""
        return {
            "models_loaded": 3,
            "avg_inference_ms": 1.2,
            "predictions_per_sec": 450,
            "signal_rate": 0.023,
            "accuracy": 0.547  # Placeholder
        }

    def _get_system_metrics(self) -> dict[str, Any]:
        """Get system health metrics."""
        import psutil

        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "network_mbps": 0,  # Placeholder
            "postgres_connections": 5,  # Placeholder
            "redis_ops_per_sec": 1200  # Placeholder
        }

    def check_alerts(self) -> None:
        """Check for system alerts."""
        self.alerts.clear()

        # Check system resources
        if self.metrics.get("system", {}).get("cpu_percent", 0) > 80:
            self.alerts.append({
                "level": "WARNING",
                "message": "High CPU usage",
                "value": f"{self.metrics['system']['cpu_percent']:.1f}%"
            })

        if self.metrics.get("system", {}).get("memory_percent", 0) > 85:
            self.alerts.append({
                "level": "WARNING",
                "message": "High memory usage",
                "value": f"{self.metrics['system']['memory_percent']:.1f}%"
            })

        # Check data ingestion
        if self.metrics.get("data", {}).get("ingestion_rate", 0) < 1:
            self.alerts.append({
                "level": "INFO",
                "message": "Slow data ingestion",
                "value": f"{self.metrics['data']['ingestion_rate']:.2f} symbols/hour"
            })

        # Check model performance
        if self.metrics.get("models", {}).get("avg_inference_ms", 0) > 5:
            self.alerts.append({
                "level": "WARNING",
                "message": "High inference latency",
                "value": f"{self.metrics['models']['avg_inference_ms']:.1f}ms"
            })


class DashboardUI:
    """Rich terminal UI for monitoring dashboard."""

    def __init__(self, monitor: SystemMonitor) -> None:
        """Initialize dashboard UI."""
        self.monitor = monitor
        self.console = Console()

    def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()

        # Split into header and body
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=4)
        )

        # Split body into columns
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Split left column
        layout["left"].split_column(
            Layout(name="data_panel"),
            Layout(name="feature_panel")
        )

        # Split right column
        layout["right"].split_column(
            Layout(name="model_panel"),
            Layout(name="system_panel")
        )

        return layout

    def render_header(self) -> Panel:
        """Render header panel."""
        header = Text("🚀 NAUTILUS TRADER ML - REAL-TIME MONITOR", style="bold cyan")
        subheader = Text(f"Last Update: {self.monitor.last_update.strftime('%H:%M:%S')}", style="dim")

        content = Text()
        content.append(header)
        content.append("\n")
        content.append(subheader)

        return Panel(content, style="cyan")

    def render_data_panel(self) -> Panel:
        """Render data metrics panel."""
        data = self.monitor.metrics.get("data", {})

        table = Table(show_header=False, box=None)
        table.add_column("Metric")
        table.add_column("Value", style="cyan")

        table.add_row("L0 Symbols:", f"{data.get('l0_symbols', 0)}/78")
        table.add_row("L1 Symbols:", f"{data.get('l1_symbols', 0)}/78")
        table.add_row("L2 Symbols:", f"{data.get('l2_symbols', 0)}/78")
        table.add_row("Total Size:", f"{data.get('total_size_gb', 0):.2f} GB")
        table.add_row("Ingestion Rate:", f"{data.get('ingestion_rate', 0):.1f} sym/hr")

        return Panel(table, title="📊 Data Ingestion", border_style="blue")

    def render_feature_panel(self) -> Panel:
        """Render feature metrics panel."""
        features = self.monitor.metrics.get("features", {})

        table = Table(show_header=False, box=None)
        table.add_column("Metric")
        table.add_column("Value", style="green")

        table.add_row("Computed:", f"{features.get('computed_symbols', 0)}/77")
        table.add_row("Features:", f"{features.get('total_features', 0)}")
        table.add_row("Avg Latency:", f"{features.get('avg_latency_ms', 0):.1f}ms")
        table.add_row("Cache Hit:", f"{features.get('cache_hit_rate', 0)*100:.1f}%")

        return Panel(table, title="🔬 Features", border_style="green")

    def render_model_panel(self) -> Panel:
        """Render model metrics panel."""
        models = self.monitor.metrics.get("models", {})

        table = Table(show_header=False, box=None)
        table.add_column("Metric")
        table.add_column("Value", style="yellow")

        table.add_row("Models:", f"{models.get('models_loaded', 0)}")
        table.add_row("Inference:", f"{models.get('avg_inference_ms', 0):.1f}ms")
        table.add_row("Predictions/s:", f"{models.get('predictions_per_sec', 0)}")
        table.add_row("Signal Rate:", f"{models.get('signal_rate', 0)*100:.2f}%")
        table.add_row("Accuracy:", f"{models.get('accuracy', 0)*100:.1f}%")

        return Panel(table, title="🤖 Models", border_style="yellow")

    def render_system_panel(self) -> Panel:
        """Render system metrics panel."""
        system = self.monitor.metrics.get("system", {})

        table = Table(show_header=False, box=None)
        table.add_column("Metric")
        table.add_column("Value", style="magenta")

        cpu = system.get("cpu_percent", 0)
        mem = system.get("memory_percent", 0)

        # Color code based on thresholds
        cpu_style = "red" if cpu > 80 else "yellow" if cpu > 60 else "green"
        mem_style = "red" if mem > 85 else "yellow" if mem > 70 else "green"

        table.add_row("CPU:", f"[{cpu_style}]{cpu:.1f}%[/]")
        table.add_row("Memory:", f"[{mem_style}]{mem:.1f}%[/]")
        table.add_row("Disk:", f"{system.get('disk_usage_percent', 0):.1f}%")
        table.add_row("Network:", f"{system.get('network_mbps', 0):.1f} Mbps")
        table.add_row("Postgres:", f"{system.get('postgres_connections', 0)} conn")
        table.add_row("Redis:", f"{system.get('redis_ops_per_sec', 0)} ops/s")

        return Panel(table, title="💻 System", border_style="magenta")

    def render_footer(self) -> Panel:
        """Render alerts/status footer."""
        if self.monitor.alerts:
            # Show alerts
            table = Table(show_header=True, box=None)
            table.add_column("Level", style="bold")
            table.add_column("Alert")
            table.add_column("Value")

            for alert in self.monitor.alerts[:3]:  # Show max 3 alerts
                level_style = "red" if alert["level"] == "ERROR" else "yellow" if alert["level"] == "WARNING" else "blue"
                table.add_row(
                    f"[{level_style}]{alert['level']}[/]",
                    alert["message"],
                    alert["value"]
                )

            return Panel(table, title="⚠️ Alerts", border_style="yellow")
        else:
            status = Text("✅ All systems operational", style="green")
            return Panel(status, title="Status", border_style="green")

    def render(self, layout: Layout) -> None:
        """Render all panels."""
        layout["header"].update(self.render_header())
        layout["data_panel"].update(self.render_data_panel())
        layout["feature_panel"].update(self.render_feature_panel())
        layout["model_panel"].update(self.render_model_panel())
        layout["system_panel"].update(self.render_system_panel())
        layout["footer"].update(self.render_footer())


async def run_dashboard() -> None:
    """Run the monitoring dashboard."""
    monitor = SystemMonitor()
    ui = DashboardUI(monitor)

    # Create layout
    layout = ui.create_layout()

    with Live(layout, refresh_per_second=1, screen=True) as live:
        while True:
            # Update metrics
            monitor.update_metrics()
            monitor.check_alerts()

            # Render UI
            ui.render(layout)

            # Update display
            live.update(layout)

            # Wait before next update
            await asyncio.sleep(1)


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time ML monitoring dashboard")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    args = parser.parse_args()

    if args.web:
        print("Web interface not yet implemented")
    else:
        # Run terminal dashboard
        try:
            asyncio.run(run_dashboard())
        except KeyboardInterrupt:
            print("\n✋ Dashboard stopped")


if __name__ == "__main__":
    main()
