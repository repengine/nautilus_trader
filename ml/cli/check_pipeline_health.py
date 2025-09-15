#!/usr/bin/env python3
"""
ML Pipeline Health Check Script.

Queries health monitoring views and generates comprehensive health reports.
Supports both human-readable and JSON output for integration with monitoring systems.

Usage:
    python check_pipeline_health.py                    # Human-readable output
    python check_pipeline_health.py --json            # JSON output for dashboards
    python check_pipeline_health.py --critical-only   # Show only critical issues
    python check_pipeline_health.py --export report.json  # Export to file

"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, cast


try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    print(
        "Warning: psycopg2 not installed. Install with: pip install psycopg2-binary",
        file=sys.stderr,
    )

try:
    from tabulate import tabulate  # type: ignore[import-untyped]

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

    # Fallback implementation for tabulate
    def tabulate(data: list[list[Any]], headers: list[str], tablefmt: str = "simple") -> str:
        """
        Simple fallback table formatter.
        """
        _ = tablefmt
        if not data:
            return ""

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in data:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Format header
        lines = []
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Format data rows
        for row in data:
            row_line = " | ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
            lines.append(row_line)

        return "\n".join(lines)


# Health status levels
class HealthStatus(Enum):
    """
    Health status levels for pipeline components.
    """

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# Monitoring thresholds
class Thresholds:
    """
    Configurable thresholds for health checks.
    """

    DATA_STALENESS_WARNING = 3600  # 1 hour in seconds
    DATA_STALENESS_CRITICAL = 86400  # 24 hours in seconds
    ERROR_RATE_WARNING = 0.05  # 5% error rate
    ERROR_RATE_CRITICAL = 0.10  # 10% error rate
    MIN_INSTRUMENTS = 1  # Minimum instruments to process
    MIN_CONFIDENCE = 0.6  # Minimum model confidence
    INFERENCE_LATENCY_WARNING = 500  # ms
    INFERENCE_LATENCY_CRITICAL = 1000  # ms
    FEATURE_COMPUTATION_WARNING = 200  # ms
    FEATURE_COMPUTATION_CRITICAL = 500  # ms
    NULL_RATE_WARNING = 0.05  # 5% null values
    NULL_RATE_CRITICAL = 0.10  # 10% null values


@dataclass
class ComponentHealth:
    """
    Health status for a pipeline component.
    """

    name: str
    status: HealthStatus
    message: str
    metrics: dict[str, Any]
    last_update: datetime | None = None
    issues: list[str] | None = None

    def __post_init__(self) -> None:
        """
        Initialize issues list if not provided.
        """
        if self.issues is None:
            self.issues = []


class PipelineHealthChecker:
    """
    ML Pipeline health checker.
    """

    def __init__(self, connection_string: str, thresholds: Thresholds | None = None) -> None:
        """
        Initialize health checker.

        Args:
            connection_string: PostgreSQL connection string
            thresholds: Custom thresholds for health checks

        """
        self.connection_string = connection_string
        self.thresholds = thresholds or Thresholds()
        self._conn: psycopg2.connection | None = None

    def __enter__(self) -> "PipelineHealthChecker":
        """
        Context manager entry.
        """
        self.connect()
        return self

    def __exit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """
        Context manager exit.
        """
        self.disconnect()

    def connect(self) -> None:
        """
        Establish database connection.
        """
        try:
            self._conn = psycopg2.connect(self.connection_string)
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    def disconnect(self) -> None:
        """
        Close database connection.
        """
        if self._conn:
            self._conn.close()
            self._conn = None

    def _execute_query(self, query: str) -> list[dict[str, Any]]:
        """
        Execute a SQL query and return results.

        Args:
            query: SQL query to execute

        Returns:
            List of result dictionaries

        """
        if not self._conn:
            raise RuntimeError("Not connected to database")

        try:
            with self._conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query)
                results = cast(list[dict[str, Any]], cursor.fetchall())
                return results
        except psycopg2.Error as e:
            raise RuntimeError(f"Query execution failed: {e}")

    def check_pipeline_health(self) -> ComponentHealth:
        """
        Check overall pipeline health.

        Returns:
            ComponentHealth object with pipeline status

        """
        query = """
        SELECT * FROM ml.pipeline_health
        WHERE date >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY date DESC
        LIMIT 7
        """

        results = self._execute_query(query)

        if not results:
            return ComponentHealth(
                name="Pipeline Overall",
                status=HealthStatus.UNKNOWN,
                message="No pipeline data available",
                metrics={},
            )

        latest = results[0]
        staleness = float(latest.get("staleness_seconds", 0))
        health_score = float(latest.get("health_score", 0))
        instruments = int(latest.get("instruments_processed", 0))

        # Determine status
        issues = []
        if staleness > self.thresholds.DATA_STALENESS_CRITICAL:
            status = HealthStatus.CRITICAL
            issues.append(f"Data is {staleness/3600:.1f} hours stale")
        elif staleness > self.thresholds.DATA_STALENESS_WARNING:
            status = HealthStatus.WARNING
            issues.append(f"Data is {staleness/3600:.1f} hours stale")
        elif instruments < self.thresholds.MIN_INSTRUMENTS:
            status = HealthStatus.WARNING
            issues.append(f"Only {instruments} instruments being processed")
        else:
            status = HealthStatus.HEALTHY

        metrics = {
            "health_score": health_score,
            "instruments_processed": instruments,
            "total_features": int(latest.get("total_features", 0)),
            "staleness_seconds": staleness,
            "days_with_data": len(results),
        }

        return ComponentHealth(
            name="Pipeline Overall",
            status=status,
            message=f"Health score: {health_score}/100",
            metrics=metrics,
            last_update=latest.get("last_update_time"),
            issues=issues,
        )

    def check_data_collection(self) -> ComponentHealth:
        """
        Check data collection health.

        Returns:
            ComponentHealth object with collection status

        """
        query = """
        SELECT
            COUNT(DISTINCT instrument_id) as instruments,
            SUM(records_collected) as total_records,
            AVG(records_per_minute) as avg_rate,
            COUNT(CASE WHEN collection_status = 'gaps_detected' THEN 1 END) as gaps_count
        FROM ml.data_collection_stats
        WHERE hour >= NOW() - INTERVAL '1 hour'
        """

        results = self._execute_query(query)

        if not results or not results[0]["total_records"]:
            return ComponentHealth(
                name="Data Collection",
                status=HealthStatus.CRITICAL,
                message="No data collected in last hour",
                metrics={"total_records": 0},
            )

        data = results[0]
        gaps_count = int(data.get("gaps_count", 0))
        total_records = int(data.get("total_records", 0))

        issues = []
        if gaps_count > 0:
            status = HealthStatus.WARNING
            issues.append(f"{gaps_count} collection gaps detected")
        else:
            status = HealthStatus.HEALTHY

        metrics = {
            "instruments_active": int(data.get("instruments", 0)),
            "records_last_hour": total_records,
            "avg_records_per_minute": float(data.get("avg_rate", 0)),
            "gaps_detected": gaps_count,
        }

        return ComponentHealth(
            name="Data Collection",
            status=status,
            message=f"Collected {total_records} records from {metrics['instruments_active']} instruments",
            metrics=metrics,
            issues=issues,
        )

    def check_feature_computation(self) -> ComponentHealth:
        """
        Check feature computation health.

        Returns:
            ComponentHealth object with feature computation status

        """
        # Align to current schema: summarize from public.ml_feature_computation_stats table
        query = """
        SELECT
            COUNT(DISTINCT instrument_id) AS instruments,
            AVG(computation_time_ms) AS avg_latency_ms,
            MAX(computation_time_ms) AS max_latency_ms
        FROM public.ml_feature_computation_stats
        WHERE ns_to_timestamp(ts_event) >= DATE_TRUNC('day', NOW())
        """

        try:
            results = self._execute_query(query)
        except Exception:
            results = []

        if not results or not results[0].get("instruments"):
            return ComponentHealth(
                name="Feature Computation",
                status=HealthStatus.WARNING,
                message="No feature computations today",
                metrics={},
            )

        data = results[0]
        avg_latency = float(data.get("avg_latency_ms", 0))
        max_latency = float(data.get("max_latency_ms", 0))
        critical_count = 0
        warning_count = 0

        issues = []
        if critical_count > 0:
            status = HealthStatus.CRITICAL
            issues.append(f"{critical_count} components with critical latency")
        elif max_latency > self.thresholds.FEATURE_COMPUTATION_CRITICAL:
            status = HealthStatus.CRITICAL
            issues.append(f"Max P95 latency: {max_latency:.0f}ms")
        elif warning_count > 0 or avg_latency > self.thresholds.FEATURE_COMPUTATION_WARNING:
            status = HealthStatus.WARNING
            issues.append(f"{warning_count} components with warning latency")
        else:
            status = HealthStatus.HEALTHY

        metrics = {
            "instruments": int(data.get("instruments", 0)),
            "avg_p95_latency_ms": avg_latency,
            "max_p95_latency_ms": max_latency,
            "critical_components": critical_count,
            "warning_components": warning_count,
        }

        return ComponentHealth(
            name="Feature Computation",
            status=status,
            message=f"Avg P95 latency: {avg_latency:.0f}ms",
            metrics=metrics,
            issues=issues,
        )

    def check_data_freshness(self) -> ComponentHealth:
        """
        Check data freshness across instruments.

        Returns:
            ComponentHealth object with freshness status

        """
        # Align to current schema: derive freshness from latest ts_init in feature values
        query = """
        SELECT instrument_id, MAX(ts_init) AS last_update_ns
        FROM public.ml_feature_values
        GROUP BY instrument_id
        """

        try:
            rows = self._execute_query(query)
        except Exception:
            rows = []

        if not rows:
            return ComponentHealth(
                name="Data Freshness",
                status=HealthStatus.WARNING,
                message="No feature data available",
                metrics={"total_instruments": 0, "fresh_count": 0},
            )

        critical_count = 0
        warning_count = 0
        no_data_count = 0
        max_staleness = 0.0
        fresh_count = 0
        stale_count = 0

        from ml.common.timestamps import sanitize_timestamp_ns as _sanitize
        now_ns = _sanitize(int(datetime.now().timestamp() * 1e9), context="cli.check_pipeline_health:now")
        for row in rows:
            last_ns = int(row.get("last_update_ns", 0))
            if last_ns <= 0:
                no_data_count += 1
                continue
            staleness = max(0.0, (now_ns - last_ns) / 1e9)
            max_staleness = max(max_staleness, staleness)
            if staleness > self.thresholds.DATA_STALENESS_CRITICAL:
                critical_count += 1
            elif staleness > self.thresholds.DATA_STALENESS_WARNING:
                warning_count += 1
                stale_count += 1
            else:
                fresh_count += 1

        status = (
            HealthStatus.CRITICAL
            if critical_count > 0
            else (
                HealthStatus.WARNING
                if warning_count > 0 or no_data_count > 0
                else HealthStatus.HEALTHY
            )
        )

        metrics = {
            "total_instruments": len(rows),
            "fresh_count": fresh_count,
            "delayed_count": stale_count,
            "warning_count": warning_count,
            "critical_count": critical_count,
            "no_data_count": no_data_count,
            "max_staleness_hours": max_staleness / 3600,
        }

        return ComponentHealth(
            name="Data Freshness",
            status=status,
            message=f"{metrics['fresh_count']}/{metrics['total_instruments']} instruments fresh",
            metrics=metrics,
            issues=[],
        )

    def check_errors(self) -> ComponentHealth:
        """
        Check for errors and issues.

        Returns:
            ComponentHealth object with error status

        """
        # Align to current schema: summarize failures from public.ml_data_events
        query = """
        SELECT
            COALESCE(error, 'unknown') AS error_type,
            COUNT(*) AS total_errors,
            COUNT(DISTINCT dataset_id) AS affected_components
        FROM public.ml_data_events
        WHERE status = 'failed'
          AND ts_event > (EXTRACT(EPOCH FROM (NOW() - INTERVAL '7 days')) * 1000000000)::BIGINT
        GROUP BY COALESCE(error, 'unknown')
        """

        try:
            results = self._execute_query(query)
        except Exception:
            results = []

        if not results:
            return ComponentHealth(
                name="Error Monitoring",
                status=HealthStatus.HEALTHY,
                message="No errors detected today",
                metrics={"total_errors": 0},
            )

        total_errors = sum(int(r.get("total_errors", 0)) for r in results)
        max_errors = max(int(r.get("total_errors", 0)) for r in results)

        issues = []
        error_breakdown = {}
        for row in results:
            error_type = row["error_type"]
            count = int(row["total_errors"])
            error_breakdown[f"{error_type}_errors"] = count
            if count > 100:
                issues.append(f"{count} {error_type} errors")

        if max_errors > 100:
            status = HealthStatus.CRITICAL
        elif total_errors > 50:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        metrics = {
            "total_errors": total_errors,
            "affected_components": sum(int(r.get("affected_components", 0)) for r in results),
            "max_errors_per_type": max_errors,
            **error_breakdown,
        }

        return ComponentHealth(
            name="Error Monitoring",
            status=status,
            message=f"{total_errors} total errors across {metrics['affected_components']} components",
            metrics=metrics,
            issues=issues,
        )

    def check_model_performance(self) -> ComponentHealth:
        """
        Check model inference performance.

        Returns:
            ComponentHealth object with model performance status

        """
        query = """
        SELECT
            COUNT(DISTINCT model_id) as model_count,
            AVG(avg_confidence) as overall_confidence,
            AVG(p99_inference_ms) as avg_p99_latency,
            MAX(p99_inference_ms) as max_p99_latency,
            COUNT(CASE WHEN health_status != 'healthy' THEN 1 END) as unhealthy_count
        FROM ml.model_performance_summary
        WHERE date >= CURRENT_DATE
        """

        results = self._execute_query(query)

        if not results or not results[0]["model_count"]:
            return ComponentHealth(
                name="Model Performance",
                status=HealthStatus.WARNING,
                message="No model predictions today",
                metrics={},
            )

        data = results[0]
        confidence = float(data.get("overall_confidence", 0))
        max_latency = float(data.get("max_p99_latency", 0))
        unhealthy = int(data.get("unhealthy_count", 0))

        issues = []
        if confidence < self.thresholds.MIN_CONFIDENCE:
            status = HealthStatus.WARNING
            issues.append(f"Low average confidence: {confidence:.2f}")
        elif max_latency > self.thresholds.INFERENCE_LATENCY_CRITICAL:
            status = HealthStatus.CRITICAL
            issues.append(f"High P99 latency: {max_latency:.0f}ms")
        elif unhealthy > 0:
            status = HealthStatus.WARNING
            issues.append(f"{unhealthy} unhealthy model-instrument pairs")
        else:
            status = HealthStatus.HEALTHY

        metrics = {
            "active_models": int(data.get("model_count", 0)),
            "avg_confidence": confidence,
            "avg_p99_latency_ms": float(data.get("avg_p99_latency", 0)),
            "max_p99_latency_ms": max_latency,
            "unhealthy_pairs": unhealthy,
        }

        return ComponentHealth(
            name="Model Performance",
            status=status,
            message=f"{metrics['active_models']} models, confidence: {confidence:.2f}",
            metrics=metrics,
            issues=issues,
        )

    def check_all_components(self) -> dict[str, ComponentHealth]:
        """
        Run all health checks.

        Returns:
            Dictionary of component health statuses

        """
        checks = {
            "pipeline": self.check_pipeline_health,
            "data_collection": self.check_data_collection,
            "feature_computation": self.check_feature_computation,
            "data_freshness": self.check_data_freshness,
            "errors": self.check_errors,
            "model_performance": self.check_model_performance,
        }

        results = {}
        for name, check_func in checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {e!s}",
                    metrics={},
                    issues=[str(e)],
                )

        return results

    def get_overall_status(
        self,
        component_health: dict[str, ComponentHealth],
    ) -> tuple[HealthStatus, int]:
        """
        Determine overall system health and exit code.

        Args:
            component_health: Dictionary of component health statuses

        Returns:
            Overall status and exit code (0=healthy, 1=warning, 2=critical)

        """
        has_critical = any(c.status == HealthStatus.CRITICAL for c in component_health.values())
        has_warning = any(c.status == HealthStatus.WARNING for c in component_health.values())
        has_unknown = any(c.status == HealthStatus.UNKNOWN for c in component_health.values())

        if has_critical:
            return HealthStatus.CRITICAL, 2
        elif has_warning or has_unknown:
            return HealthStatus.WARNING, 1
        else:
            return HealthStatus.HEALTHY, 0


def format_human_output(
    component_health: dict[str, ComponentHealth],
    overall_status: HealthStatus,
) -> str:
    """
    Format health check results for human consumption.

    Args:
        component_health: Component health statuses
        overall_status: Overall system status

    Returns:
        Formatted string output

    """
    output = []

    # Header
    output.append("=" * 80)
    output.append("ML PIPELINE HEALTH CHECK REPORT")
    output.append(f"Timestamp: {datetime.now().isoformat()}")
    output.append(f"Overall Status: {overall_status.value.upper()}")
    output.append("=" * 80)
    output.append("")

    # Component details
    for name, health in component_health.items():
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓",
        }.get(health.status, "")

        output.append(f"{status_emoji} {health.name}")
        output.append(f"   Status: {health.status.value}")
        output.append(f"   Message: {health.message}")

        if health.issues:
            output.append("   Issues:")
            for issue in health.issues:
                output.append(f"     - {issue}")

        if health.metrics:
            output.append("   Metrics:")
            for key, value in health.metrics.items():
                if isinstance(value, float):
                    output.append(f"     - {key}: {value:.2f}")
                else:
                    output.append(f"     - {key}: {value}")

        if health.last_update:
            output.append(f"   Last Update: {health.last_update}")

        output.append("")

    # Summary table
    output.append("SUMMARY")
    output.append("-" * 40)

    summary_data = []
    for name, health in component_health.items():
        summary_data.append(
            [
                health.name,
                health.status.value,
                "Yes" if health.issues else "No",
            ],
        )

    output.append(
        tabulate(
            summary_data,
            headers=["Component", "Status", "Has Issues"],
            tablefmt="simple",
        ),
    )

    return "\n".join(output)


def format_json_output(
    component_health: dict[str, ComponentHealth],
    overall_status: HealthStatus,
    exit_code: int,
) -> str:
    """
    Format health check results as JSON.

    Args:
        component_health: Component health statuses
        overall_status: Overall system status
        exit_code: System exit code

    Returns:
        JSON string

    """
    components: dict[str, Any] = {}
    data: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": overall_status.value,
        "exit_code": exit_code,
        "components": components,
    }

    for name, health in component_health.items():
        components[name] = {
            "name": health.name,
            "status": health.status.value,
            "message": health.message,
            "metrics": health.metrics,
            "issues": health.issues,
            "last_update": health.last_update.isoformat() if health.last_update else None,
        }

    return json.dumps(data, indent=2, default=str)


def main() -> int:
    """
    Main entry point for health check script.

    Returns:
        Exit code (0=healthy, 1=warning, 2=critical)

    """
    if not HAS_PSYCOPG2:
        print(
            "Error: psycopg2 is required. Install with: pip install psycopg2-binary",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description="Check ML pipeline health")
    parser.add_argument(
        "--connection-string",
        default=os.environ.get(
            "ML_DB_CONNECTION",
            "postgresql://postgres:postgres@localhost:5432/nautilus",
        ),
        help="PostgreSQL connection string",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--critical-only",
        action="store_true",
        help="Show only critical issues",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export results to file",
    )

    args = parser.parse_args()

    try:
        # Run health checks
        with PipelineHealthChecker(args.connection_string) as checker:
            component_health = checker.check_all_components()
            overall_status, exit_code = checker.get_overall_status(component_health)

        # Filter results if requested
        if args.critical_only:
            component_health = {
                k: v for k, v in component_health.items() if v.status == HealthStatus.CRITICAL
            }

            if not component_health:
                print("No critical issues found")
                return 0

        # Format output
        if args.json:
            output = format_json_output(component_health, overall_status, exit_code)
        else:
            output = format_human_output(component_health, overall_status)

        # Display or export
        if args.export:
            with open(args.export, "w") as f:
                f.write(output)
            print(f"Results exported to {args.export}")
        else:
            print(output)

        return exit_code

    except Exception as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
