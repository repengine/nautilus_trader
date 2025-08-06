#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------
"""
Example demonstrating ML monitoring infrastructure usage.

This example shows how to use the monitoring components for tracking ML model
performance and system health.

"""

import asyncio
import time

import numpy as np

from ml.monitoring import MetricsServer
from ml.monitoring import MLMetricsCollector
from ml.monitoring import MonitoringConfig


def simulate_ml_prediction() -> tuple[str, float]:
    """
    Simulate ML model prediction.

    Returns
    -------
    tuple[str, float]
        Prediction class and confidence score.

    """
    # Generate random prediction for example
    rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
    predictions = ["buy", "sell", "hold"]
    prediction = rng.choice(predictions)
    confidence = rng.uniform(0.5, 0.95)
    return prediction, confidence


def simulate_feature_computation() -> None:
    """
    Simulate feature computation work.
    """
    time.sleep(0.001)  # Simulate 1ms computation


async def run_monitoring_example() -> None:
    """
    Run monitoring infrastructure example.

    Demonstrates metrics collection and server functionality.

    """
    print("=== ML Monitoring Infrastructure Example ===")
    print()

    # Configure monitoring
    config = MonitoringConfig(
        enabled=True,
        metrics_port=8090,  # Use different port to avoid conflicts
        metrics_prefix="example_ml",
        health_check_interval=30.0,
        export_interval=5.0,
    )

    # Create metrics collector
    collector = MLMetricsCollector(config)
    print(f"Metrics collector enabled: {collector.enabled}")

    # Start metrics server (if Prometheus is available)
    server = MetricsServer(config)
    try:
        server.start()
        if server.is_running():
            print(f"Metrics server started on {server.get_metrics_url()}")
            print(f"Health endpoint available at {server.get_health_url()}")
        else:
            print("Metrics server not started (Prometheus client not available)")
    except Exception as e:
        print(f"Could not start metrics server: {e}")

    print()
    print("Simulating ML operations...")

    # Simulate various ML operations
    for i in range(10):
        # Record prediction with timing
        with collector.time_prediction("xgboost_v1", "EURUSD") as timer:
            prediction, confidence = simulate_ml_prediction()
            timer.set_prediction(prediction, confidence)

        print(f"Prediction {i+1}: {prediction} (confidence: {confidence:.3f})")

        # Record feature computation
        with collector.time_feature_computation("EURUSD", "technical"):
            simulate_feature_computation()

        # Occasionally simulate errors
        if i == 3:
            collector.record_error(
                model="xgboost_v1",
                instrument="EURUSD",
                error_type="inference_timeout",
            )
            print("  -> Simulated inference timeout error")

        if i == 7:
            collector.record_error(
                model="xgboost_v1",
                instrument="EURUSD",
                error_type="feature_computation",
            )
            print("  -> Simulated feature computation error")

        # Small delay between operations
        await asyncio.sleep(0.1)

    print()
    print("=== Example Summary ===")
    print("Total predictions: 10")
    print("Total errors: 2")
    print(f"Collector configuration: {config.metrics_prefix}")

    if server.is_running():
        print()
        print("Metrics server is running. You can:")
        print(f"  - View metrics: curl {server.get_metrics_url()}")
        print(f"  - Check health: curl {server.get_health_url()}")
        print("  - Configure Prometheus to scrape the metrics endpoint")
        print()
        print("Press Ctrl+C to stop...")

        # Keep running for a short time for demonstration
        print("Running for 3 seconds...")
        await asyncio.sleep(3)
    else:
        print()
        print("To enable full monitoring:")
        print("  pip install 'nautilus-trader[ml]'")

    # Cleanup
    server.stop()
    print("Monitoring example completed.")


def main() -> None:
    """
    Execute the monitoring example.
    """
    asyncio.run(run_monitoring_example())


if __name__ == "__main__":
    main()
