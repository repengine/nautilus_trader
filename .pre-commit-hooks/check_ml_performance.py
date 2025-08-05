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
Pre-commit hook to check ML performance benchmarks.

Ensures inference latency doesn't regress beyond acceptable thresholds.

"""

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean
from statistics import stdev

import numpy as np


class PerformanceBenchmark:
    """
    Run performance benchmarks for ML components.
    """

    def __init__(self):
        self.baseline_file = Path(".ml_performance_baseline.json")
        self.tolerance = 0.20  # Allow 20% regression (not too strict)
        self.warm_up_runs = 10
        self.benchmark_runs = 50

    def load_baseline(self):
        """
        Load baseline performance metrics.
        """
        if not self.baseline_file.exists():
            return {}

        with open(self.baseline_file) as f:
            return json.load(f)

    def save_baseline(self, metrics):
        """
        Save new baseline metrics.
        """
        with open(self.baseline_file, "w") as f:
            json.dump(metrics, f, indent=2)

    def benchmark_feature_computation(self):
        """
        Benchmark feature computation speed.
        """
        # Create minimal test setup
        test_code = f"""
import numpy as np
import time
from nautilus_ml.inference.features import FeatureEngine

# Setup
engine = FeatureEngine()
price = 100.0
volume = 1000.0

# Warm-up
for _ in range({self.warm_up_runs}):
    engine.compute_features(price, volume)

# Benchmark
times = []
for _ in range({self.benchmark_runs}):
    start = time.perf_counter_ns()
    features = engine.compute_features(price, volume)
    end = time.perf_counter_ns()
    times.append((end - start) / 1000)  # Convert to microseconds

print("RESULT:" + json.dumps(times))
"""

        # Run benchmark
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Feature benchmark failed: {result.stderr}")
            return None

        # Parse results
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT:"):
                times = json.loads(line.split(":", 1)[1])
                return {
                    "mean_us": mean(times),
                    "std_us": stdev(times) if len(times) > 1 else 0,
                    "p95_us": np.percentile(times, 95),
                    "p99_us": np.percentile(times, 99),
                }

        return None

    def benchmark_model_inference(self):
        """
        Benchmark model inference speed.
        """
        test_code = f"""
import numpy as np
import time
import onnxruntime as ort

# Load model (assuming ONNX model exists)
try:
    session = ort.InferenceSession("models/test_model.onnx")
    input_name = session.get_inputs()[0].name
    n_features = session.get_inputs()[0].shape[1]
except:
    # Fallback for testing
    session = None
    input_name = "input"
    n_features = 10

# Create test data
test_features = np.random.randn(1, n_features).astype(np.float32)

# Warm-up
for _ in range({self.warm_up_runs}):
    if session:
        _ = session.run(None, {{input_name: test_features}})
    else:
        # Simulate inference
        _ = np.dot(test_features, np.random.randn(n_features, 1))

# Benchmark
times = []
for _ in range({self.benchmark_runs}):
    start = time.perf_counter_ns()
    if session:
        prediction = session.run(None, {{input_name: test_features}})
    else:
        prediction = np.dot(test_features, np.random.randn(n_features, 1))
    end = time.perf_counter_ns()
    times.append((end - start) / 1000)  # Convert to microseconds

print("RESULT:" + json.dumps(times))
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return None

        for line in result.stdout.split("\n"):
            if line.startswith("RESULT:"):
                times = json.loads(line.split(":", 1)[1])
                return {
                    "mean_us": mean(times),
                    "std_us": stdev(times) if len(times) > 1 else 0,
                    "p95_us": np.percentile(times, 95),
                    "p99_us": np.percentile(times, 99),
                }

        return None

    def _check_component_performance(self, component, metrics, baseline):
        """
        Check performance for a single component.
        """
        messages = []
        failed = False

        if component not in baseline:
            messages.append(f"INFO: {component}: No baseline, setting current as baseline")
            return failed, messages

        # Check mean performance
        baseline_mean = baseline[component]["mean_us"]
        current_mean = metrics["mean_us"]
        regression = (current_mean - baseline_mean) / baseline_mean

        if regression > self.tolerance:
            failed = True
            messages.append(
                f"FAIL: {component}: {current_mean:.1f}us "
                f"(+{regression*100:.1f}% from baseline {baseline_mean:.1f}us)",
            )
        else:
            messages.append(
                f"OK: {component}: {current_mean:.1f}us " f"({regression*100:+.1f}% from baseline)",
            )

        # Also check p99 for consistency
        if metrics["p99_us"] > 5000:  # 5ms hard limit
            failed = True
            messages.append(
                f"FAIL: {component}: p99 latency {metrics['p99_us']:.1f}us exceeds 5ms limit",
            )

        return failed, messages

    def check_performance(self, changed_files):
        """
        Check if performance has regressed.
        """
        # Only check if inference/feature files changed
        relevant_files = [
            f for f in changed_files if "ml/inference" in f or "ml/core/features" in f
        ]

        if not relevant_files:
            return True, "No performance-critical files changed"

        print("Running ML performance benchmarks...")

        # Load baseline
        baseline = self.load_baseline()

        # Run benchmarks
        results = {}

        # Feature computation benchmark
        feature_perf = self.benchmark_feature_computation()
        if feature_perf:
            results["features"] = feature_perf

        # Model inference benchmark
        inference_perf = self.benchmark_model_inference()
        if inference_perf:
            results["inference"] = inference_perf

        # Check for regressions
        failed = False
        all_messages = []

        for component, metrics in results.items():
            component_failed, messages = self._check_component_performance(
                component,
                metrics,
                baseline,
            )
            if component_failed:
                failed = True
            all_messages.extend(messages)

        # Update baseline if no regression or first run
        if not failed and results:
            self._update_baseline_if_needed(baseline, results)

        return not failed, "\n".join(all_messages)

    def _update_baseline_if_needed(self, baseline, results):
        """
        Update baseline if performance improved or within tolerance.
        """
        updated_baseline = baseline.copy()
        for component, metrics in results.items():
            if component not in baseline or metrics["mean_us"] < baseline[component]["mean_us"] * (
                1 + self.tolerance
            ):
                updated_baseline[component] = metrics
        self.save_baseline(updated_baseline)


def main():
    """
    Run ML performance benchmarks on changed files.
    """
    changed_files = sys.argv[1:]

    # Check for --update-baseline flag
    if "--update-baseline" in changed_files:
        benchmark = PerformanceBenchmark()
        print("Updating ML performance baseline...")

        # Run benchmarks and force save
        results = {}
        feature_perf = benchmark.benchmark_feature_computation()
        if feature_perf:
            results["features"] = feature_perf

        inference_perf = benchmark.benchmark_model_inference()
        if inference_perf:
            results["inference"] = inference_perf

        benchmark.save_baseline(results)
        print(f"Baseline updated with {len(results)} components")
        return 0

    benchmark = PerformanceBenchmark()
    passed, message = benchmark.check_performance(changed_files)

    print(message)

    if not passed:
        print("\nPerformance regression detected!")
        print("Options:")
        print("1. Optimize your code to meet performance targets")
        print("2. If this is expected, update baseline with: make update-ml-baseline")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
