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
Pre-commit hook to validate feature parity between training and inference.

Ensures that features computed in both paths are exactly identical.

"""

import ast
import json
import subprocess
import sys
from pathlib import Path


class FeatureParityChecker:
    """
    Check that training and inference compute identical features.
    """

    def __init__(self):
        self.feature_manifest = Path(".ml_feature_manifest.json")
        self.tolerance = 1e-10  # Extremely tight tolerance

    def extract_feature_definitions(self, file_path):
        """
        Extract feature computation logic from Python files.
        """
        with open(file_path) as f:
            tree = ast.parse(f.read())

        features = {}

        for node in ast.walk(tree):
            # Look for feature computation methods
            if isinstance(node, ast.FunctionDef):
                if "feature" in node.name.lower() or "compute" in node.name.lower():
                    # Extract feature names and computation
                    feature_info = {
                        "function": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "returns": [],
                        "computations": [],
                    }

                    # Analyze function body
                    for stmt in node.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name):
                                    feature_info["computations"].append(target.id)
                        elif isinstance(stmt, ast.Return):
                            if isinstance(stmt.value, ast.Dict):
                                # Returning dict of features
                                for key in stmt.value.keys:
                                    if isinstance(key, ast.Constant):
                                        feature_info["returns"].append(key.value)

                    features[node.name] = feature_info

        return features

    def run_feature_comparison_test(self):
        """
        Run the actual feature parity test.
        """
        test_code = """
import numpy as np
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.model.data import Bar
from nautilus_ml.training.features import TrainingFeatureEngine
from nautilus_ml.inference.features import InferenceFeatureEngine

# Create test data
instrument = TestInstrumentProvider.btcusdt_binance()
bars = []
for i in range(100):
    bar = Bar(
        bar_type=instrument.bar_type(),
        open=100.0 + i * 0.1,
        high=100.5 + i * 0.1,
        low=99.5 + i * 0.1,
        close=100.0 + i * 0.1,
        volume=1000.0 + i * 10,
        ts_event=i * 1000000000,
        ts_init=i * 1000000000,
    )
    bars.append(bar)

# Training path
train_engine = TrainingFeatureEngine()
train_features = []
for bar in bars:
    features = train_engine.compute_features(bar)
    train_features.append(features)

# Inference path
inference_engine = InferenceFeatureEngine()
inference_features = []
for bar in bars:
    features = inference_engine.compute_features(bar)
    inference_features.append(features)

# Compare
all_match = True
for i, (train_feat, inf_feat) in enumerate(zip(train_features, inference_features)):
    # Check keys match
    if set(train_feat.keys()) != set(inf_feat.keys()):
        print(f"FAIL: Feature keys mismatch at index {i}")
        print(f"  Training: {sorted(train_feat.keys())}")
        print(f"  Inference: {sorted(inf_feat.keys())}")
        all_match = False
        break

    # Check values match
    for key in train_feat.keys():
        if not np.allclose(train_feat[key], inf_feat[key], rtol=1e-10, atol=1e-10):
            diff = abs(train_feat[key] - inf_feat[key])
            print(f"FAIL: Feature '{key}' mismatch at index {i}")
            print(f"  Training: {train_feat[key]}")
            print(f"  Inference: {inf_feat[key]}")
            print(f"  Difference: {diff}")
            all_match = False
            break

    if not all_match:
        break

if all_match:
    print("PASS: All features match exactly")
else:
    print("FAIL: Feature parity violation detected")
    exit(1)
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0, result.stdout + result.stderr

    def check_indicator_consistency(self):
        """
        Verify both paths use same indicators with same parameters.
        """
        # Check that indicator initialization is consistent
        check_code = '''
import ast
import sys

def get_indicators(file_path):
    """Extract indicator usage from file."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    indicators = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'id'):
                # Direct function call
                func_name = node.func.id
            elif hasattr(node.func, 'attr'):
                # Method call
                func_name = node.func.attr
            else:
                continue

            # Check if it's an indicator
            indicator_names = [
                'SimpleMovingAverage', 'SMA',
                'ExponentialMovingAverage', 'EMA',
                'RelativeStrengthIndex', 'RSI',
                'AverageTrueRange', 'ATR',
                'BollingerBands', 'MACD'
            ]

            if any(ind in func_name for ind in indicator_names):
                # Extract parameters
                params = []
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        params.append(arg.value)

                indicators[func_name] = params

    return indicators

# Check both files
training_indicators = get_indicators('nautilus_ml/training/features.py')
inference_indicators = get_indicators('nautilus_ml/inference/features.py')

# Compare
if training_indicators != inference_indicators:
    print("FAIL: Indicator usage mismatch")
    print(f"Training: {training_indicators}")
    print(f"Inference: {inference_indicators}")
    exit(1)
else:
    print("PASS: Indicator usage consistent")
'''

        result = subprocess.run(
            [sys.executable, "-c", check_code],
            capture_output=True,
            text=True,
        )

        return result.returncode == 0, result.stdout

    def validate_feature_manifest(self):
        """
        Check if feature definitions have changed.
        """
        # Generate current manifest
        current_manifest = {}

        # Get feature definitions from training
        training_path = Path("nautilus_ml/training/features.py")
        if training_path.exists():
            current_manifest["training"] = self.extract_feature_definitions(training_path)

        # Get feature definitions from inference
        inference_path = Path("nautilus_ml/inference/features.py")
        if inference_path.exists():
            current_manifest["inference"] = self.extract_feature_definitions(inference_path)

        # Load previous manifest
        if self.feature_manifest.exists():
            with open(self.feature_manifest) as f:
                previous_manifest = json.load(f)

            # Check for changes
            if current_manifest != previous_manifest:
                return False, "Feature definitions have changed - rerun parity tests"

        # Save current manifest
        with open(self.feature_manifest, "w") as f:
            json.dump(current_manifest, f, indent=2)

        return True, "Feature manifest updated"

    def check_parity(self, changed_files):
        """
        Check parity between training and inference feature definitions.
        """
        # Check if feature files changed
        feature_files = [
            f for f in changed_files if "features.py" in f and ("training" in f or "inference" in f)
        ]

        if not feature_files:
            return True, "No feature files changed"

        print("Checking ML feature parity...")

        # Run checks
        checks = []

        # 1. Validate feature manifest
        passed, msg = self.validate_feature_manifest()
        checks.append((passed, f"Feature manifest: {msg}"))

        # 2. Check indicator consistency
        passed, msg = self.check_indicator_consistency()
        checks.append((passed, f"Indicator consistency: {msg}"))

        # 3. Run actual parity test
        passed, msg = self.run_feature_comparison_test()
        checks.append((passed, f"Feature computation: {msg}"))

        # Report results
        all_passed = all(check[0] for check in checks)

        for passed, message in checks:
            if passed:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")

        return all_passed, ""


def main():
    """
    Execute the main feature parity checking process.
    """
    changed_files = sys.argv[1:]

    checker = FeatureParityChecker()
    passed, message = checker.check_parity(changed_files)

    if message:
        print(message)

    if not passed:
        print("\n❌ Feature parity check failed!")
        print("Ensure training and inference compute EXACTLY the same features.")
        print("Even small differences (1e-10) can lead to model degradation in production.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
