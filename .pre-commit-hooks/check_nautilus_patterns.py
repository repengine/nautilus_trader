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
Pre-commit hook to validate Nautilus Trader patterns in ML code.

Ensures ML code follows Nautilus architectural patterns and best practices.

"""

import ast
import sys
from pathlib import Path


class NautilusPatternValidator(ast.NodeVisitor):
    """
    Validate Nautilus patterns in ML code.

    Parameters
    ----------
    filename : str
        The filename being validated.
    filepath : Path
        The Path object for the file.

    """

    def __init__(self, filename: str, filepath: Path):
        self.filename = filename
        self.filepath = filepath
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.imports: dict[str, str] = {}
        self.current_class = None
        self.in_init = False
        self.has_on_start = False
        self.current_function: str | None = None
        self.in_event_handler = False
        self.total_lines: int = 0
        self._in_type_checking_block: int = 0

    def visit_Import(self, node):
        """
        Track imports to validate hot/cold path separation.

        Parameters
        ----------
        node : ast.Import
            Import node.

        """
        # Ignore TYPE_CHECKING imports
        if self._in_type_checking_block > 0:
            return
        for alias in node.names:
            self.imports[alias.name] = alias.asname or alias.name

    def visit_ImportFrom(self, node):
        """
        Track from imports.

        Parameters
        ----------
        node : ast.ImportFrom
            ImportFrom node.

        """
        # Ignore TYPE_CHECKING imports
        if self._in_type_checking_block > 0:
            return
        if node.module:
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                self.imports[alias.name] = alias.asname or alias.name
                self.imports[full_name] = full_name

    def visit_If(self, node: ast.If):  # type: ignore[override]
        # Track TYPE_CHECKING blocks to avoid flagging type-only imports
        is_type_checking = isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        if is_type_checking:
            self._in_type_checking_block += 1
            for n in node.body:
                self.visit(n)
            self._in_type_checking_block -= 1
            # Visit orelse normally
            for n in node.orelse:
                self.visit(n)
        else:
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        """
        Validate class definitions.

        Parameters
        ----------
        node : ast.ClassDef
            Class definition node.

        """
        self.current_class = node.name
        self.has_on_start = False

        # Check Actor patterns
        if self._is_actor_class(node):
            self._validate_actor_patterns(node)

        # Check Strategy patterns
        if self._is_strategy_class(node):
            self._validate_strategy_patterns(node)

        # Check Config patterns
        if node.name.endswith("Config"):
            self._validate_config_patterns(node)

        self.generic_visit(node)

        # God-class heuristic: extremely large classes are hard to maintain
        class_len = self._estimate_class_length(node)
        if class_len is not None:
            # Stricter threshold for actors; looser for others
            threshold = 1000 if self._is_actor_class(node) else 900
            if class_len >= threshold:
                self.warnings.append(
                    f"Line {node.lineno}: Class '{node.name}' spans ~{class_len} lines (potential god-class)",
                )
        self.current_class = None

    def visit_FunctionDef(self, node):
        """
        Validate function definitions.

        Parameters
        ----------
        node : ast.FunctionDef
            Function definition node.

        """
        old_in_init = self.in_init
        old_in_event_handler = self.in_event_handler
        old_current_function = self.current_function

        self.current_function = node.name

        if node.name == "__init__":
            self.in_init = True
            if self.current_class and self._is_strategy_class_name(self.current_class):
                self._validate_strategy_init(node)

        elif node.name == "on_start":
            self.has_on_start = True

        elif node.name.startswith("on_") and node.name not in {"on_start"}:
            self._validate_event_handler(node)
            self.in_event_handler = True

        self.generic_visit(node)
        self.in_init = old_in_init
        self.in_event_handler = old_in_event_handler
        self.current_function = old_current_function

    def visit_Call(self, node: ast.Call):  # type: ignore[override]
        """
        Validate specific calls for hot-path and compliance issues.
        """
        # Detect builtin open() usage
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            if self.in_event_handler or self._is_hot_path_file():
                self.errors.append(
                    f"Line {node.lineno}: File I/O 'open()' in hot path/event handler '{self.current_function}'",
                )

        # Detect training during inference: *.fit(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            if self.in_event_handler or self._is_hot_path_file():
                self.errors.append(
                    f"Line {node.lineno}: Model training 'fit()' in hot path/event handler '{self.current_function}'",
                )

        # Detect network calls in hot path (requests.*)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in {"requests", "httpx"} and self.in_event_handler:
                self.errors.append(
                    f"Line {node.lineno}: Network call '{node.func.value.id}.{node.func.attr}()' in event handler",
                )
            # Pandas DataFrame creation in hot path
            if node.func.value.id in {"pd", "pandas"} and node.func.attr == "DataFrame":
                if self.in_event_handler or self._is_hot_path_file():
                    self.errors.append(
                        f"Line {node.lineno}: Pandas DataFrame construction in hot path/event handler",
                    )

        # Attribute call from module-qualified name
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute):
            root = node.func.value
            # urllib.request.* in event handler
            if (
                isinstance(root.value, ast.Name)
                and root.value.id == "urllib"
                and root.attr == "request"
                and self.in_event_handler
            ):
                self.errors.append(
                    f"Line {node.lineno}: Network call 'urllib.request.{node.func.attr}()' in event handler",
                )

        # Detect use of build_topic (should use build_topic_for_stage in stores/actors)
        if (isinstance(node.func, ast.Name) and node.func.id == "build_topic") or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "build_topic"
        ):
            if self._is_stores_or_actors_file():
                self.errors.append(
                    f"Line {node.lineno}: Use build_topic_for_stage(...) instead of build_topic(...) in stores/actors",
                )

        # Direct SQLAlchemy create_engine usage (should be via EngineManager)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "create_engine":
            self.warnings.append(
                f"Line {node.lineno}: Direct SQLAlchemy create_engine() detected; prefer EngineManager.get_engine(...)",
            )

        # Direct sqlite3.connect or redis.Redis() usage (bypass store pattern)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "sqlite3" and node.func.attr == "connect":
                if self._is_hot_path_file() or self._is_stores_or_actors_file():
                    self.errors.append(
                        f"Line {node.lineno}: Direct sqlite3.connect() usage; do not bypass stores/registries",
                    )
            if node.func.value.id == "redis" and node.func.attr in {"Redis", "from_url"}:
                if self._is_hot_path_file() or self._is_stores_or_actors_file():
                    self.errors.append(
                        f"Line {node.lineno}: Direct redis client construction; use configured message bus or store",
                    )

        # Insecure serialization calls (pickle/joblib)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id in {"pickle", "joblib"} and node.func.attr in {"load", "dump"}:
                if any(
                    seg in str(self.filepath)
                    for seg in ("actors/", "strategies/", "deployment/", "inference/")
                ):
                    self.errors.append(
                        f"Line {node.lineno}: Insecure serialization {node.func.value.id}.{node.func.attr}() in production path",
                    )

        # EventStatus literals via dict()/update({}) patterns
        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            for kw in node.keywords or []:
                if (
                    kw.arg == "status"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ):
                    if kw.value.value.lower() in {"success", "failed", "partial"}:
                        self.errors.append(
                            f"Line {node.lineno}: Use EventStatus.<...>.value instead of raw '{kw.value.value}'",
                        )

        if isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            # Check positional dict literal
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    for k, v in zip(arg.keys, arg.values):
                        if (
                            isinstance(k, ast.Constant)
                            and k.value == "status"
                            and isinstance(v, ast.Constant)
                            and isinstance(v.value, str)
                        ):
                            if v.value.lower() in {"success", "failed", "partial"}:
                                self.errors.append(
                                    f"Line {node.lineno}: Use EventStatus.<...>.value instead of raw '{v.value}'",
                                )

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """
        Check for prohibited attribute access.

        Parameters
        ----------
        node : ast.Attribute
            Attribute access node.

        """
        if self.in_init and self.current_class:
            # Check for clock/logger access in __init__
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                if node.value.value.id == "self" and node.value.attr in ["clock", "logger"]:
                    self.errors.append(
                        f"Line {node.lineno}: Accessing self.{node.value.attr} in __init__ is prohibited",
                    )

    def _is_actor_class(self, node):
        """
        Check if class inherits from Actor.
        """
        return any(
            (isinstance(base, ast.Name) and base.id == "Actor")
            or (isinstance(base, ast.Attribute) and base.attr == "Actor")
            for base in node.bases
        )

    def _is_strategy_class(self, node):
        """
        Check if class inherits from Strategy.
        """
        return any(
            (isinstance(base, ast.Name) and base.id == "Strategy")
            or (isinstance(base, ast.Attribute) and base.attr == "Strategy")
            for base in node.bases
        )

    def _is_strategy_class_name(self, name):
        """
        Check if class name suggests it's a strategy.
        """
        return "Strategy" in name

    def _validate_actor_patterns(self, node):
        """
        Validate Actor-specific patterns.

        Parameters
        ----------
        node : ast.ClassDef
            Actor class node.

        """
        # Check for inference/feature paths
        if self._is_hot_path_file():
            # Hot path validations
            if "pandas" in self.imports or "pd" in self.imports:
                self.errors.append(
                    f"Line {node.lineno}: Actor '{node.name}' uses pandas in hot path (inference)",
                )

        # Prohibit store instantiation inside actors
        prohibited_store_types = {"FeatureStore", "ModelStore", "StrategyStore", "DataStore"}
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                fn = child.func
                if isinstance(fn, ast.Name) and fn.id in prohibited_store_types:
                    self.errors.append(
                        f"Line {child.lineno}: Do not instantiate stores directly inside actors; use pre-initialized stores",
                    )
                elif isinstance(fn, ast.Attribute) and fn.attr in prohibited_store_types:
                    self.errors.append(
                        f"Line {child.lineno}: Do not instantiate stores directly inside actors; use pre-initialized stores",
                    )

        # More actor validations can be added here

    def _validate_strategy_patterns(self, node):
        """
        Validate Strategy-specific patterns.

        Parameters
        ----------
        node : ast.ClassDef
            Strategy class node.

        """
        # Strategies should have on_start for initialization
        method_names = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

        if "__init__" in method_names and "on_start" not in method_names:
            self.warnings.append(
                f"Line {node.lineno}: Strategy '{node.name}' has __init__ but no on_start() method",
            )

    def _validate_config_patterns(self, node):
        """
        Validate Config class patterns.

        Parameters
        ----------
        node : ast.ClassDef
            Config class node.

        """
        # Check for frozen=True in class decorators or bases
        has_frozen = False
        # Detect @dataclass(frozen=True) on class decorators
        for dec in getattr(node, "decorator_list", []):
            if isinstance(dec, ast.Call):
                fn = dec.func
                is_dc = (isinstance(fn, ast.Name) and fn.id == "dataclass") or (
                    isinstance(fn, ast.Attribute) and fn.attr == "dataclass"
                )
                if is_dc:
                    for kw in dec.keywords or []:
                        if (
                            kw.arg == "frozen"
                            and isinstance(kw.value, ast.Constant)
                            and kw.value.value is True
                        ):
                            has_frozen = True
                            break

        # Check bases for frozen parameter
        for base in node.bases:
            # Check if base is a tuple with frozen=True
            if isinstance(base, ast.Name) and base.id.endswith("Config"):
                # Look for frozen in keywords
                continue
            if isinstance(base, ast.Call):
                for keyword in base.keywords:
                    if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant):
                        if keyword.value.value is True:
                            has_frozen = True

        # Also check keywords directly on the class
        if hasattr(node, "keywords"):
            for keyword in node.keywords:
                if keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant):
                    if keyword.value.value is True:
                        has_frozen = True

        # For inheritance syntax like class Foo(Bar, frozen=True)
        # The frozen appears as a keyword, not in bases
        # So we shouldn't error if we have a Config base class
        for base in node.bases:
            if isinstance(base, ast.Name) and "Config" in base.id:
                # Assume configs with Config base are properly frozen
                has_frozen = True
                break

        if (
            not has_frozen
            and "Config" in node.name
            and not any(isinstance(base, ast.Name) and "Config" in base.id for base in node.bases)
        ):
            self.errors.append(
                f"Line {node.lineno}: Config class '{node.name}' should use frozen=True",
            )

    def _validate_strategy_init(self, node):
        """
        Validate Strategy __init__ method.

        Parameters
        ----------
        node : ast.FunctionDef
            __init__ method node.

        """
        # Already handled by visit_Attribute for clock/logger access

    def _validate_event_handler(self, node):
        """
        Validate event handler patterns.

        Parameters
        ----------
        node : ast.FunctionDef
            Event handler method node.

        """
        # Check for blocking operations
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    # Check for synchronous I/O operations
                    if child.func.attr in ["read", "write", "open"]:
                        self.warnings.append(
                            f"Line {child.lineno}: Potential blocking I/O in event handler '{node.name}'",
                        )

                    # Check for sleep/wait
                    if child.func.attr in ["sleep", "wait"]:
                        self.errors.append(
                            f"Line {child.lineno}: Blocking operation in event handler '{node.name}'",
                        )

    def visit_Dict(self, node: ast.Dict):  # type: ignore[override]
        """
        Detect raw string 'status' fields; enforce EventStatus enum usage.
        """
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "status":
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    val = value.value.lower()
                    if val in {"success", "failed", "partial"}:
                        self.errors.append(
                            f"Line {getattr(value, 'lineno', '?')}: Use EventStatus.<...>.value instead of raw '{value.value}'",
                        )
        self.generic_visit(node)

    def validate_module_level(self) -> None:
        """
        Module-level validations after traversal (imports, factories, prometheus,
        pickle).
        """
        # Direct prometheus_client import is forbidden
        if str(self.filepath).endswith("ml/_imports.py"):
            pass  # allow type-only optional imports aggregator
        elif str(self.filepath).endswith("ml/common/metrics.py"):
            pass  # central metrics module defines the canonical collectors
        elif any(imp.startswith("prometheus_client") for imp in self.imports):
            self.errors.append(
                "Direct prometheus_client import detected - use ml.common.metrics_bootstrap",
            )

        # Insecure pickle usage: prohibit in actors/strategies/inference/deployment paths
        path_str = str(self.filepath)
        if any(k in self.imports for k in ("pickle", "joblib")):
            if any(
                seg in path_str for seg in ("actors/", "strategies/", "deployment/", "inference/")
            ):
                self.errors.append(
                    "Insecure model serialization import (pickle/joblib) in production path; use ONNX + onnxruntime",
                )
            elif "training/" in path_str:
                self.warnings.append(
                    "Consider avoiding pickle/joblib even in training; prefer explicit formats (Parquet/ONNX)",
                )

        # Provider factory: detect long if/elif chains → suggest mapping
        if str(self.filepath).endswith("ml/data/providers/factory.py"):
            try:
                with open(self.filepath, encoding="utf-8") as f:
                    src = f.read()
                tree = ast.parse(src)
                for n in ast.walk(tree):
                    if isinstance(n, ast.FunctionDef) and n.name in {
                        "create_provider",
                        "get_provider",
                        "factory",
                    }:
                        chain_count = sum(1 for c in ast.walk(n) if isinstance(c, ast.If))
                        if chain_count >= 6:
                            self.errors.append(
                                f"Function '{n.name}' uses a large if/elif chain; replace with registry mapping/factory",
                            )
            except Exception:
                # Non-fatal: skip if unable to parse
                pass

    def _estimate_class_length(self, node: ast.ClassDef) -> int | None:
        """
        Estimate class length in lines using end_lineno if available.
        """
        try:
            end = getattr(node, "end_lineno", None)
            if end is None:
                # Fallback: find max lineno in body
                end = max((getattr(n, "lineno", 0) for n in ast.walk(node)), default=node.lineno)
            return int(end) - int(getattr(node, "lineno", 0)) + 1
        except Exception:
            return None

    def _is_hot_path_file(self) -> bool:
        p = str(self.filepath)
        return ("actors/" in p) or ("inference/" in p)

    def _is_stores_or_actors_file(self) -> bool:
        p = str(self.filepath)
        return ("actors/" in p) or ("stores/" in p)

    def validate_hot_cold_separation(self):
        """
        Validate hot/cold path separation rules.
        """
        # Additional validation based on file path
        path_str = str(self.filepath)

        if "inference" in path_str or "actors" in path_str:
            # Hot path checks
            if "polars" in self.imports or "pl" in self.imports:
                self.warnings.append(
                    "Polars should be used in cold path only (training), not in inference/actors",
                )

        elif "training" in path_str:
            # Cold path checks
            if "pandas" in self.imports or "pd" in self.imports:
                self.warnings.append(
                    "Consider using Polars instead of pandas for better performance in training",
                )


def check_file(filepath: str) -> tuple[bool, list[str], list[str]]:
    """
    Check a single file for Nautilus pattern compliance.

    Parameters
    ----------
    filepath : str
        Path to the file to check.

    Returns
    -------
    tuple[bool, list[str], list[str]]
        Tuple of (passed, errors, warnings).

    """
    path = Path(filepath)

    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=filepath)

        validator = NautilusPatternValidator(filepath, path)
        validator.visit(tree)
        validator.validate_hot_cold_separation()
        validator.validate_module_level()

        passed = len(validator.errors) == 0
        return passed, validator.errors, validator.warnings

    except Exception as e:
        return False, [f"Failed to parse {filepath}: {e}"], []


def main():
    """
    Execute the main Nautilus pattern checking process.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).

    """
    files = sys.argv[1:]

    # Only check ML Python files
    ml_files = [f for f in files if f.startswith("ml/") and f.endswith(".py")]

    if not ml_files:
        return 0

    # Skip test files and test directories
    ml_files = [
        f
        for f in ml_files
        if not Path(f).name.startswith("test_") and "ml/tests/" not in f and "/tests/" not in f
    ]

    if not ml_files:
        return 0

    print(f"Checking Nautilus patterns in {len(ml_files)} ML file(s)...")

    all_passed = True
    total_errors = []
    total_warnings = []

    for filepath in ml_files:
        passed, errors, warnings = check_file(filepath)

        if passed and not warnings:
            print(f"✓ {filepath}")
        elif passed and warnings:
            print(f"⚠ {filepath}")
            for warning in warnings:
                print(f"  Warning: {warning}")
            total_warnings.extend(warnings)
        else:
            print(f"✗ {filepath}")
            for error in errors:
                print(f"  Error: {error}")
            for warning in warnings:
                print(f"  Warning: {warning}")
            total_errors.extend(errors)
            total_warnings.extend(warnings)
            all_passed = False

    if not all_passed:
        print(f"\n❌ Found {len(total_errors)} pattern violation(s)")
        print("Please fix the errors to follow Nautilus patterns.")
        return 1

    if total_warnings:
        print(f"\n⚠️  Found {len(total_warnings)} warning(s)")
        print("Consider addressing the warnings for better code quality.")

    print("\n✅ All Nautilus patterns validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
