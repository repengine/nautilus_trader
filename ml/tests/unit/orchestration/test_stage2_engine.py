from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from ml.orchestration import stage2_engine as stage2_module


class _FakeFrame:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def sort_values(self, _key: str) -> _FakeFrame:
        return self

    def tail(self, _n: int) -> _FakeFrame:
        return self


def _install_fake_parquet(
    monkeypatch: Any,
    *,
    columns: list[str],
    num_rows: int,
    row_group_rows: list[int],
) -> None:
    """Install fake pyarrow.parquet modules for helper tests."""
    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")

    class _RowGroupMeta:
        def __init__(self, rows: int) -> None:
            self.num_rows = rows

    class _Metadata:
        def __init__(self, rows: int, groups: list[int]) -> None:
            self.num_rows = rows
            self._groups = groups

        def row_group(self, index: int) -> _RowGroupMeta:
            return _RowGroupMeta(self._groups[index])

    class _Table:
        def to_pandas(self) -> _FakeFrame:
            return _FakeFrame(columns)

    class _ParquetFile:
        def __init__(self, _path: str) -> None:
            self.metadata = _Metadata(num_rows, row_group_rows)
            self.num_row_groups = len(row_group_rows)

        def read_row_groups(self, _row_groups: list[int], columns: list[str]) -> _Table:
            del columns
            return _Table()

    fake_parquet.ParquetFile = _ParquetFile  # type: ignore[attr-defined]
    fake_pyarrow.parquet = fake_parquet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)


def test_load_validation_arrays_success_and_missing_keys(tmp_path: Path) -> None:
    """Array loader should return flattened arrays only when required keys exist."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "teacher_preds.npz", q_val=np.array([[1.0, 2.0]]), y_val_true=np.array([0.0, 1.0]))

    arrays = stage2_module._load_validation_arrays(str(out_dir))
    assert arrays is not None
    q_val, y_true = arrays
    assert q_val.tolist() == [1.0, 2.0]
    assert y_true.tolist() == [0.0, 1.0]

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(missing_dir / "teacher_preds.npz", q_only=np.array([1.0]))
    assert stage2_module._load_validation_arrays(str(missing_dir)) is None
    assert stage2_module._load_validation_arrays(str(tmp_path / "does_not_exist")) is None


def test_load_validation_tail_parquet_success_and_empty_cases(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Parquet tail helper should handle successful and empty metadata paths."""
    import ml._imports as imports_module

    monkeypatch.setattr(imports_module, "pd", object())
    monkeypatch.setattr(imports_module, "HAS_PANDAS", True)
    _install_fake_parquet(
        monkeypatch,
        columns=["time_index", "timestamp", "instrument_id"],
        num_rows=10,
        row_group_rows=[3, 3, 4],
    )

    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")
    tail = stage2_module._load_validation_tail_parquet(str(dataset_path), 5)
    assert tail is not None

    _install_fake_parquet(
        monkeypatch,
        columns=["time_index", "timestamp", "instrument_id"],
        num_rows=0,
        row_group_rows=[0],
    )
    assert stage2_module._load_validation_tail_parquet(str(dataset_path), 5) is None


def test_load_validation_tail_parquet_returns_none_for_missing_columns(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Parquet tail helper should return None when required columns are absent."""
    import ml._imports as imports_module

    monkeypatch.setattr(imports_module, "pd", object())
    monkeypatch.setattr(imports_module, "HAS_PANDAS", True)
    _install_fake_parquet(
        monkeypatch,
        columns=["timestamp", "instrument_id"],
        num_rows=6,
        row_group_rows=[6],
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")
    assert stage2_module._load_validation_tail_parquet(str(dataset_path), 3) is None


def test_load_validation_tail_csv_paths(monkeypatch: Any, tmp_path: Path) -> None:
    """CSV fallback should return tail when required columns exist."""
    class _PandasModule:
        @staticmethod
        def read_csv(_path: str) -> _FakeFrame:
            return _FakeFrame(["time_index", "timestamp", "instrument_id"])

    import ml._imports as imports_module

    monkeypatch.setattr(imports_module, "pd", _PandasModule())
    monkeypatch.setattr(imports_module, "HAS_PANDAS", True)

    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("time_index,timestamp,instrument_id\n0,1,SPY\n", encoding="utf-8")
    tail = stage2_module._load_validation_tail(str(csv_path), None, 1)
    assert tail is not None

    class _BadPandasModule:
        @staticmethod
        def read_csv(_path: str) -> _FakeFrame:
            return _FakeFrame(["time_index", "timestamp"])

    monkeypatch.setattr(imports_module, "pd", _BadPandasModule())
    assert stage2_module._load_validation_tail(str(csv_path), None, 1) is None


def test_build_engine_respects_backtest_toggle(monkeypatch: Any) -> None:
    """Engine factory should route to backtest runner only when enabled."""
    monkeypatch.setattr(stage2_module, "_BACKTEST_ENABLED", True)
    assert isinstance(stage2_module.build_engine("backtest"), stage2_module.BacktestStage2EngineRunner)
    assert isinstance(stage2_module.build_engine("returns"), stage2_module.ReturnsStage2Engine)


def test_load_validation_arrays_handles_numpy_load_error(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Array loader should return None when numpy loading fails."""
    out_dir = tmp_path / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "teacher_preds.npz").write_bytes(b"not-a-real-npz")

    def _raise_load_error(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("npz failure")

    monkeypatch.setattr(np, "load", _raise_load_error)
    assert stage2_module._load_validation_arrays(str(out_dir)) is None


def test_load_validation_tail_parquet_dependency_and_metadata_guards(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Parquet helper should handle pandas dependency setup and empty metadata."""
    import ml._imports as imports_module

    dependency_calls: list[tuple[str, ...]] = []

    def _check_dependencies(deps: list[str]) -> None:
        dependency_calls.append(tuple(deps))
        imports_module.pd = object()

    monkeypatch.setattr(imports_module, "pd", None)
    monkeypatch.setattr(imports_module, "HAS_PANDAS", False)
    monkeypatch.setattr(imports_module, "check_ml_dependencies", _check_dependencies)

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")

    class _ParquetFile:
        def __init__(self, _path: str) -> None:
            self.metadata = None

    fake_parquet.ParquetFile = _ParquetFile  # type: ignore[attr-defined]
    fake_pyarrow.parquet = fake_parquet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")
    assert stage2_module._load_validation_tail_parquet(str(dataset_path), 5) is None
    assert dependency_calls == [("pandas",)]


def test_load_validation_tail_parquet_additional_branches(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Parquet helper should cover non-break and missing required-column branches."""
    import ml._imports as imports_module

    monkeypatch.setattr(imports_module, "pd", object())
    monkeypatch.setattr(imports_module, "HAS_PANDAS", True)

    _install_fake_parquet(
        monkeypatch,
        columns=["time_index", "timestamp", "instrument_id"],
        num_rows=10,
        row_group_rows=[3, 3, 4],
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")
    # n_tail larger than total rows exercises loop path with no early break.
    assert stage2_module._load_validation_tail_parquet(str(dataset_path), 99) is not None

    _install_fake_parquet(
        monkeypatch,
        columns=["time_index", "timestamp"],
        num_rows=4,
        row_group_rows=[4],
    )
    assert stage2_module._load_validation_tail_parquet(str(dataset_path), 2) is None


def test_load_validation_tail_parquet_handles_reader_exception(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Parquet helper should return None when pyarrow reader setup fails."""
    import ml._imports as imports_module

    monkeypatch.setattr(imports_module, "pd", object())
    monkeypatch.setattr(imports_module, "HAS_PANDAS", True)

    fake_pyarrow = types.ModuleType("pyarrow")
    fake_parquet = types.ModuleType("pyarrow.parquet")

    class _ParquetFile:
        def __init__(self, _path: str) -> None:
            raise RuntimeError("reader failure")

    fake_parquet.ParquetFile = _ParquetFile  # type: ignore[attr-defined]
    fake_pyarrow.parquet = fake_parquet  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pyarrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", fake_parquet)

    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"placeholder")
    assert stage2_module._load_validation_tail_parquet(str(dataset_path), 3) is None


def test_load_validation_tail_prefers_parquet_and_covers_csv_fallback_paths(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Tail loader should short-circuit parquet and handle CSV dependency/error paths."""
    marker = object()
    monkeypatch.setattr(
        stage2_module,
        "_load_validation_tail_parquet",
        lambda _parquet_path, _n_tail: (marker, marker),
    )
    assert stage2_module._load_validation_tail("dataset.csv", "dataset.parquet", 5) == (marker, marker)

    import ml._imports as imports_module

    class _BadCsvPandas:
        @staticmethod
        def read_csv(_path: str) -> _FakeFrame:
            return _FakeFrame(["timestamp", "instrument_id"])

    monkeypatch.setattr(stage2_module, "_load_validation_tail_parquet", lambda _parquet_path, _n_tail: None)
    monkeypatch.setattr(imports_module, "pd", None)
    monkeypatch.setattr(imports_module, "HAS_PANDAS", False)
    monkeypatch.setattr(imports_module, "check_ml_dependencies", lambda _deps: setattr(imports_module, "pd", _BadCsvPandas()))
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("time_index,timestamp,instrument_id\n0,1,SPY\n", encoding="utf-8")
    assert stage2_module._load_validation_tail(str(csv_path), None, 1) is None

    class _ExplodingCsvPandas:
        @staticmethod
        def read_csv(_path: str) -> _FakeFrame:
            raise RuntimeError("csv failure")

    monkeypatch.setattr(imports_module, "pd", _ExplodingCsvPandas())
    monkeypatch.setattr(imports_module, "HAS_PANDAS", True)
    assert stage2_module._load_validation_tail(str(csv_path), None, 1) is None
