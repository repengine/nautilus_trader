#!/usr/bin/env python3

"""Integration tests for multi-symbol orchestration.

Tests verify that the orchestrator can process multiple symbols independently
with proper result isolation and no cross-contamination.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml.orchestration.config_types import DatasetBuildConfig
from ml.orchestration.config_types import OrchestratorConfig
from ml.orchestration.config_types import StudentDistillConfig
from ml.orchestration.config_types import TeacherTrainConfig
from ml.tests.utils.targets import build_default_target_semantics_payload
from ml.orchestration.pipeline_orchestrator import MLPipelineOrchestrator


@pytest.mark.integration
@pytest.mark.slow
class TestMultiSymbolOrchestration:
    """Integration tests for multi-symbol processing with result isolation."""

    @pytest.fixture
    def mock_orchestrator(self) -> MLPipelineOrchestrator:
        """Create orchestrator with mocked dependencies.

        Returns:
            MLPipelineOrchestrator instance with mocked components

        """
        # Create mock coverage provider
        mock_coverage = MagicMock()

        # Create mock writer
        mock_writer = MagicMock()

        # Create mock build_main - will be configured per test
        mock_build_main = MagicMock(return_value=0)

        # Create mock teacher_main
        mock_teacher_main = MagicMock(return_value=0)

        orchestrator = MLPipelineOrchestrator(
            coverage=mock_coverage,
            writer=mock_writer,
            build_main=mock_build_main,
            teacher_main=mock_teacher_main,
        )

        return orchestrator

    @pytest.fixture
    def multi_symbol_config(self, tmp_path: Path) -> OrchestratorConfig:
        """Create orchestrator configuration for multi-symbol testing.

        Args:
            tmp_path: Pytest temporary directory fixture

        Returns:
            OrchestratorConfig instance with multi-symbol settings

        """
        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        return OrchestratorConfig(
            dataset=DatasetBuildConfig(
                data_dir=str(data_dir),
                symbols="AAPL,GOOGL,MSFT",  # Three symbols
                out_dir=str(out_dir),
                start_iso="2024-01-01",
                end_iso="2024-01-31",
                target_semantics=build_default_target_semantics_payload(),
            ),
            hpo=None,  # Disable HPO for faster testing
            teacher=TeacherTrainConfig(
                enabled=False,  # Disable for faster testing
                model_id="test_teacher",
                max_epochs=1,
            ),
            student=StudentDistillConfig(
                enabled=False,  # Disable for faster testing
                model_id="test_student",
            ),
            promotions=None,  # Disable promotions for faster testing
            integration=None,  # Disable integration for faster testing
        )

    def test_e2e_multi_symbol_orchestration(
        self,
        mock_orchestrator: MLPipelineOrchestrator,
        multi_symbol_config: OrchestratorConfig,
        tmp_path: Path,
    ) -> None:
        """Test multi-symbol processing with result isolation.

        Property Under Test: Multi-symbol processing is independent and parallel-safe

        Given:
        - Three instruments: AAPL, GOOGL, MSFT
        - Same pipeline configuration applied to all symbols
        - Orchestrator configured for parallel processing

        When:
        - Running orchestrator.run(config) with symbols="AAPL,GOOGL,MSFT"
        - Each symbol processed independently (potentially in parallel)

        Then:
        - Separate datasets generated for each symbol
        - No cross-contamination (AAPL data not in GOOGL dataset)
        - All symbols processed successfully (3/3 success)
        - Results isolated (features/models per symbol separate)

        """
        # Create mock dataset files for each symbol
        out_dir = Path(multi_symbol_config.dataset.out_dir)

        def mock_build_dataset(cfg: DatasetBuildConfig) -> int:
            """Mock build_dataset to create symbol-specific CSV files."""
            # Parse symbols from config
            symbols = [s.strip() for s in cfg.symbols.split(",")]

            # Create dataset for each symbol
            for symbol in symbols:
                symbol_out_dir = Path(cfg.out_dir)
                symbol_out_dir.mkdir(parents=True, exist_ok=True)

                # Create mock dataset with symbol-specific data
                dataset_path = symbol_out_dir / "dataset.csv"

                # Create DataFrame with symbol-specific instrument_id
                df = pd.DataFrame(
                    {
                        "instrument_id": [f"{symbol}.NASDAQ"] * 100,
                        "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
                        "close": [100.0 + i for i in range(100)],
                        "volume": [1000 + i for i in range(100)],
                    },
                )

                df.to_csv(dataset_path, index=False)

            return 0

        # Use patch on the class method (works with __slots__)
        with patch.object(
            MLPipelineOrchestrator,
            "build_dataset",
            side_effect=mock_build_dataset,
        ):
            # Run orchestrator with multi-symbol config
            exit_code = mock_orchestrator.run(multi_symbol_config)

        # Assert: Pipeline completed successfully
        assert exit_code == 0, "Multi-symbol orchestration should complete successfully"

        # Assert: Separate datasets generated for each symbol
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            symbol_out_dir = out_dir / symbol
            dataset_path = symbol_out_dir / "dataset.csv"

            assert dataset_path.exists(), f"Dataset for {symbol} should exist at {dataset_path}"

            # Load dataset
            df = pd.read_csv(dataset_path)

            # Assert: No cross-contamination - all instrument_ids start with symbol
            assert df["instrument_id"].str.startswith(symbol).all(), (
                f"All instrument_ids in {symbol} dataset should start with {symbol}"
            )

        # Assert: Verify isolation - different symbols have different data
        aapl_df = pd.read_csv(out_dir / "AAPL" / "dataset.csv")
        googl_df = pd.read_csv(out_dir / "GOOGL" / "dataset.csv")
        msft_df = pd.read_csv(out_dir / "MSFT" / "dataset.csv")

        # Verify different instrument_ids
        assert aapl_df["instrument_id"].iloc[0] != googl_df["instrument_id"].iloc[0], (
            "AAPL and GOOGL datasets should have different instrument_ids"
        )
        assert googl_df["instrument_id"].iloc[0] != msft_df["instrument_id"].iloc[0], (
            "GOOGL and MSFT datasets should have different instrument_ids"
        )

    def test_multi_symbol_backward_compatibility(
        self,
        mock_orchestrator: MLPipelineOrchestrator,
        tmp_path: Path,
    ) -> None:
        """Test that single-symbol usage still works (backward compatibility).

        Property Under Test: Single-symbol processing is preserved

        Given:
        - Single instrument: AAPL
        - Standard pipeline configuration

        When:
        - Running orchestrator.run(config) with symbols="AAPL" (single symbol)

        Then:
        - Dataset generated in root output directory (not symbol subdirectory)
        - Existing behavior preserved

        """
        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create single-symbol config
        config = OrchestratorConfig(
            dataset=DatasetBuildConfig(
                data_dir=str(data_dir),
                symbols="AAPL",  # Single symbol
                out_dir=str(out_dir),
                start_iso="2024-01-01",
                end_iso="2024-01-31",
                target_semantics=build_default_target_semantics_payload(),
            ),
            hpo=None,
            teacher=TeacherTrainConfig(
                enabled=False,
                model_id="test_teacher",
                max_epochs=1,
            ),
            student=StudentDistillConfig(
                enabled=False,
                model_id="test_student",
            ),
            promotions=None,
            integration=None,
        )

        def mock_build_dataset(cfg: DatasetBuildConfig) -> int:
            """Mock build_dataset to create CSV file."""
            symbol_out_dir = Path(cfg.out_dir)
            symbol_out_dir.mkdir(parents=True, exist_ok=True)

            # Create mock dataset
            dataset_path = symbol_out_dir / "dataset.csv"
            df = pd.DataFrame(
                {
                    "instrument_id": ["AAPL.NASDAQ"] * 50,
                    "timestamp": pd.date_range("2024-01-01", periods=50, freq="1h"),
                    "close": [100.0 + i for i in range(50)],
                },
            )
            df.to_csv(dataset_path, index=False)
            return 0

        # Use patch on the class method (works with __slots__)
        with patch.object(
            MLPipelineOrchestrator,
            "build_dataset",
            side_effect=mock_build_dataset,
        ):
            # Run orchestrator
            exit_code = mock_orchestrator.run(config)

        # Assert: Pipeline completed successfully
        assert exit_code == 0

        # Assert: Dataset created in root output directory (not subdirectory)
        dataset_path = out_dir / "dataset.csv"
        assert dataset_path.exists(), "Dataset should exist in root output directory for single symbol"

        # Verify content
        df = pd.read_csv(dataset_path)
        assert df["instrument_id"].str.startswith("AAPL").all()
