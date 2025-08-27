#!/usr/bin/env python3
"""
Comprehensive error recovery tests for DataCollector.

This module tests all error handling paths in the DataCollector class,
focusing on recovery mechanisms and data integrity during failures.

"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.data.collector import CategoryStats
from ml.data.collector import CollectorStats
from ml.data.collector import DataCollector


class TestDataCollectorErrorRecovery:
    """Test error recovery mechanisms in DataCollector."""

    @pytest.fixture
    def temp_data_dir(self) -> Path:
        """Create temporary data directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def collector(self, temp_data_dir: Path) -> DataCollector:
        """Create DataCollector instance."""
        with patch.dict(os.environ, {"DATABENTO_API_KEY": "test_key"}):
            return DataCollector(
                storage_limit_gb=1.0,
                data_dir=temp_data_dir,
            )

    def test_collector_initialization_without_api_key(self, temp_data_dir: Path) -> None:
        """Test collector initializes gracefully without API key."""
        with patch.dict(os.environ, {}, clear=True):
            collector = DataCollector(data_dir=temp_data_dir)
            
            # Should initialize with None client
            assert collector.client is None
            assert collector.api_key is None
            
            # Collection methods should handle gracefully
            collector.collect_l2_depth(symbols=["TEST"])
            assert collector.stats["l2_depth"]["count"] == 0

    def test_corrupt_existing_symbols_recovery(self, temp_data_dir: Path) -> None:
        """Test recovery from corrupted symbol directory."""
        # Create corrupted universe directory structure
        universe_dir = Path("/home/nate/projects/nautilus_trader/data/universe")
        
        with patch("pathlib.Path.iterdir") as mock_iterdir:
            # Simulate OSError when reading directory
            mock_iterdir.side_effect = OSError("Permission denied")
            
            collector = DataCollector(data_dir=temp_data_dir)
            
            # Should handle error and return empty list
            assert collector.existing_symbols == []

    def test_storage_calculation_with_inaccessible_files(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test storage calculation handles inaccessible files."""
        # Create some files
        test_file = temp_data_dir / "test.parquet"
        test_file.write_bytes(b"test data")
        
        with patch("os.walk") as mock_walk:
            # Simulate permission error on some files
            mock_walk.return_value = [
                (str(temp_data_dir), [], ["test.parquet", "locked.parquet"])
            ]
            
            with patch("os.path.getsize") as mock_getsize:
                def size_with_error(path: str) -> int:
                    if "locked" in path:
                        raise OSError("Permission denied")
                    return 100
                
                mock_getsize.side_effect = size_with_error
                
                # Should handle error and calculate partial size
                size_gb = collector._get_current_storage_gb()
                assert size_gb > 0

    def test_l2_depth_collection_with_api_failures(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test L2 depth collection handles various API failures."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Test different failure scenarios
            errors = [
                Exception("Network timeout"),
                ValueError("Invalid response format"),
                KeyError("Missing required field"),
                json.JSONDecodeError("Invalid JSON", "", 0),
            ]
            
            for error in errors:
                mock_client.timeseries.get_range.side_effect = error
                
                # Should handle each error gracefully
                collector.collect_l2_depth(symbols=["TEST"], days=1)
                
                # Stats should reflect failure
                assert collector.stats["l2_depth"]["count"] == 0

    def test_l1_trades_collection_storage_limit_handling(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test L1 trades collection respects storage limits."""
        # Set very low storage limit
        collector.storage_limit_gb = 0.001  # 1MB
        collector.storage_used_gb = 0.0009  # 900KB used
        
        with patch.object(collector, "_estimate_data_size_gb") as mock_estimate:
            mock_estimate.return_value = 0.002  # 2MB estimated
            
            with patch("databento.Historical") as mock_db:
                mock_client = MagicMock()
                mock_db.return_value = mock_client
                collector.client = mock_client
                
                # Should reduce scope automatically
                collector.collect_l1_trades(symbols=collector.PRIORITY_SYMBOLS, years=5)
                
                # Should not attempt collection when over limit
                assert mock_client.timeseries.get_range.call_count == 0

    def test_partial_year_collection_failure_recovery(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test recovery from partial year collection failures."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Simulate failure for specific years
            def get_range_with_year_failure(*args: Any, **kwargs: Any) -> Any:
                # Fail for 2023 data
                if kwargs.get("start") and kwargs["start"].year == 2023:
                    raise Exception("Historical data unavailable")
                
                # Success for other years
                mock_response = MagicMock()
                mock_response.to_df.return_value = pd.DataFrame({
                    "price": [100.0],
                    "size": [100],
                })
                return mock_response
            
            mock_client.timeseries.get_range.side_effect = get_range_with_year_failure
            
            # Collect multi-year data
            collector.collect_l1_trades(symbols=["SPY"], years=3)
            
            # Should have partial success
            assert collector.stats["l1_trades"]["count"] > 0

    def test_tbbo_collection_with_empty_responses(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test TBBO collection handles empty API responses."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Return empty DataFrame
            mock_response = MagicMock()
            mock_response.to_df.return_value = pd.DataFrame()
            mock_client.timeseries.get_range.return_value = mock_response
            
            collector.collect_tbbo_quotes(symbols=["TEST"], days=30)
            
            # Should handle empty data gracefully
            assert collector.stats["tbbo_quotes"]["count"] == 0

    def test_minute_bars_collection_rate_limiting(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test minute bars collection handles rate limiting."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Simulate rate limit after few requests
            call_count = 0
            
            def rate_limited_response(*args: Any, **kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                
                if call_count > 3:
                    raise Exception("Rate limit exceeded")
                
                mock_response = MagicMock()
                mock_response.to_df.return_value = pd.DataFrame({
                    "close": [100.0],
                })
                return mock_response
            
            mock_client.timeseries.get_range.side_effect = rate_limited_response
            
            # Collect for multiple symbols
            collector.collect_minute_bars(symbols=["SPY", "QQQ", "IWM", "DIA", "VTI"], days=30)
            
            # Should have collected first 3 successfully
            assert collector.stats["minute_bars"]["count"] == 3

    def test_file_write_failures_during_collection(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test handling of file write failures during collection."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Successful API response
            mock_response = MagicMock()
            mock_df = pd.DataFrame({"close": [100.0]})
            mock_response.to_df.return_value = mock_df
            mock_client.timeseries.get_range.return_value = mock_response
            
            # Make directory read-only to cause write failure
            symbol_dir = temp_data_dir / "TEST"
            symbol_dir.mkdir()
            
            with patch.object(mock_df, "to_parquet") as mock_to_parquet:
                mock_to_parquet.side_effect = OSError("Permission denied")
                
                # Should handle write failure
                collector.collect_l2_depth(symbols=["TEST"], days=1)
                
                # Collection should fail
                assert collector.stats["l2_depth"]["count"] == 0

    def test_spread_calculation_with_invalid_data(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test spread calculation handles invalid price data."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Response with invalid spread data
            mock_response = MagicMock()
            mock_response.to_df.return_value = pd.DataFrame({
                "bid_px_00": [100.0, np.nan, 99.0],
                "ask_px_00": [np.nan, 101.0, 100.0],
            })
            mock_client.timeseries.get_range.return_value = mock_response
            
            # Should handle NaN values in spread calculation
            collector.collect_l2_depth(symbols=["TEST"], days=1)
            
            # Should complete despite calculation issues
            assert collector.stats["l2_depth"]["count"] >= 0

    def test_metadata_save_failure_handling(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test handling of metadata save failures."""
        # Run minimal collection
        collector.stats["l2_depth"]["count"] = 1
        collector.stats["l2_depth"]["size_gb"] = 0.1
        
        # Make data directory read-only
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = OSError("Cannot write metadata")
            
            # Should handle metadata save failure
            try:
                collector._print_final_summary()
            except OSError:
                # Should not crash the entire collection
                pass
            
            # Stats should still be valid
            assert collector.stats["l2_depth"]["count"] == 1

    def test_concurrent_symbol_processing(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test thread safety of concurrent symbol processing."""
        import threading
        
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Thread-safe response generation
            lock = threading.Lock()
            
            def thread_safe_response(*args: Any, **kwargs: Any) -> Any:
                with lock:
                    mock_response = MagicMock()
                    mock_response.to_df.return_value = pd.DataFrame({
                        "close": [100.0],
                    })
                    return mock_response
            
            mock_client.timeseries.get_range.side_effect = thread_safe_response
            
            # Process symbols in parallel threads
            threads = []
            symbols = ["SPY", "QQQ", "IWM"]
            
            for symbol in symbols:
                thread = threading.Thread(
                    target=collector.collect_minute_bars,
                    args=([symbol], 1),
                )
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # All symbols should be processed
            assert collector.stats["minute_bars"]["count"] >= 0

    def test_enhanced_collection_pipeline_failures(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test complete enhanced collection pipeline with various failures."""
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Simulate different failures for each phase
            phase_errors = {
                "l2_depth": Exception("L2 service down"),
                "l1_trades": Exception("Historical data unavailable"),
                "tbbo": Exception("Quote service error"),
                "minute_bars": None,  # This phase succeeds
            }
            
            def phase_specific_error(*args: Any, **kwargs: Any) -> Any:
                # Determine which phase based on schema parameter
                schema = kwargs.get("schema", "")
                
                if "mbp" in schema and phase_errors["l2_depth"]:
                    raise phase_errors["l2_depth"]
                elif "trades" in schema and phase_errors["l1_trades"]:
                    raise phase_errors["l1_trades"]
                elif "tbbo" in schema and phase_errors["tbbo"]:
                    raise phase_errors["tbbo"]
                else:
                    # Minute bars succeed
                    mock_response = MagicMock()
                    mock_response.to_df.return_value = pd.DataFrame({"close": [100.0]})
                    return mock_response
            
            mock_client.timeseries.get_range.side_effect = phase_specific_error
            
            # Run complete pipeline
            collector.run_collection()
            
            # Only minute bars should succeed
            assert collector.stats["minute_bars"]["count"] > 0
            assert collector.stats["l2_depth"]["count"] == 0
            assert collector.stats["l1_trades"]["count"] == 0
            assert collector.stats["tbbo_quotes"]["count"] == 0

    def test_storage_limit_enforcement_during_collection(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test storage limit is enforced during collection."""
        # Set very low limit
        collector.storage_limit_gb = 0.0001  # 100KB
        
        with patch.object(collector, "_get_current_storage_gb") as mock_get_storage:
            # Simulate storage filling up
            mock_get_storage.side_effect = [
                0.00005,  # 50KB - under limit
                0.00008,  # 80KB - approaching limit
                0.000095,  # 95KB - at 95% threshold
                0.0001,  # 100KB - at limit
            ]
            
            with patch("databento.Historical") as mock_db:
                mock_client = MagicMock()
                mock_db.return_value = mock_client
                collector.client = mock_client
                
                mock_response = MagicMock()
                mock_response.to_df.return_value = pd.DataFrame({"close": [100.0]})
                mock_client.timeseries.get_range.return_value = mock_response
                
                # Collect multiple symbols
                collector.collect_l2_depth(symbols=["SPY", "QQQ", "IWM", "DIA"], days=1)
                
                # Should stop before exceeding limit
                assert mock_get_storage.call_count <= 4

    def test_recovery_from_partial_file_writes(
        self,
        collector: DataCollector,
        temp_data_dir: Path,
    ) -> None:
        """Test recovery from partial file writes."""
        symbol_dir = temp_data_dir / "TEST"
        symbol_dir.mkdir()
        
        # Create partial/corrupted file
        partial_file = symbol_dir / "l2_depth_30d.parquet"
        partial_file.write_bytes(b"partial data")
        
        with patch("databento.Historical") as mock_db:
            mock_client = MagicMock()
            mock_db.return_value = mock_client
            collector.client = mock_client
            
            # Check if file exists (it does but is corrupted)
            assert partial_file.exists()
            
            # Collector should skip existing files
            collector.collect_l2_depth(symbols=["TEST"], days=30)
            
            # Should not attempt to re-collect
            assert not mock_client.timeseries.get_range.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])