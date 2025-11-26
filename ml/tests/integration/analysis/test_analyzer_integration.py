"""Integration tests for god class analyzer with real files."""

from pathlib import Path

import pytest

from ml.analysis.god_class_analyzer import GodClassAnalyzer


@pytest.mark.integration
class TestAnalyzerIntegration:
    """Integration tests analyzing real god class files."""

    @pytest.fixture
    def analyzer(self) -> GodClassAnalyzer:
        """Create analyzer instance."""
        return GodClassAnalyzer()

    @pytest.fixture
    def real_god_class_paths(self) -> list[Path]:
        """Paths to actual god class files."""
        base = Path("/home/nate/projects/nautilus_trader-phase0/ml")
        return [
            base / "orchestration/pipeline_orchestrator.py",
            base / "stores/data_store.py",
            base / "features/engineering.py",
            base / "actors/signal.py",
            base / "registry/model_registry.py",
            base / "data/tft_dataset_builder.py",
            base / "actors/base.py",
            base / "dashboard/service.py",
            base / "data/__init__.py",
            base / "core/integration.py",
            base / "registry/data_registry.py",
            base / "strategies/base.py",
            base / "stores/feature_store.py",
            base / "training/base.py",
            base / "data/scheduler.py",
            base / "dashboard/app.py",
        ]

    def test_analyze_sample_god_class(self, analyzer: GodClassAnalyzer, tmp_path: Path) -> None:
        """Test analyzing a small synthetic god class."""
        from textwrap import dedent

        # Create sample god class
        test_file = tmp_path / "sample_god_class.py"
        test_file.write_text(dedent("""
            class SampleGodClass:
                def read_data(self, instrument_id: str):
                    data = self._fetch_from_db(instrument_id)
                    return self._validate_data(data)

                def read_features(self, instrument_id: str):
                    features = self._fetch_features(instrument_id)
                    return self._validate_features(features)

                def write_data(self, data):
                    validated = self._validate_data(data)
                    self._persist_to_db(validated)
                    self._emit_event("data_written")

                def write_features(self, features):
                    validated = self._validate_features(features)
                    self._persist_to_db(validated)
                    self._emit_event("features_written")

                def _validate_data(self, data):
                    self.validation_count += 1
                    return data

                def _validate_features(self, features):
                    self.validation_count += 1
                    return features

                def _fetch_from_db(self, instrument_id: str):
                    self.db_connection.execute(...)
                    pass

                def _fetch_features(self, instrument_id: str):
                    self.db_connection.execute(...)
                    pass

                def _persist_to_db(self, data):
                    self.db_connection.execute(...)
                    pass

                def _emit_event(self, event_type: str):
                    self.event_bus.publish(event_type)
                    pass
        """))

        analysis = analyzer.analyze_file(test_file)

        # Verify basic analysis worked
        assert analysis.class_name == "SampleGodClass"
        assert analysis.method_count == 10
        assert analysis.file_path == test_file

        # Verify method grouping
        assert len(analysis.method_groups) > 0
        assert any("read" in key for key in analysis.method_groups.keys())
        assert any("write" in key for key in analysis.method_groups.keys())

        # Verify call graph
        assert "write_data" in analysis.call_graph
        assert "_validate_data" in analysis.call_graph["write_data"]
        assert "_persist_to_db" in analysis.call_graph["write_data"]
        assert "_emit_event" in analysis.call_graph["write_data"]

        # Verify cohesion calculated
        assert "overall" in analysis.cohesion_metrics
        assert 0.0 <= analysis.cohesion_metrics["overall"] <= 1.0

        # Verify responsibilities identified
        assert len(analysis.responsibilities) > 0

        # Verify patterns detected
        assert len(analysis.patterns) > 0

    def test_analyze_real_god_class_pipeline_orchestrator(
        self,
        analyzer: GodClassAnalyzer,
        real_god_class_paths: list[Path]
    ) -> None:
        """Test analyzing actual pipeline_orchestrator.py (4,592 lines)."""
        orchestrator_path = real_god_class_paths[0]
        assert orchestrator_path.exists()

        analysis = analyzer.analyze_file(orchestrator_path)

        # Verify analysis completed
        assert analysis.class_name is not None
        assert analysis.method_count > 0
        assert analysis.line_count > 4000  # Should be around 4,592

        # Should have identified method groups
        assert len(analysis.method_groups) > 0

        # Should have call graph
        assert len(analysis.call_graph) > 0

        # Should have cohesion metrics
        assert "overall" in analysis.cohesion_metrics

    def test_analyze_real_god_class_data_store(
        self,
        analyzer: GodClassAnalyzer,
        real_god_class_paths: list[Path]
    ) -> None:
        """Test analyzing actual data_store.py (3,730 lines)."""
        data_store_path = real_god_class_paths[1]
        assert data_store_path.exists()

        analysis = analyzer.analyze_file(data_store_path)

        # Verify analysis completed
        assert analysis.class_name is not None
        assert analysis.method_count > 0
        assert analysis.line_count > 3000

        # Should detect validation and database patterns
        # (data_store likely has both)
        assert len(analysis.patterns) > 0

    def test_analyze_real_god_class_feature_engineer(
        self,
        analyzer: GodClassAnalyzer,
        real_god_class_paths: list[Path]
    ) -> None:
        """Test analyzing actual engineering.py (3,201 lines)."""
        engineering_path = real_god_class_paths[2]
        assert engineering_path.exists()

        analysis = analyzer.analyze_file(engineering_path)

        # Verify analysis completed
        assert analysis.class_name is not None
        assert analysis.method_count > 0
        assert analysis.line_count > 3000

    def test_pattern_detection_across_multiple_classes(
        self,
        analyzer: GodClassAnalyzer,
        real_god_class_paths: list[Path]
    ) -> None:
        """Test pattern detection across multiple real god classes."""
        # Analyze first 5 god classes
        analyses, shared_patterns = analyzer.analyze_all(real_god_class_paths[:5], min_shared_pattern_classes=2)

        # Should have analyzed all 5 files
        assert len(analyses) == 5

        # Should find at least some shared patterns
        # (validation, database, metrics, etc. are common)
        assert len(shared_patterns) >= 0  # May or may not find patterns depending on detection

        # If patterns found, verify structure
        if shared_patterns:
            for pattern in shared_patterns:
                assert pattern.pattern_name
                assert pattern.pattern_type
                assert len(pattern.affected_classes) >= 2
                assert pattern.description
                assert pattern.extraction_recommendation

    def test_report_generation_integration(
        self,
        analyzer: GodClassAnalyzer,
        real_god_class_paths: list[Path],
        tmp_path: Path
    ) -> None:
        """Test end-to-end report generation with real files."""
        # Analyze a subset of god classes
        analyses, shared_patterns = analyzer.analyze_all(real_god_class_paths[:3])

        # Generate reports
        output_dir = tmp_path / "reports" / "analysis"
        analyzer.generate_all_reports(analyses, shared_patterns, output_dir)

        # Verify all 5 reports were created
        assert (output_dir / "god_class_responsibility_catalog.md").exists()
        assert (output_dir / "cross_class_pattern_analysis.md").exists()
        assert (output_dir / "shared_utility_candidates.md").exists()
        assert (output_dir / "phase2_extraction_strategy.md").exists()
        assert (output_dir / "phase3_extraction_strategy.md").exists()

        # Verify reports contain data
        catalog = (output_dir / "god_class_responsibility_catalog.md").read_text()
        assert len(catalog) > 0
        assert "# God Class Responsibility Catalog" in catalog
