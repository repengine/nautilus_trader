"""Unit tests for report generation."""

from pathlib import Path

import pytest

from ml.analysis.god_class_analyzer import (
    ClassAnalysis,
    MethodInfo,
    ReportGenerator,
    ResponsibilityGroup,
    SharedPattern,
)


class TestReportGenerator:
    """Tests for ReportGenerator component."""

    @pytest.fixture
    def sample_analyses(self) -> list[ClassAnalysis]:
        """Create sample analyses for testing."""
        return [
            ClassAnalysis(
                class_name="DataStore",
                file_path=Path("ml/stores/data_store_facade.py"),
                line_count=3730,
                method_count=45,
                methods={
                    "read_data": MethodInfo(name="read_data", line_number=100),
                    "write_data": MethodInfo(name="write_data", line_number=200),
                    "validate_schema": MethodInfo(name="validate_schema", line_number=300)
                },
                method_groups={
                    "prefix:read": ["read_data"],
                    "prefix:write": ["write_data"],
                    "prefix:validate": ["validate_schema"]
                },
                call_graph={
                    "read_data": [],
                    "write_data": ["validate_schema"],
                    "validate_schema": []
                },
                cohesion_metrics={"overall": 0.65},
                responsibilities=[
                    ResponsibilityGroup(
                        name="Data Reading",
                        methods=["read_data"],
                        description="Reads data from storage",
                        cohesion_score=1.0,
                        extraction_recommendation="Extract to DataReader component"
                    ),
                    ResponsibilityGroup(
                        name="Data Writing",
                        methods=["write_data"],
                        description="Writes data to storage",
                        cohesion_score=1.0,
                        extraction_recommendation="Extract to DataWriter component"
                    )
                ],
                patterns=["validation", "database"]
            ),
            ClassAnalysis(
                class_name="FeatureStore",
                file_path=Path("ml/stores/feature_store_facade.py"),
                line_count=1677,
                method_count=25,
                methods={
                    "read_features": MethodInfo(name="read_features", line_number=50),
                    "write_features": MethodInfo(name="write_features", line_number=150)
                },
                method_groups={
                    "prefix:read": ["read_features"],
                    "prefix:write": ["write_features"]
                },
                call_graph={
                    "read_features": [],
                    "write_features": []
                },
                cohesion_metrics={"overall": 0.75},
                responsibilities=[
                    ResponsibilityGroup(
                        name="Feature Reading",
                        methods=["read_features"],
                        description="Reads features",
                        cohesion_score=1.0,
                        extraction_recommendation="Extract to FeatureReader"
                    )
                ],
                patterns=["validation"]
            )
        ]

    @pytest.fixture
    def sample_patterns(self) -> list[SharedPattern]:
        """Create sample shared patterns for testing."""
        return [
            SharedPattern(
                pattern_name="validation",
                pattern_type="validation",
                affected_classes=["DataStore", "FeatureStore", "ModelStore"],
                description="Schema validation pattern shared across 3 classes",
                extraction_recommendation="Extract to ml/common/validation.py",
                priority=3
            ),
            SharedPattern(
                pattern_name="database",
                pattern_type="database",
                affected_classes=["DataStore", "FeatureStore"],
                description="Database operations pattern shared across 2 classes",
                extraction_recommendation="Extract to ml/common/database.py",
                priority=2
            )
        ]

    def test_responsibility_catalog_generation(
        self,
        sample_analyses: list[ClassAnalysis],
        tmp_path: Path
    ) -> None:
        """Verify responsibility catalog report generation."""
        generator = ReportGenerator()
        output_path = tmp_path / "catalog.md"

        generator.generate_responsibility_catalog(sample_analyses, output_path)

        assert output_path.exists()
        content = output_path.read_text()

        # Check required sections
        assert "# God Class Responsibility Catalog" in content
        assert "**Total Classes Analyzed:**" in content
        assert "DataStore" in content
        assert "FeatureStore" in content
        assert "### Responsibilities" in content
        assert "### Method Groups" in content

    def test_cross_class_pattern_analysis_generation(
        self,
        sample_patterns: list[SharedPattern],
        tmp_path: Path
    ) -> None:
        """Verify cross-class pattern report generation."""
        generator = ReportGenerator()
        output_path = tmp_path / "patterns.md"

        generator.generate_cross_class_pattern_report(sample_patterns, output_path)

        assert output_path.exists()
        content = output_path.read_text()

        # Check structure
        assert "# Cross-Class Pattern Analysis" in content
        assert "**Total Patterns Found:**" in content
        assert "## Pattern: validation" in content
        assert "## Pattern: database" in content
        assert "### Description" in content
        assert "### Affected Classes" in content
        assert "### Extraction Recommendation" in content

    def test_shared_utility_candidates_generation(
        self,
        sample_patterns: list[SharedPattern],
        tmp_path: Path
    ) -> None:
        """Verify shared utility candidates report generation."""
        generator = ReportGenerator()
        output_path = tmp_path / "utilities.md"

        generator.generate_shared_utilities_report(sample_patterns, output_path)

        assert output_path.exists()
        content = output_path.read_text()

        # Check structure
        assert "# Shared Utility Extraction Candidates" in content
        assert "**Total Candidates:**" in content
        assert "Priority:" in content
        assert "### Extraction Target" in content
        assert "### Impact" in content

        # Verify prioritization (validation should come first with priority 3)
        validation_pos = content.find("validation")
        database_pos = content.find("database")
        assert validation_pos < database_pos  # Higher priority comes first

    def test_extraction_strategy_generation(
        self,
        sample_analyses: list[ClassAnalysis],
        tmp_path: Path
    ) -> None:
        """Verify extraction strategy report generation."""
        generator = ReportGenerator()
        output_path = tmp_path / "strategy.md"

        generator.generate_extraction_strategy(
            sample_analyses,
            output_path,
            title="Test Strategy",
            description="Test extraction strategy"
        )

        assert output_path.exists()
        content = output_path.read_text()

        # Check structure
        assert "# Test Strategy" in content
        assert "Test extraction strategy" in content
        assert "**Classes Covered:**" in content
        assert "### Extraction Approach" in content
        assert "### Dependencies" in content

    def test_report_markdown_formatting(
        self,
        sample_analyses: list[ClassAnalysis],
        tmp_path: Path
    ) -> None:
        """Verify generated reports use valid markdown syntax."""
        generator = ReportGenerator()
        output_path = tmp_path / "test.md"

        generator.generate_responsibility_catalog(sample_analyses, output_path)

        content = output_path.read_text()

        # Check basic markdown formatting
        assert content.count("#") > 0  # Has headers
        assert content.count("**") % 2 == 0  # Bold tags are balanced
        assert content.count("`") % 2 == 0  # Code tags are balanced
        assert "---" in content  # Has horizontal rules

    def test_report_completeness(
        self,
        sample_analyses: list[ClassAnalysis],
        sample_patterns: list[SharedPattern],
        tmp_path: Path
    ) -> None:
        """Verify all required information is present in reports."""
        generator = ReportGenerator()

        # Test responsibility catalog completeness
        catalog_path = tmp_path / "catalog.md"
        generator.generate_responsibility_catalog(sample_analyses, catalog_path)
        catalog_content = catalog_path.read_text()

        # Should include all classes
        for analysis in sample_analyses:
            assert analysis.class_name in catalog_content
            assert str(analysis.file_path) in catalog_content
            assert str(analysis.line_count) in catalog_content
            assert str(analysis.method_count) in catalog_content

        # Test pattern report completeness
        pattern_path = tmp_path / "patterns.md"
        generator.generate_cross_class_pattern_report(sample_patterns, pattern_path)
        pattern_content = pattern_path.read_text()

        # Should include all patterns
        for pattern in sample_patterns:
            assert pattern.pattern_name in pattern_content
            for class_name in pattern.affected_classes:
                assert class_name in pattern_content

    def test_report_generation_with_empty_data(self, tmp_path: Path) -> None:
        """Verify reports handle empty data gracefully."""
        generator = ReportGenerator()

        # Empty analyses
        catalog_path = tmp_path / "empty_catalog.md"
        generator.generate_responsibility_catalog([], catalog_path)
        assert catalog_path.exists()
        content = catalog_path.read_text()
        assert "**Total Classes Analyzed:** 0" in content

        # Empty patterns
        pattern_path = tmp_path / "empty_patterns.md"
        generator.generate_cross_class_pattern_report([], pattern_path)
        assert pattern_path.exists()
        content = pattern_path.read_text()
        assert "**Total Patterns Found:** 0" in content

    def test_report_file_creation_in_nested_directory(
        self,
        sample_analyses: list[ClassAnalysis],
        tmp_path: Path
    ) -> None:
        """Verify reports can be created in nested directories."""
        generator = ReportGenerator()

        # Create nested path that doesn't exist yet
        nested_path = tmp_path / "reports" / "analysis" / "catalog.md"

        generator.generate_responsibility_catalog(sample_analyses, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()
