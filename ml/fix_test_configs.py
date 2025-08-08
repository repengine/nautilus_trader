#!/usr/bin/env python3
"""
Script to fix test configurations for the new nested config structure.
"""

import re
from pathlib import Path


def fix_config_fields(content: str) -> str:
    """
    Fix config field references to use nested configs.
    """
    # Pattern to find UnifiedXGBoostConfig or UnifiedLightGBMConfig instantiations
    config_pattern = r"(Unified(?:XGBoost|LightGBM)Config\([^)]*)"

    def replace_config(match):
        config_str = match.group(1)

        # Check if we need to add imports
        needs_advanced_import = any(
            field in config_str
            for field in [
                "track_feature_decay=",
                "export_onnx=",
                "enable_monitoring=",
                "onnx_output_path=",
            ]
        )

        if not needs_advanced_import:
            return config_str

        # Extract the fields that need to be moved to advanced_config
        advanced_fields = []
        remaining_str = config_str

        for field in [
            "track_feature_decay",
            "export_onnx",
            "enable_monitoring",
            "onnx_output_path",
        ]:
            pattern = f"{field}=([^,)]+)"
            match = re.search(pattern, remaining_str)
            if match:
                advanced_fields.append(f"{field}={match.group(1)}")
                remaining_str = re.sub(pattern + r",?\s*", "", remaining_str)

        if advanced_fields:
            # Add advanced_config
            advanced_config_str = (
                f"advanced_config=AdvancedTrainingConfig({', '.join(advanced_fields)})"
            )

            # Insert it into the config
            if remaining_str.endswith("("):
                remaining_str += f"\n            {advanced_config_str},"
            else:
                remaining_str = remaining_str.rstrip(",") + f",\n            {advanced_config_str},"

        return remaining_str

    # Apply the replacement
    content = re.sub(config_pattern, replace_config, content)

    # Ensure AdvancedTrainingConfig is imported if needed
    if (
        "AdvancedTrainingConfig" in content
        and "from ml.config.shared import AdvancedTrainingConfig" not in content
    ):
        # Add the import after other ml.config imports
        import_pattern = r"(from ml\.config\.[^\n]+\n)"
        if re.search(import_pattern, content):
            content = re.sub(
                import_pattern,
                r"\1from ml.config.shared import AdvancedTrainingConfig\n",
                content,
                count=1,
            )
        else:
            # Add at the beginning of imports
            content = re.sub(
                r"(import [^\n]+\n)",
                r"\1from ml.config.shared import AdvancedTrainingConfig\n",
                content,
                count=1,
            )

    return content


def main():
    # Find all test files that need fixing
    test_files = [
        Path("ml/tests/unit/test_xgboost_unified.py"),
        Path("ml/tests/unit/test_lightgbm_unified.py"),
        Path("ml/tests/integration/test_xgboost_unified_integration.py"),
        Path("ml/tests/integration/test_lightgbm_unified_integration.py"),
        Path("ml/tests/qa_functional_test.py"),
        Path("ml/tests/qa_integration_test.py"),
    ]

    for test_file in test_files:
        if not test_file.exists():
            print(f"Skipping {test_file} - not found")
            continue

        print(f"Processing {test_file}...")

        content = test_file.read_text()
        original_content = content

        # Fix the config fields
        content = fix_config_fields(content)

        if content != original_content:
            test_file.write_text(content)
            print(f"  Updated {test_file}")
        else:
            print(f"  No changes needed for {test_file}")


if __name__ == "__main__":
    main()
