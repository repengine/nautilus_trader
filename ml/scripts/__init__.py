"""
Compatibility shim package for older import paths used in tests.

Provides ml.scripts.build_tft_dataset -> ml.cli.build_tft_dataset.
"""

# Re-export submodules so attribute access via monkeypatch works
from . import build_tft_dataset  # noqa: F401
