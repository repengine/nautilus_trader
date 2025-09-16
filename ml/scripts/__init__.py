"""
Compatibility shim package for older import paths used in docs/tests.

Provides compatibility for:
- ml.scripts.apply_migrations -> ml.cli.apply_migrations
- ml.scripts.build_tft_dataset -> ml.cli.build_tft_dataset
"""

# Re-export submodules so attribute access via monkeypatch works
from . import apply_migrations  # noqa: F401
from . import build_tft_dataset  # noqa: F401
