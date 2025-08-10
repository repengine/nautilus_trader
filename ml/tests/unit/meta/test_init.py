
"""
Test ML package initialization.
"""

import sys
from pathlib import Path


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import ml


def test_ml_version() -> None:
    """
    Test that ML package has version.
    """
    assert hasattr(ml, "__version__")
    assert ml.__version__ == "0.1.0"


def test_ml_docstring() -> None:
    """
    Test that ML package has proper docstring.
    """
    assert ml.__doc__ is not None
    assert "Nautilus ML" in ml.__doc__
    assert "Machine Learning Integration" in ml.__doc__
