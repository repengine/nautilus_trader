
"""
Test ML package initialization.
"""

import ml


def test_ml_version():
    """
    Test that ML package has version.
    """
    assert hasattr(ml, "__version__")
    assert ml.__version__ == "0.1.0"


def test_ml_docstring():
    """
    Test that ML package has proper docstring.
    """
    assert ml.__doc__ is not None
    assert "Nautilus ML" in ml.__doc__
    assert "Machine Learning Integration" in ml.__doc__
    assert "Cold Path" in ml.__doc__
    assert "Hot Path" in ml.__doc__
