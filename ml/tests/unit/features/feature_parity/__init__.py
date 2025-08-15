"""
Feature parity validation tests.

This package contains comprehensive tests to ensure perfect feature parity between batch
(training) and online (inference) feature computations.

The tests validate that feature calculations produce identical results with < 1e-10
tolerance between batch and streaming modes.

"""
