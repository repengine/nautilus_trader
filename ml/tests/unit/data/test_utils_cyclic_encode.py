from __future__ import annotations

import math

from ml.data.providers.utils import cyclic_encode


def test_cyclic_encode_maps_to_unit_circle() -> None:
    for value in [0.0, 1.0, 6.0, 12.0, 18.0, 23.0]:
        s, c = cyclic_encode(value, period=24)
        assert math.isclose(s * s + c * c, 1.0, rel_tol=1e-6, abs_tol=1e-6)
