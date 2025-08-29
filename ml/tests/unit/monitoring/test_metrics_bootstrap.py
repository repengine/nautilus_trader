from __future__ import annotations

from ml.common.metrics_bootstrap import get_counter
from ml.common.metrics_bootstrap import get_gauge
from ml.common.metrics_bootstrap import get_histogram


def test_metrics_bootstrap_idempotent() -> None:
    c1 = get_counter("test_counter_bootstrap", "desc", ["label_a"])
    c2 = get_counter("test_counter_bootstrap", "desc", ["label_a"])
    assert c1 is c2

    h1 = get_histogram("test_hist_bootstrap", "desc", ["label_b"], buckets=(0.1, 1.0))
    h2 = get_histogram("test_hist_bootstrap", "desc", ["label_b"], buckets=(0.1, 1.0))
    assert h1 is h2

    g1 = get_gauge("test_gauge_bootstrap", "desc", ["label_c"])
    g2 = get_gauge("test_gauge_bootstrap", "desc", ["label_c"])
    assert g1 is g2
