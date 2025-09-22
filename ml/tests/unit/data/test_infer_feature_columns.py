from __future__ import annotations

import pandas as pd

from ml.data import _infer_feature_columns


def test_infer_feature_columns_excludes_forward_return() -> None:
    df = pd.DataFrame(
        {
            "time_index": [0, 1],
            "feature_a": [0.1, 0.2],
            "forward_return": [0.05, -0.01],
            "y": [0, 1],
        },
    )
    cols = _infer_feature_columns(df)
    assert "feature_a" in cols
    assert "forward_return" not in cols
    assert "y" not in cols
