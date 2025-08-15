from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_tft_teacher_fit_predict_smoke() -> None:
    # Skip if heavy deps missing
    pf = pytest.importorskip("pytorch_forecasting", reason="pytorch_forecasting not installed")
    pl = pytest.importorskip("pytorch_lightning", reason="pytorch_lightning not installed")
    # Skip if the installed TFT is not a LightningModule subclass (version incompatibility)
    try:
        if not issubclass(pf.TemporalFusionTransformer, pl.LightningModule):
            pytest.skip("TFT class is not a LightningModule in this environment")
    except Exception:
        pytest.skip("Unable to validate TFT class inheritance; skipping smoke test")

    from ml.training.teacher.tft_teacher import TFTTeacher, TFTTeacherConfig

    n = 40
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "time_index": np.arange(n, dtype=np.int64),
            "instrument_id": ["X"] * n,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    df["y"] = (df["f1"] + 0.2 * df["f2"] + rng.normal(scale=0.1, size=n) > 0).astype(int)

    teacher = TFTTeacher(
        TFTTeacherConfig(),
        max_encoder_length=8,
        max_prediction_length=1,
        time_varying_unknown_reals=["f1", "f2", "f3"],
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        max_epochs=1,
    )

    teacher.fit(df)
    z = teacher.predict_logits(df)
    assert z.shape[0] == len(df)
    assert z.dtype == np.float64
