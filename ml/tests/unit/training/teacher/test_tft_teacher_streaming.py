from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml._imports import HAS_PANDAS
from ml._imports import HAS_TORCH
from ml._imports import check_ml_dependencies
from ml._imports import pd
from ml._imports import torch as _torch
from ml.training.teacher import streaming_loader as stream
from ml.training.teacher.streaming_loader import TFTStreamingConfig
from ml.training.teacher.streaming_loader import split_metadata_by_row_fraction
from ml.training.teacher.tft_teacher import TFTTeacher
from ml.training.teacher.tft_teacher import TFTTeacherConfig

try:  # Optional dependency gate for PyTorch Forecasting
    from pytorch_forecasting import TemporalFusionTransformer  # noqa: F401

    HAS_PYTORCH_FORECASTING = True
except Exception:  # pragma: no cover - dependency missing
    HAS_PYTORCH_FORECASTING = False


@pytest.mark.skipif(
    not (HAS_PANDAS and HAS_TORCH and HAS_PYTORCH_FORECASTING),
    reason="Streaming training requires pandas, torch, and pytorch-forecasting",
)
def test_fit_streaming_returns_logits(tmp_path: Path) -> None:
    """Ensure TFTTeacher.fit_streaming trains on capped streaming data."""
    if pd is None:
        check_ml_dependencies(["pandas"])
        pytest.skip("pandas import guard triggered")

    frame = pd.DataFrame(
        {
            "time_index": np.arange(20, dtype=np.int64),
            "instrument_id": ["AAPL"] * 10 + ["MSFT"] * 10,
            "y": np.linspace(0.0, 1.0, num=20, dtype=np.float32),
            "feature": np.linspace(10.0, 30.0, num=20, dtype=np.float32),
        },
    )
    parquet_path = tmp_path / "stream_train.parquet"
    frame.to_parquet(parquet_path, index=False)

    metadata = stream.collect_streaming_metadata(
        parquet_path,
        feature_names=("feature", "y"),
        categorical_columns=("instrument_id",),
        numeric_columns=("feature", "y"),
        group_id_col="instrument_id",
        time_index_col="time_index",
        target_col="y",
        shard_row_budget=10,
    )
    metadata = stream.filter_metadata_by_instruments(metadata, ["AAPL"])
    train_meta, val_meta = split_metadata_by_row_fraction(metadata, train_fraction=0.8)

    config = TFTStreamingConfig(
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        max_encoder_length=4,
        max_prediction_length=1,
        batch_size=4,
        drop_last=False,
        shuffle_shards=False,
        seed=7,
        num_workers=0,
        max_shards=2,
        max_total_rows=10,
        max_total_sequences=100,
    )

    train_loader = stream.build_streaming_dataloader(parquet_path, train_meta, config)
    val_loader = stream.build_streaming_dataloader(parquet_path, val_meta, config)

    teacher = TFTTeacher(
        TFTTeacherConfig(architecture="TFT"),
        max_encoder_length=4,
        max_prediction_length=1,
        static_categoricals=("instrument_id",),
        static_reals=(),
        time_varying_known_reals=(),
        time_varying_unknown_reals=("feature",),
        time_idx_col="time_index",
        group_id_col="instrument_id",
        target_col="y",
        max_epochs=1,
        batch_size=4,
        dataloader_workers=0,
    )
    result = teacher.fit_streaming(
        parquet_path=parquet_path,
        train_loader=train_loader,
        val_loader=val_loader,
        train_metadata=train_meta,
        val_metadata=val_meta,
        full_metadata=metadata,
        streaming_config=config,
    )

    assert result.z_train.size > 0
    assert result.z_val.size > 0
    assert result.y_val.size == result.z_val.size
    assert result.val_rows is not None
    assert set(result.val_rows.instrument_ids.tolist()) == {"AAPL"}


@pytest.mark.skipif(
    not (HAS_TORCH and _torch is not None),
    reason="Torch required to verify device alignment",
)
def test_collect_streaming_logits_aligns_with_model_device() -> None:
    torch = pytest.importorskip("torch")

    class _CudaStub:
        def __init__(self, device: torch.device) -> None:
            self._device = device
            self._param = torch.nn.Parameter(torch.zeros(1, device=device))
            self.seen_devices: set[str] = set()

        def parameters(self):
            yield self._param

        def eval(self) -> _CudaStub:
            return self

        def __call__(self, batch_inputs):
            for tensor in batch_inputs.values():
                if hasattr(tensor, "device"):
                    self.seen_devices.add(str(tensor.device))
            prediction = batch_inputs["decoder_target"].detach().clone().to(self._device)
            return {"prediction": prediction}

    teacher = TFTTeacher(
        TFTTeacherConfig(loss_name="bce"),
        max_encoder_length=1,
        max_prediction_length=1,
        dataloader_workers=0,
        batch_size=1,
    )
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    stub = _CudaStub(device)
    teacher._tft = stub  # type: ignore[attribute-defined-outside-init]

    batch_inputs = {
        "encoder_cont": torch.zeros((1, 1, 1), dtype=torch.float32),
        "decoder_cont": torch.zeros((1, 1, 1), dtype=torch.float32),
        "encoder_cat": torch.zeros((1, 1, 0), dtype=torch.int64),
        "decoder_cat": torch.zeros((1, 1, 0), dtype=torch.int64),
        "encoder_target": torch.zeros((1, 1), dtype=torch.float32),
        "decoder_target": torch.zeros((1, 1), dtype=torch.float32),
        "encoder_lengths": torch.ones((1,), dtype=torch.int64),
        "decoder_lengths": torch.ones((1,), dtype=torch.int64),
        "groups": torch.tensor([[0]], dtype=torch.int64),
        "target_scale": torch.ones((1, 2), dtype=torch.float32),
        "decoder_time_idx": torch.tensor([[123]], dtype=torch.int64),
        "decoder_group_ids": torch.tensor([[0]], dtype=torch.int64),
    }
    decoder_target = torch.zeros((1, 1), dtype=torch.float32)
    loader = [
        (
            batch_inputs,
            (decoder_target, None),
        ),
    ]
    group_inverse_map = {0: "AAPL"}
    logits, y, rows = teacher._collect_streaming_logits(loader, torch, group_inverse_map=group_inverse_map)
    assert logits.shape == (1,)
    assert y.shape == (1,)
    assert rows is not None
    assert stub.seen_devices == {str(device)}
