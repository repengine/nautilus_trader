from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

from ml._imports import HAS_TORCH, check_ml_dependencies, torch as _torch

# Ensure torch is available at runtime (training-only dependency)
if not HAS_TORCH or _torch is None:  # pragma: no cover - environment dependent
    check_ml_dependencies(["torch"])  # raises with helpful message

torch = _torch  # alias for local use
if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn
else:
    nn = cast(Any, _torch).nn


class TFTScriptAdapter(nn.Module):
    """
    Wraps a TFT-like module that expects keyword tensor inputs into a module
    that accepts a positional list/tuple of tensors in a fixed key order.
    """

    def __init__(self, model: nn.Module, input_keys: list[str]) -> None:
        super().__init__()
        self.model = model
        self.input_keys = list(input_keys)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        inputs: dict[str, torch.Tensor] = {k: v for k, v in zip(self.input_keys, args)}
        out_obj: object = self.model(**inputs)
        if isinstance(out_obj, torch.Tensor):
            return out_obj
        # Fallback if model returns a mapping
        if isinstance(out_obj, dict):
            # Prefer common key names
            from typing import cast
            for key in ("pred", "prediction", "logits"):
                if key in out_obj and isinstance(out_obj[key], torch.Tensor):
                    return cast(torch.Tensor, out_obj[key])
            # Else take first tensor value
            for v in out_obj.values():
                if isinstance(v, torch.Tensor):
                    return v
        raise RuntimeError("Unexpected TFT output type; cannot adapt to TorchScript")


def export_tft_to_torchscript_from_batch(
    tft_module: nn.Module,
    batch_x: dict[str, torch.Tensor],
    out_path: str | Path,
    key_filter: Iterable[str] | None = None,
) -> Path:
    """
    Export a TFT-like module to TorchScript using a real batch dict of tensors.

    Parameters
    ----------
    tft_module : nn.Module
        Trained TFT (LightningModule/nn.Module) with forward(**kwargs).
    batch_x : dict[str, Tensor]
        Batch inputs as provided by a TimeSeriesDataSet dataloader (x part).
    out_path : Path-like
        Output file path without suffix; '.pt' is added.
    key_filter : Iterable[str] | None
        Optional explicit list of keys to include; if None, uses tensor-valued keys
        in sorted order for stability.

    Returns
    -------
    Path
        Saved TorchScript file path (.pt).
    """
    tft_module.eval().cpu()

    # Select tensor inputs
    tensor_items = {k: v.detach().cpu().contiguous() for k, v in batch_x.items() if torch.is_tensor(v)}
    if key_filter is None:
        input_keys = sorted(tensor_items.keys())
    else:
        input_keys = [k for k in key_filter if k in tensor_items]
    if not input_keys:
        raise ValueError("No tensor inputs found for TorchScript export")

    example_args = tuple(tensor_items[k] for k in input_keys)
    adapter = TFTScriptAdapter(tft_module, input_keys)
    def _jit_trace(mod: object, example: object) -> Any:  # noqa: ANN401
        return torch.jit.trace(mod, example)  # type: ignore[no-untyped-call]

    with torch.inference_mode():
        scripted = _jit_trace(adapter, example_args)
    out_path = Path(out_path).with_suffix(".pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    return out_path
