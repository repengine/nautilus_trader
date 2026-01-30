"""
Prediction surface normalization helpers for ML inference and strategy decisions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt


__all__ = [
    "decision_from_probability",
    "neutral_band_bounds",
    "normalize_prediction_batch",
    "normalize_prediction_output",
    "resolve_output_is_logits",
]


def resolve_output_is_logits(metadata: Mapping[str, Any] | None) -> bool:
    """
    Resolve whether model outputs should be interpreted as logits.

    Parameters
    ----------
    metadata : Mapping[str, Any] | None
        Model metadata that may include output flags.

    Returns
    -------
    bool
        True when logits interpretation is enabled.

    """
    if not metadata:
        return False
    if bool(metadata.get("output_is_logits")) or bool(metadata.get("onnx_output_is_logits")):
        return True
    decision_cfg = metadata.get("decision_config")
    if isinstance(decision_cfg, Mapping):
        return bool(
            decision_cfg.get("output_is_logits") or decision_cfg.get("onnx_output_is_logits"),
        )
    return False


def normalize_prediction_output(
    prediction: Any,
    confidence: Any | None,
    *,
    classes: Sequence[Any] | None = None,
    output_is_logits: bool = False,
) -> tuple[float, float]:
    """
    Normalize raw model outputs into canonical probability + confidence.

    Parameters
    ----------
    prediction : Any
        Raw model prediction output (scalar or probability vector).
    confidence : Any | None
        Optional confidence output (scalar).
    classes : Sequence[Any] | None, optional
        Optional classifier class labels (used to select positive class prob).
    output_is_logits : bool, default False
        Whether the prediction values should be interpreted as logits.

    Returns
    -------
    tuple[float, float]
        Normalized (probability, confidence) in [0, 1].

    Examples
    --------
    >>> prob, conf = normalize_prediction_output([0.2, 0.8], None)
    >>> round(prob, 2), round(conf, 2)
    (0.8, 0.8)

    """
    array = np.asarray(prediction)
    if array.size > 1:
        prob, default_conf = _probability_from_vector(array, classes=classes)
        return prob, _normalize_confidence(confidence, fallback=default_conf)

    scalar = _scalar_from_array(array)
    prob = _normalize_scalar_probability(scalar, output_is_logits=output_is_logits)
    default_conf = _confidence_from_probability(prob)
    return prob, _normalize_confidence(confidence, fallback=default_conf)


def normalize_prediction_batch(
    predictions: npt.NDArray[np.floating[Any]],
    confidences: npt.NDArray[np.floating[Any]] | None = None,
    *,
    output_is_logits: bool = False,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Normalize batched predictions into canonical probability + confidence arrays.

    Parameters
    ----------
    predictions : npt.NDArray[np.floating[Any]]
        Batched prediction outputs with shape (N,), (N, 1), or (N, K).
    confidences : npt.NDArray[np.floating[Any]] | None, optional
        Optional confidence outputs with shape (N,) or (N, 1).
    output_is_logits : bool, default False
        Whether the prediction values should be interpreted as logits.

    Returns
    -------
    tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Tuple of (probabilities, confidences) arrays shaped (N,).

    Examples
    --------
    >>> preds = np.array([[0.1, 0.9], [0.7, 0.3]], dtype=np.float32)
    >>> probs, confs = normalize_prediction_batch(preds)
    >>> probs.shape, confs.shape
    ((2,), (2,))

    """
    preds = np.asarray(predictions, dtype=np.float64)

    if preds.ndim == 2 and preds.shape[1] > 1 and confidences is None:
        probs = np.where(np.isfinite(preds), preds, 0.0)
        probs = np.clip(probs, 0.0, 1.0)
        pos_idx = probs.shape[1] - 1
        prob = probs[:, pos_idx]
        conf = np.max(probs, axis=1)
        return prob.astype(np.float32), conf.astype(np.float32)

    if preds.ndim > 1:
        preds = preds.reshape(-1)

    if output_is_logits:
        prob = _sigmoid_vector(preds)
    else:
        prob = _map_scalar_vector(preds)

    prob = np.where(np.isfinite(prob), prob, 0.5)
    prob = np.clip(prob, 0.0, 1.0)

    fallback = np.maximum(prob, 1.0 - prob)
    if confidences is None:
        conf = fallback
        return prob.astype(np.float32), conf.astype(np.float32)

    conf_arr = np.asarray(confidences, dtype=np.float64)
    if conf_arr.ndim > 1:
        conf_arr = conf_arr.reshape(-1)
    if conf_arr.size != prob.size:
        conf = fallback
        return prob.astype(np.float32), conf.astype(np.float32)

    conf_arr = np.where(np.isfinite(conf_arr), conf_arr, np.nan)
    conf_arr = np.clip(conf_arr, 0.0, 1.0)
    conf = np.where(np.isfinite(conf_arr), conf_arr, fallback)
    return prob.astype(np.float32), conf.astype(np.float32)


def decision_from_probability(
    probability: float,
    *,
    neutral_band: float,
    threshold: float = 0.5,
) -> str:
    """
    Map a probability to a BUY/SELL/HOLD decision using a neutral band.

    Parameters
    ----------
    probability : float
        Prediction probability in [0, 1].
    neutral_band : float
        Neutral band half-width around the threshold (0.0 to 0.5).
    threshold : float, default 0.5
        Decision threshold for BUY/SELL.

    Returns
    -------
    str
        One of "BUY", "SELL", or "HOLD".

    Examples
    --------
    >>> decision_from_probability(0.52, neutral_band=0.05)
    'HOLD'

    """
    prob = _clip_probability(probability)
    band = max(0.0, float(neutral_band))
    if band <= 0.0:
        return "BUY" if prob > float(threshold) else "SELL"

    lower, upper = neutral_band_bounds(band, threshold=threshold)
    if prob >= upper:
        return "BUY"
    if prob <= lower:
        return "SELL"
    return "HOLD"


def neutral_band_bounds(
    neutral_band: float,
    *,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Compute lower/upper bounds for a neutral band around the threshold.

    Parameters
    ----------
    neutral_band : float
        Neutral band half-width around the threshold.
    threshold : float, default 0.5
        Center threshold for the band.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds clipped to [0, 1].

    """
    band = max(0.0, float(neutral_band))
    center = float(threshold)
    lower = max(0.0, center - band)
    upper = min(1.0, center + band)
    return lower, upper


def _probability_from_vector(
    probabilities: npt.NDArray[np.floating[Any]],
    *,
    classes: Sequence[Any] | None,
) -> tuple[float, float]:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size == 0:
        return 0.5, 0.5
    probs = np.where(np.isfinite(probs), probs, 0.0)
    probs = np.clip(probs, 0.0, 1.0)
    pos_idx = _resolve_positive_class_index(classes, probs.size)
    if pos_idx >= probs.size:
        pos_idx = probs.size - 1
    p_pos = float(probs[pos_idx])
    conf = float(np.max(probs))
    return p_pos, conf


def _resolve_positive_class_index(
    classes: Sequence[Any] | None,
    num_classes: int,
) -> int:
    if classes:
        class_list = list(classes)
        if len(class_list) == num_classes:
            for candidate in (1, "1", True, "BUY", "LONG", "UP", "POSITIVE"):
                if candidate in class_list:
                    return class_list.index(candidate)
            numeric = [
                (idx, value)
                for idx, value in enumerate(class_list)
                if isinstance(value, (int, float))
            ]
            if numeric:
                return max(numeric, key=lambda item: item[1])[0]
    return max(0, num_classes - 1)


def _normalize_scalar_probability(value: float, *, output_is_logits: bool) -> float:
    if not math.isfinite(value):
        return 0.5
    if output_is_logits:
        return float(_sigmoid_scalar(value))
    if 0.0 <= value <= 1.0:
        return float(value)
    if -1.0 <= value <= 1.0:
        return float(0.5 + 0.5 * value)
    return 0.5


def _map_scalar_vector(values: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    within_prob = (values >= 0.0) & (values <= 1.0)
    within_signed = (values >= -1.0) & (values <= 1.0)
    mapped = np.where(within_prob, values, np.where(within_signed, 0.5 + 0.5 * values, 0.5))
    return mapped.astype(np.float64)


def _confidence_from_probability(probability: float) -> float:
    return max(probability, 1.0 - probability)


def _normalize_confidence(confidence: Any | None, *, fallback: float) -> float:
    if confidence is None:
        return float(fallback)
    arr = np.asarray(confidence)
    if arr.size != 1:
        return float(fallback)
    value = _scalar_from_array(arr)
    if not math.isfinite(value):
        return float(fallback)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _scalar_from_array(array: npt.NDArray[Any]) -> float:
    return float(array.reshape(-1)[0]) if array.size else 0.5


def _sigmoid_scalar(value: float) -> float:
    clipped = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _sigmoid_vector(values: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float64]:
    clipped = np.clip(values.astype(np.float64), -60.0, 60.0)
    output = 1.0 / (1.0 + np.exp(-clipped))
    return cast(npt.NDArray[np.float64], output)


def _clip_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
