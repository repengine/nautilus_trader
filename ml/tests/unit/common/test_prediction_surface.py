from __future__ import annotations

import numpy as np
import pytest

from ml.common import decision_from_probability
from ml.common import normalize_prediction_batch
from ml.common import normalize_prediction_output


def test_normalize_prediction_output_when_probabilities_vector_returns_positive_probability() -> None:
    prob, conf = normalize_prediction_output([0.2, 0.8], None, positive_class_index=1)
    assert prob == pytest.approx(0.8)
    assert conf == pytest.approx(0.8)


def test_normalize_prediction_output_when_signed_score_maps_to_probability() -> None:
    prob, conf = normalize_prediction_output(-0.5, None)
    assert prob == pytest.approx(0.25)
    assert conf == pytest.approx(0.75)


def test_normalize_prediction_output_when_logits_returns_sigmoid_probability() -> None:
    prob, conf = normalize_prediction_output(0.0, None, output_is_logits=True)
    assert prob == pytest.approx(0.5)
    assert conf == pytest.approx(0.5)


def test_normalize_prediction_batch_when_multiclass_vector_returns_expected_arrays() -> None:
    preds = np.array([[0.1, 0.9], [0.7, 0.3]], dtype=np.float32)
    probs, confs = normalize_prediction_batch(preds, positive_class_index=1)
    np.testing.assert_allclose(probs, np.array([0.9, 0.3], dtype=np.float32))
    np.testing.assert_allclose(confs, np.array([0.9, 0.7], dtype=np.float32))


def test_normalize_prediction_output_when_vector_without_positive_index_raises() -> None:
    with pytest.raises(ValueError, match="positive_class_index"):
        normalize_prediction_output([0.2, 0.8], None)


def test_normalize_prediction_batch_when_vector_without_positive_index_raises() -> None:
    preds = np.array([[0.1, 0.9]], dtype=np.float32)
    with pytest.raises(ValueError, match="positive_class_index"):
        normalize_prediction_batch(preds)


def test_decision_from_probability_when_inside_neutral_band_returns_hold() -> None:
    assert decision_from_probability(0.52, neutral_band=0.05) == "HOLD"
    assert decision_from_probability(0.6, neutral_band=0.05) == "BUY"
    assert decision_from_probability(0.4, neutral_band=0.05) == "SELL"
