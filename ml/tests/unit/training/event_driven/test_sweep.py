from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pytest

from ml._imports import HAS_OPTUNA
from ml.training.event_driven.sweep import FileTrialLogger
from ml.training.event_driven.sweep import SweepSearchSpace
from ml.training.event_driven.sweep import SweepTrialOutcome
from ml.training.event_driven.sweep import SweepTrialParameters
from ml.training.event_driven.sweep import StreamingTrialRunner
from ml.training.event_driven.sweep import StreamingWorkerStudyRunner


class _StubRunner(StreamingTrialRunner):
    def __init__(self) -> None:
        self._invocations = 0

    def run(self, params: SweepTrialParameters, *, trial_dir: Path) -> SweepTrialOutcome:
        self._invocations += 1
        trial_dir.mkdir(parents=True, exist_ok=True)
        objective = 0.6 + 0.05 * self._invocations
        metrics: Mapping[str, float] = {"roc_auc": objective, "pr_auc": objective - 0.05}
        artifacts = {"logits": str(trial_dir / "logits.npz")}
        return SweepTrialOutcome(
            objective=objective,
            metrics=metrics,
            artifacts=artifacts,
            status="success",
        )


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna dependency required for sweep tests")
def test_study_runner_records_trials(tmp_path: Path) -> None:
    output_dir = tmp_path / "sweep"
    runner = _StubRunner()
    search_space = SweepSearchSpace(
        batch_sizes=(64,),
        hidden_sizes=(16,),
        lstm_layers=(1,),
        attention_head_sizes=(2,),
        dropouts=(0.1,),
        learning_rate_range=(1e-4, 1e-3),
        optimizers=("adam",),
        lr_schedulers=("reduce_on_plateau",),
        max_epochs=(1,),
    )
    logger = FileTrialLogger(output_dir)
    study_runner = StreamingWorkerStudyRunner(
        runner=runner,
        search_space=search_space,
        output_dir=output_dir,
        logger=logger,
        study_name="unit-test-sweep",
    )

    study = study_runner.run(max_trials=2, seed=123)

    assert study.best_value == pytest.approx(0.7)
    assert (output_dir / "trials" / "trial_000.json").exists()
    assert (output_dir / "trials" / "trial_001.json").exists()
    summary_path = output_dir / "study_summary.json"
    assert summary_path.exists()
