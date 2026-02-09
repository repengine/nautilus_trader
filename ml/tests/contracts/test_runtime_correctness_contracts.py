from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.contracts,
    pytest.mark.runtime_correctness,
]


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_CORRECTNESS_TARGET = "pytest-ml-runtime-correctness"
REGISTRY_HARDENING_TARGET = "pytest-ml-registry-hardening"
STRICT_POLICY_TARGET = "pytest-ml-strict-policy"
RUNTIME_CORRECTNESS_TEST_FILES: tuple[str, ...] = (
    "ml/tests/unit/actors/test_inference_deadline_guard.py",
    "ml/tests/unit/actors/test_signal_facade_impl_unit.py",
    "ml/tests/unit/actors/test_base_actor_cold_path.py",
    "ml/tests/unit/actors/test_multi_signal_actor.py",
    "ml/tests/integration/replay/test_actor_reset_on_rewind.py",
)


def _extract_make_target_block(*, makefile_text: str, target: str) -> str:
    target_header = f"{target}:"
    start_index = makefile_text.index(target_header)
    remaining_text = makefile_text[start_index:]
    next_phony_index = remaining_text.find("\n.PHONY:")
    if next_phony_index == -1:
        return remaining_text
    return remaining_text[:next_phony_index]


def test_runtime_correctness_entrypoint_targets_expected_test_modules() -> None:
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert RUNTIME_CORRECTNESS_TARGET in makefile_text
    assert '-m "runtime_correctness"' in makefile_text
    for relative_path in RUNTIME_CORRECTNESS_TEST_FILES:
        assert relative_path in makefile_text
    assert "ml/tests/contracts/test_runtime_correctness_contracts.py" in makefile_text


def test_runtime_correctness_marker_registered_in_pytest_configs() -> None:
    root_pytest_ini = (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8")
    ml_pytest_ini = (REPO_ROOT / "ml/pytest.ini").read_text(encoding="utf-8")

    assert "runtime_correctness:" in root_pytest_ini
    assert "runtime_correctness:" in ml_pytest_ini


def test_runtime_correctness_test_modules_are_marker_scoped() -> None:
    for relative_path in RUNTIME_CORRECTNESS_TEST_FILES:
        module_text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "runtime_correctness" in module_text


def test_runtime_correctness_ingress_guard_runs_before_prepare_and_halt_gates() -> None:
    base_actor_text = (REPO_ROOT / "ml/actors/base.py").read_text(encoding="utf-8")
    on_bar_text = base_actor_text.split("def on_bar(self, bar: Bar) -> None:", maxsplit=1)[1]

    guard_index = on_bar_text.index("_apply_ingress_causality_monotonic_guard(bar)")
    prepare_index = on_bar_text.index("self._prepare_bar_runtime_state(bar)")
    halt_index = on_bar_text.index("if self._ml_inference_halted:")

    assert guard_index < prepare_index
    assert guard_index < halt_index


def test_runtime_correctness_configured_failure_action_is_guarded_when_halted() -> None:
    base_actor_text = (REPO_ROOT / "ml/actors/base.py").read_text(encoding="utf-8")
    apply_failure_text = base_actor_text.split(
        "def _apply_configured_ml_failure_action(",
        maxsplit=1,
    )[1]

    halted_guard_index = apply_failure_text.index("if self._ml_inference_halted:")
    policy_index = apply_failure_text.index("policy = self._config.remediation_policy")

    assert halted_guard_index < policy_index


def test_runtime_correctness_covers_deadline_failure_and_transition_hook_outcomes() -> None:
    guard_tests_text = (
        REPO_ROOT / "ml/tests/unit/actors/test_inference_deadline_guard.py"
    ).read_text(encoding="utf-8")

    assert "test_deadline_guard_drop_skips_persist_and_publish" in guard_tests_text
    assert "test_prediction_failure_applies_configured_failure_action" in guard_tests_text
    assert "test_deadline_guard_halt_risk_transition_uses_replay_safe_live_flag" in guard_tests_text
    assert (
        "test_deadline_guard_halt_marks_missing_risk_transition_when_hook_write_fails"
        in guard_tests_text
    )


def test_runtime_correctness_covers_multi_signal_deadline_and_failure_paths() -> None:
    multi_signal_tests_text = (
        REPO_ROOT / "ml/tests/unit/actors/test_multi_signal_actor.py"
    ).read_text(encoding="utf-8")

    assert "test_multi_signal_deadline_guard_drop_skips_dispatch" in multi_signal_tests_text
    assert (
        "test_multi_signal_batch_inference_failure_applies_configured_failure_action"
        in multi_signal_tests_text
    )


def test_strict_policy_entrypoint_composes_required_subtargets() -> None:
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    strict_policy_block = _extract_make_target_block(
        makefile_text=makefile_text,
        target=STRICT_POLICY_TARGET,
    )

    assert STRICT_POLICY_TARGET in makefile_text
    assert f"$(MAKE) {RUNTIME_CORRECTNESS_TARGET}" in strict_policy_block
    assert f"$(MAKE) {REGISTRY_HARDENING_TARGET}" in strict_policy_block


def test_strict_policy_entrypoint_enforces_fail_fast_order() -> None:
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    strict_policy_block = _extract_make_target_block(
        makefile_text=makefile_text,
        target=STRICT_POLICY_TARGET,
    )
    runtime_command = f"$(MAKE) {RUNTIME_CORRECTNESS_TARGET} || exit $$?"
    registry_command = f"$(MAKE) {REGISTRY_HARDENING_TARGET} || exit $$?"

    assert runtime_command in strict_policy_block
    assert registry_command in strict_policy_block
    assert strict_policy_block.index(runtime_command) < strict_policy_block.index(registry_command)
