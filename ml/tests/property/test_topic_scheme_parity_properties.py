from __future__ import annotations

import re
import pytest

try:  # optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
    from hypothesis.strategies import composite
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from typing import Final

from ml.common.message_topics import build_topic_for_stage, map_stage_to_topic_segments
from ml.config.events import Stage


# Utility: safe prefix strings for stage_first scheme
@composite
def prefixes(draw: st.DrawFn) -> str:  # type: ignore[name-defined]
    # Allow lowercase letters, numbers, dots, and dashes
    part = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=10)
    count = draw(st.integers(min_value=1, max_value=3))
    parts = [draw(part) for _ in range(count)]
    return ".".join(parts)


@given(
    stage=st.sampled_from(list(Stage)),
    instrument=st.text(min_size=0, max_size=60),
)
def test_topic_scheme_parity_instrument_normalization(stage: Stage, instrument: str) -> None:
    t_domain = build_topic_for_stage(stage, instrument, scheme="domain_op", prefix="events.ml")
    t_stage = build_topic_for_stage(stage, instrument, scheme="stage_first", prefix="events.ml")

    # Extract normalized instrument suffixes
    d_parts = t_domain.split(".")
    s_parts = t_stage.split(".")
    inst_domain = ".".join(d_parts[3:]) if len(d_parts) > 3 else ""
    inst_stage = ".".join(s_parts[3:]) if len(s_parts) > 3 else ""

    # Empty instrument normalizes to 'UNKNOWN' for domain-op scheme and to no suffix for stage-first
    if instrument == "":
        assert inst_domain == "UNKNOWN" and inst_stage == ""
    else:
        assert inst_domain == inst_stage


@given(
    stage=st.sampled_from(list(Stage)),
    instrument=st.text(min_size=1, max_size=60),
    prefix=prefixes(),
)
def test_stage_first_applies_prefix_and_domain_op_ignores(
    stage: Stage,
    instrument: str,
    prefix: str,
) -> None:
    t_domain = build_topic_for_stage(stage, instrument, scheme="domain_op", prefix=prefix)
    t_stage = build_topic_for_stage(stage, instrument, scheme="stage_first", prefix=prefix)

    # domain_op format: ml.{domain}.{operation}.{instrument}
    assert t_domain.startswith("ml."), "domain_op topics must use 'ml.' root"
    # stage_first format: {prefix}.{STAGE}[.{instrument}]
    assert t_stage.startswith(f"{prefix}."), "stage_first must include given prefix"
    assert Stage(stage.value).value in t_stage, "stage_first must include STAGE value"

    # Ensure prefix does not leak into domain_op scheme
    assert not t_domain.startswith(f"{prefix}."), "domain_op must ignore prefix entirely"


@given(stage=st.sampled_from(list(Stage)), instrument=st.text(min_size=0, max_size=60))
def test_domain_op_segments_match_mapping(stage: Stage, instrument: str) -> None:
    domain, op = map_stage_to_topic_segments(stage)
    topic = build_topic_for_stage(stage, instrument, scheme="domain_op")
    parts = topic.split(".")
    # Expect at least 4 parts: ml, domain, operation, instrument...
    assert len(parts) >= 4 and parts[0] == "ml"
    assert parts[1] == domain and parts[2] == op

    # Validate domain/op formats
    assert re.fullmatch(r"[a-z]+", parts[1]) is not None
    assert re.fullmatch(r"[a-z_]+", parts[2]) is not None


@given(instruments=st.lists(st.text(min_size=0, max_size=40), min_size=1, max_size=10))
def test_instrument_normalization_parity_across_all_stages(instruments: list[str]) -> None:
    # For each stage and instrument, normalized suffix parity must hold between schemes
    for stage in list(Stage):
        for instrument in instruments:
            t_domain = build_topic_for_stage(
                stage,
                instrument,
                scheme="domain_op",
                prefix="events.ml",
            )
            t_stage = build_topic_for_stage(
                stage,
                instrument,
                scheme="stage_first",
                prefix="events.ml",
            )

            d_parts = t_domain.split(".")
            s_parts = t_stage.split(".")
            inst_domain = ".".join(d_parts[3:]) if len(d_parts) > 3 else ""
            inst_stage = ".".join(s_parts[3:]) if len(s_parts) > 3 else ""

            if instrument == "":
                assert inst_domain == "UNKNOWN" and inst_stage == ""
            else:
                assert inst_domain == inst_stage
