from __future__ import annotations

import pytest

try:  # optional dependency
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:  # pragma: no cover - hypothesis optional
    pytest.skip("hypothesis not available", allow_module_level=True)

from typing import Final

from ml.common.message_topics import build_topic_for_stage
from ml.config.events import Stage


@given(
    stage=st.sampled_from(list(Stage)),
    instrument=st.text(min_size=0, max_size=40),
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
