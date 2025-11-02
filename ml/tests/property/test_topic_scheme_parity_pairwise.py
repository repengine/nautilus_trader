from __future__ import annotations

import pytest

from typing import Final

from ml.common.message_topics import build_topic_for_stage
from ml.config.events import Stage


STAGES: Final[list[Stage]] = list(Stage)
INSTRUMENTS: Final[list[str]] = [
    "",  # empty – special case
    "EUR/USD.SIM",
    "EURUSD/SIM",
    "BTC-USD.SIM",
    "ES.M2025.SIM",
    "foo#bar",  # reserved chars normalized
    "X",
]
PREFIXES: Final[list[str]] = ["events.ml", "dev.ml"]
SCHEMES: Final[list[str]] = ["domain_op", "stage_first"]


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("instrument", INSTRUMENTS)
@pytest.mark.parametrize("prefix", PREFIXES)
def test_pairwise_topic_parity(stage: Stage, instrument: str, prefix: str) -> None:
    # Scheme parity: normalized instrument suffix equality across schemes
    t_domain = build_topic_for_stage(stage, instrument, scheme="domain_op", prefix=prefix)
    t_stage = build_topic_for_stage(stage, instrument, scheme="stage_first", prefix=prefix)

    d_parts = t_domain.split(".")
    s_parts = t_stage.split(".")
    inst_domain = ".".join(d_parts[3:]) if len(d_parts) > 3 else ""
    inst_stage = ".".join(s_parts[3:]) if len(s_parts) > 3 else ""

    if instrument == "":
        assert inst_domain == "UNKNOWN" and inst_stage == ""
    else:
        assert inst_domain == inst_stage

    # Prefix behavior: stage_first uses prefix; domain_op ignores
    assert t_stage.startswith(f"{prefix}."), "stage_first must honor prefix"
    assert t_domain.startswith("ml."), "domain_op must use 'ml.' root"
