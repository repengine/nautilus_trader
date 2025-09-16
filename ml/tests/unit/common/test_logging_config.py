from __future__ import annotations

import json
import logging
import os
from typing import Any

from ml.common.logging_config import bind_log_context, configure_logging


def test_structured_logging_json_output(capsys: Any) -> None:
    # Force JSON output and INFO level
    os.environ["ML_LOG_FORMAT"] = "json"
    os.environ["ML_LOG_LEVEL"] = "INFO"

    # Configure logging and bind context (preserve original handlers)
    root = logging.getLogger()
    orig_handlers = list(root.handlers)
    orig_level = root.level
    try:
        configure_logging()
        bind_log_context(run_id="test_run_123", component="ml.tests.logging")

        # Emit a log via stdlib logger and capture output
        logger = logging.getLogger(__name__)
        logger.info("hello %s", "world")

        out = capsys.readouterr().out.strip()
        # Validate JSON and expected fields
        data = json.loads(out)
        assert data.get("event") == "hello world"
    finally:
        # Restore original logging handlers to avoid cross-test interference
        root.handlers = orig_handlers
        root.setLevel(orig_level)
