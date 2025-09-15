from __future__ import annotations

from pathlib import Path
import tempfile

from ml.validate_security_posture import analyze_file


def test_analyze_file_flags_pickle_import() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tmp_test.py"
        p.write_text("import pickle\n", encoding="utf-8")
        violations = analyze_file(p)
        assert any(v.violation_type.startswith("pickle_") for v in violations)

