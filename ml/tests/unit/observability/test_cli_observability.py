from __future__ import annotations

from pathlib import Path

from ml.cli.observability import main


def test_cli_flush_jsonl(tmp_path: Path) -> None:
    code = main(["flush-jsonl", "--base-path", str(tmp_path), "--format", "jsonl", "--seed-sample"])
    assert code == 0
    # Expect at least one jsonl file
    assert any(p.suffix == ".jsonl" for p in tmp_path.iterdir())


def test_cli_flush_db(tmp_path: Path) -> None:
    db = tmp_path / "obs.db"
    code = main(["flush-db", "--db-url", f"sqlite:///{db}", "--seed-sample"])
    assert code == 0
    assert db.exists()

