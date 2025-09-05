from __future__ import annotations

from typing import ClassVar, Literal

from nautilus_trader.common.config import NautilusConfig
from nautilus_trader.common.config import PositiveFloat


class ObservabilityConfig(NautilusConfig, kw_only=True, frozen=True):
    """
    Configuration for the Observability flusher/sink (off hot-path).

    Parameters
    ----------
    sink : Literal["file", "db"], default "file"
        Destination for observability tables. "file" for JSONL/CSV, "db" for SQL.
    base_path : str, default "./observability"
        Base path for file sink outputs.
    file_format : str, default "jsonl"
        File sink format ("jsonl" or "csv").
    db_connection_string : str | None, default None
        SQLAlchemy connection URL for DB sink.
    interval_seconds : PositiveFloat, default 60.0
        Flush interval in seconds for background scheduler.
    """

    sink: Literal["file", "db"] = "file"
    base_path: str = "./observability"
    file_format: str = "jsonl"
    db_connection_string: str | None = None
    interval_seconds: PositiveFloat = 60.0

    # Environment variable overrides
    _ENV_MAPPING: ClassVar[dict[str, str]] = {
        "sink": "ML_OBS_SINK",
        "base_path": "ML_OBS_BASE_PATH",
        "file_format": "ML_OBS_FILE_FORMAT",
        "db_connection_string": "ML_OBS_DB_URL",
        "interval_seconds": "ML_OBS_INTERVAL_SECONDS",
    }

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        """Build ObservabilityConfig from environment variables if present."""
        import os

        kwargs: dict[str, object] = {}
        for field, env_var in cls._ENV_MAPPING.items():
            if env_var in os.environ:
                val: str = os.environ[env_var]
                if field == "interval_seconds":
                    try:
                        kwargs[field] = float(val)
                    except ValueError:
                        continue
                elif field == "sink":
                    if val in {"file", "db"}:
                        kwargs[field] = val
                    else:
                        continue
                else:
                    kwargs[field] = val
        # Construct with explicit typing for msgspec/NautilusConfig
        from typing import Literal as _Lit
        from typing import cast

        sink_obj = kwargs.get("sink", "file")
        sink_val: _Lit["file", "db"] = cast(_Lit["file", "db"], sink_obj if sink_obj in {"file", "db"} else "file")
        base_obj = kwargs.get("base_path", "./observability")
        base_path_val: str = str(base_obj)
        fmt_obj = kwargs.get("file_format", "jsonl")
        file_format_val: str = str(fmt_obj)
        db_url_val: str | None = (
            str(kwargs.get("db_connection_string")) if kwargs.get("db_connection_string", None) is not None else None
        )
        inter_obj = kwargs.get("interval_seconds", 60.0)
        interval_val: float = float(inter_obj) if isinstance(inter_obj, (int, float, str)) else 60.0
        return cls(
            sink=sink_val, base_path=base_path_val, file_format=file_format_val, db_connection_string=db_url_val, interval_seconds=interval_val
        )
