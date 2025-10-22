"""Typed configuration for the Dashboard control-plane service (cold path)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final


def _parse_iso8601(value: str) -> dt.datetime | None:
    """Parse ISO-8601 timestamps including ``Z`` suffix, returning UTC datetimes."""
    text = value.strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            return dt.datetime.fromisoformat(text[:-1]).replace(tzinfo=dt.UTC)
        dt_val = dt.datetime.fromisoformat(text)
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=dt.UTC)
        return dt_val.astimezone(dt.UTC)
    except Exception:
        return None


@dataclass(slots=True, frozen=True)
class DashboardToken:
    """Bearer token metadata for dashboard authentication."""

    value: str
    expires_at: dt.datetime | None = None

    def is_valid(self, *, now: dt.datetime | None = None) -> bool:
        """Return ``True`` when the token remains valid at ``now`` (UTC)."""
        if now is None:
            now = dt.datetime.now(dt.UTC)
        if self.expires_at is None:
            return True
        return now < self.expires_at

    def expires_at_iso(self) -> str | None:
        """Return the expiry timestamp in ISO-8601 format when present."""
        return self.expires_at.isoformat() if self.expires_at else None


_DEFAULT_TIMEOUT_SECONDS: Final[float] = 2.5


@dataclass(slots=True, frozen=True)
class DashboardConfig:
    """Configuration for the Dashboard service."""

    compose_enabled: bool = False
    compose_file: Path | None = None
    request_timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    actor_port: int = 8000
    strategy_port: int = 8001
    pipeline_port: int = 8081
    grafana_port: int = 3000
    prometheus_port: int = 9090
    redis_port: int = 6380
    events_cache_ttl_seconds: float = 5.0
    events_cache_max_entries: int = 200
    events_poll_interval_seconds: float = 0.0

    grafana_url: str = "http://localhost:3000"
    grafana_api_token: str | None = None
    grafana_username: str | None = None
    grafana_password: str | None = None
    grafana_folder_uid: str | None = None
    grafana_dashboard_uid: str = "ml-control-plane"
    grafana_dashboard_title: str = "Nautilus ML Control Plane"
    grafana_refresh_interval: str = "30s"
    grafana_datasource_uid: str | None = None

    grafana_embed_enabled: bool = False
    grafana_embed_theme: str = "light"
    grafana_embed_panels: tuple[int, ...] = ()
    grafana_embed_org_id: int = 1
    grafana_embed_base_url: str | None = None
    grafana_provision_on_start: bool = False

    prometheus_url: str = "http://localhost:9090"
    prometheus_query_timeout_seconds: float = 2.5

    streaming_state_path: Path = Path("./ml_out/streaming_training_state.json")
    db_connection: str | None = None
    store_health_cache_ttl_seconds: float = 30.0
    store_health_cache_max_entries: int = 8
    store_health_top_datasets: int = 5
    store_health_enabled: bool = True
    auth_tokens: tuple[DashboardToken, ...] = ()

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> DashboardConfig:
        """Build a config from environment variables."""
        e = env or {}

        def _truthy(name: str, default: bool) -> bool:
            raw = e.get(name)
            if raw is None:
                import os

                raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

        def _int(name: str, default: int) -> int:
            raw = e.get(name)
            if raw is None:
                import os

                raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except Exception:
                return default

        def _float(name: str, default: float) -> float:
            raw = e.get(name)
            if raw is None:
                import os

                raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except Exception:
                return default

        def _string(name: str, default: str | None) -> str | None:
            raw = e.get(name)
            if raw is None:
                import os

                raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            return raw

        def _int_tuple(name: str) -> tuple[int, ...]:
            raw = e.get(name)
            if raw is None:
                import os

                raw = os.getenv(name)
            if raw is None or raw.strip() == "":
                return ()
            values: list[int] = []
            for part in raw.split(","):
                cleaned = part.strip()
                if not cleaned:
                    continue
                try:
                    values.append(int(cleaned))
                except ValueError:
                    continue
            return tuple(values)

        compose_path = e.get("ML_DASHBOARD_COMPOSE_FILE")
        if compose_path is None:
            import os

            compose_path = os.getenv("ML_DASHBOARD_COMPOSE_FILE")

        grafana_url = _string("GRAFANA_URL", None)
        if not grafana_url:
            grafana_url = f"http://localhost:{_int('GRAFANA_HOST_PORT', 3000)}"

        prometheus_url = _string("PROMETHEUS_URL", None)
        if not prometheus_url:
            prometheus_url = f"http://localhost:{_int('PROMETHEUS_HOST_PORT', 9090)}"

        streaming_state_raw = _string("ML_DASHBOARD_STREAMING_STATE_PATH", None)
        if streaming_state_raw and streaming_state_raw.strip():
            streaming_state_path = Path(streaming_state_raw).expanduser()
        else:
            streaming_state_path = Path("./ml_out/streaming_training_state.json")

        def _parse_tokens() -> tuple[DashboardToken, ...]:
            import json

            tokens_raw = _string("ML_DASHBOARD_TOKENS", None)
            tokens: list[DashboardToken] = []

            if tokens_raw:
                parsed: list[object] | None
                try:
                    parsed = json.loads(tokens_raw)
                    if not isinstance(parsed, list):
                        parsed = None
                except Exception:
                    parsed = None
                if parsed is not None:
                    for entry in parsed:
                        if isinstance(entry, str):
                            if entry.strip():
                                tokens.append(DashboardToken(value=entry.strip()))
                            continue
                        if isinstance(entry, dict):
                            value = str(entry.get("value", "")).strip()
                            if not value:
                                continue
                            expires_raw = entry.get("expires")
                            expires = (
                                _parse_iso8601(str(expires_raw))
                                if isinstance(expires_raw, str)
                                else None
                            )
                            tokens.append(DashboardToken(value=value, expires_at=expires))
                    return tuple(tokens)
                # Fallback: comma-separated list
                for part in tokens_raw.split(","):
                    value = part.strip()
                    if value:
                        tokens.append(DashboardToken(value=value))

            single_token = _string("ML_DASHBOARD_TOKEN", None)
            if single_token:
                expires = _parse_iso8601(_string("ML_DASHBOARD_TOKEN_EXPIRES", "") or "")
                tokens.append(DashboardToken(value=single_token, expires_at=expires))

            return tuple(tokens)

        db_conn = _string("ML_DB_CONNECTION", None)

        if not db_conn:
            db_conn_env = _string("NAUTILUS_DB_CONNECTION", None)
            db_conn = db_conn_env

        return DashboardConfig(
            compose_enabled=_truthy("ML_DASHBOARD_USE_COMPOSE", False),
            compose_file=Path(compose_path) if compose_path else None,
            request_timeout_seconds=_float("ML_DASHBOARD_TIMEOUT", _DEFAULT_TIMEOUT_SECONDS),
            actor_port=_int("ML_ACTOR_HOST_PORT", 8000),
            strategy_port=_int("ML_STRATEGY_HOST_PORT", 8001),
            pipeline_port=_int("ML_PIPELINE_HOST_PORT", 8081),
            grafana_port=_int("GRAFANA_HOST_PORT", 3000),
            prometheus_port=_int("PROMETHEUS_HOST_PORT", 9090),
            redis_port=_int("REDIS_HOST_PORT", 6380),
            events_cache_ttl_seconds=_float("ML_DASHBOARD_EVENTS_CACHE_TTL", 5.0),
            events_cache_max_entries=_int("ML_DASHBOARD_EVENTS_CACHE_MAX", 200),
            events_poll_interval_seconds=_float("ML_DASHBOARD_EVENTS_POLL_INTERVAL", 0.0),
            grafana_url=grafana_url,
            grafana_api_token=_string("GRAFANA_API_TOKEN", None),
            grafana_username=_string("GF_ADMIN_USER", None) or _string("GRAFANA_ADMIN_USER", None),
            grafana_password=_string("GF_SECURITY_ADMIN_PASSWORD", None)
            or _string("GRAFANA_ADMIN_PASSWORD", None),
            grafana_folder_uid=_string("GRAFANA_FOLDER_UID", None),
            grafana_dashboard_uid=_string("ML_DASHBOARD_GRAFANA_UID", "ml-control-plane") or "ml-control-plane",
            grafana_dashboard_title=_string("ML_DASHBOARD_GRAFANA_TITLE", "Nautilus ML Control Plane")
            or "Nautilus ML Control Plane",
            grafana_refresh_interval=_string("ML_DASHBOARD_GRAFANA_REFRESH", "30s") or "30s",
            grafana_datasource_uid=_string("ML_DASHBOARD_GRAFANA_DATASOURCE_UID", None),
            grafana_embed_enabled=_truthy("ML_DASHBOARD_GRAFANA_EMBED", False),
            grafana_embed_theme=_string("ML_DASHBOARD_GRAFANA_THEME", "light") or "light",
            grafana_embed_panels=_int_tuple("ML_DASHBOARD_GRAFANA_PANELS"),
            grafana_embed_org_id=_int("ML_DASHBOARD_GRAFANA_ORG_ID", 1),
            grafana_embed_base_url=_string("ML_DASHBOARD_GRAFANA_EMBED_URL", None),
            grafana_provision_on_start=_truthy("ML_DASHBOARD_GRAFANA_PROVISION_ON_START", False),
            prometheus_url=prometheus_url,
            prometheus_query_timeout_seconds=_float("ML_DASHBOARD_PROM_TIMEOUT", 2.5),
            streaming_state_path=streaming_state_path,
            db_connection=db_conn,
            store_health_cache_ttl_seconds=_float("ML_DASHBOARD_STORE_CACHE_TTL", 30.0),
            store_health_cache_max_entries=_int("ML_DASHBOARD_STORE_CACHE_MAX", 8),
            store_health_top_datasets=_int("ML_DASHBOARD_STORE_TOP_DATASETS", 5),
            store_health_enabled=_truthy("ML_DASHBOARD_STORE_SUMMARY", True),
            auth_tokens=_parse_tokens(),
        )

    def _dashboard_slug(self) -> str:
        words = [chunk for chunk in self.grafana_dashboard_title.lower().replace("/", " ").replace("_", " ").split() if chunk]
        return "-".join(words) or "nautilus-ml"

    def grafana_dashboard_url(self) -> str:
        base = self.grafana_url.rstrip("/")
        return f"{base}/d/{self.grafana_dashboard_uid}/{self._dashboard_slug()}"

    def grafana_embed_urls(self) -> tuple[str, ...]:
        if not self.grafana_embed_enabled:
            return ()
        if not self.grafana_dashboard_uid or not self.grafana_embed_panels:
            return ()
        base = (self.grafana_embed_base_url or self.grafana_url).rstrip("/")
        slug = self._dashboard_slug()
        return tuple(
            f"{base}/d-solo/{self.grafana_dashboard_uid}/{slug}?orgId={self.grafana_embed_org_id}&panelId={panel}&theme={self.grafana_embed_theme}"
            for panel in self.grafana_embed_panels
        )


__all__ = ["DashboardConfig", "DashboardToken"]
