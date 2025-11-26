"""Azure scheduled-event helpers for streaming checkpoint orchestration."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

from ml.config.streaming_pipeline import AzureScheduledEventsConfig


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ScheduledEventNotice:
    """Materialised Azure scheduled-event payload."""

    event_id: str
    event_type: str
    event_status: str
    resources: tuple[str, ...]
    not_before: str | None


ScheduledEventCallback = Callable[[ScheduledEventNotice], None]


class AzureScheduledEventsWatcher:
    """Poll the Azure metadata service for scheduled eviction notices."""

    def __init__(
        self,
        config: AzureScheduledEventsConfig,
        *,
        callback: ScheduledEventCallback,
        fetcher: Callable[[], Sequence[ScheduledEventNotice]] | None = None,
    ) -> None:
        self._config = config
        self._callback = callback
        self._fetcher = fetcher or self._fetch_events
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_event_ids: set[str] = set()
        self._trigger_types = {value.strip().lower() for value in config.event_types if value.strip()}
        self._trigger_statuses = {
            value.strip().lower() for value in config.status_filter if value.strip()
        }
        self._resource_filter = {value.strip() for value in config.resource_filter if value.strip()}
        self._request_url = config.build_request_url()

    def start(self) -> None:
        """Start polling for scheduled events (noop when disabled)."""
        if not self._config.enabled:
            return
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="azure-scheduled-events",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 1.0) -> None:
        """Stop the background polling thread."""
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=timeout)
        self._thread = None
        self._stop_event.clear()

    def poll_once(self) -> None:
        """Execute a single poll cycle immediately."""
        if not self._config.enabled:
            return
        try:
            notices = tuple(self._fetcher())
        except Exception:
            logger.debug("azure_scheduled_events_fetch_failed", exc_info=True)
            return
        self._process_events(notices)

    def _run(self) -> None:
        interval = float(self._config.poll_interval_seconds)
        while not self._stop_event.is_set():
            self.poll_once()
            if self._stop_event.wait(interval):
                break

    def _process_events(self, notices: Sequence[ScheduledEventNotice]) -> None:
        for notice in notices:
            if not notice.event_id:
                continue
            if notice.event_id in self._seen_event_ids:
                continue
            if notice.event_type.lower() not in self._trigger_types:
                continue
            if notice.event_status.lower() not in self._trigger_statuses:
                continue
            if not self._resource_matches(notice.resources):
                continue
            self._seen_event_ids.add(notice.event_id)
            try:
                self._callback(notice)
            except Exception:
                logger.error(
                    "azure_scheduled_events_callback_failed",
                    extra={"event_id": notice.event_id},
                    exc_info=True,
                )

    def _resource_matches(self, resources: Iterable[str]) -> bool:
        if not self._resource_filter:
            return True
        for resource in resources:
            candidate = resource.strip()
            if not candidate:
                continue
            if candidate == "*" or candidate in self._resource_filter:
                return True
        return False

    def _fetch_events(self) -> Sequence[ScheduledEventNotice]:
        request = Request(
            self._request_url,
            headers={
                "Metadata": "true",
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(
                request,
                timeout=float(self._config.request_timeout_seconds),
            ) as response:
                raw = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Failed to query Azure scheduled events: {exc}") from exc
        except OSError as exc:  # pragma: no cover - defensive guard for rare transport errors
            raise RuntimeError(f"Failed to query Azure scheduled events: {exc}") from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Invalid JSON payload from Azure scheduled events") from exc

        events = payload.get("Events")
        if not isinstance(events, Sequence):
            return ()

        notices: list[ScheduledEventNotice] = []
        for entry in events:
            if not isinstance(entry, dict):
                continue
            event_id = str(entry.get("EventId") or entry.get("EventID") or "").strip()
            event_type = str(entry.get("EventType") or "").strip()
            event_status = str(entry.get("EventStatus") or "").strip()
            resources = entry.get("Resources", ())
            if not event_id or not event_type or not event_status:
                continue
            if isinstance(resources, Sequence) and not isinstance(resources, (str, bytes)):
                resource_tuple = tuple(str(item).strip() for item in resources if str(item).strip())
            else:
                resource_tuple = ()
            not_before_raw = entry.get("NotBefore")
            not_before_value = str(not_before_raw).strip() if not_before_raw else None
            notices.append(
                ScheduledEventNotice(
                    event_id=event_id,
                    event_type=event_type,
                    event_status=event_status,
                    resources=resource_tuple,
                    not_before=not_before_value or None,
                ),
            )
        return tuple(notices)


__all__ = ["AzureScheduledEventsWatcher", "ScheduledEventNotice"]
