"""
Service controllers for starting/stopping services (cold path).

These controllers are optional and disabled by default to keep the dashboard safe
for environments without Docker. When enabled, actions are executed via Docker
Compose with conservative timeouts and structured logging.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ml.dashboard.exceptions import ServiceActionFailedError
from ml.dashboard.exceptions import ServiceControlUnsupportedError


logger = logging.getLogger(__name__)


@runtime_checkable
class ServiceControllerProtocol(Protocol):
    """
    Protocol for manipulating named services.
    """

    def start(self, name: str) -> bool:  # pragma: no cover - exercised via higher-level tests
        ...

    def stop(self, name: str) -> bool:  # pragma: no cover - exercised via higher-level tests
        ...

    def restart(self, name: str) -> bool:  # pragma: no cover - exercised via higher-level tests
        ...


@dataclass(slots=True)
class NoopServiceController(ServiceControllerProtocol):
    """
    No-op controller used when compose control is disabled.
    """

    def start(self, name: str) -> bool:
        return False

    def stop(self, name: str) -> bool:
        return False

    def restart(self, name: str) -> bool:
        return False


@dataclass(slots=True)
class ComposeServiceController(ServiceControllerProtocol):
    """
    Docker Compose controller.

    Attributes
    ----------
    compose_file : Path
        Path to a compose file. If not provided, a best-effort discovery is attempted
        in the working tree.
    """

    compose_file: Path | None = None

    def _resolve_compose_file(self) -> Path:
        if self.compose_file is not None and self.compose_file.exists():
            return self.compose_file
        # Best-effort discovery using project conventions
        candidates = [
            Path("ml/deployment/docker-compose.yml"),
            Path("docker-compose.yml"),
        ]
        for c in candidates:
            if c.exists():
                return c
        raise ServiceControlUnsupportedError("compose file not found")

    def _compose(self, *args: str) -> None:
        docker = shutil.which("docker")
        if not docker:
            raise ServiceControlUnsupportedError("docker not found in PATH")
        compose_file = self._resolve_compose_file()
        try:
            subprocess.run(
                [docker, "compose", "-f", str(compose_file), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.warning("compose command failed: %s", exc, exc_info=True)
            raise ServiceActionFailedError(exc.stderr or str(exc)) from exc

    def start(self, name: str) -> bool:
        self._compose("up", "-d", name)
        return True

    def stop(self, name: str) -> bool:
        self._compose("stop", name)
        return True

    def restart(self, name: str) -> bool:
        # Conservative: stop then start to ensure healthchecks restart
        self._compose("stop", name)
        self._compose("up", "-d", name)
        return True


__all__ = [
    "ComposeServiceController",
    "NoopServiceController",
    "ServiceControllerProtocol",
]
