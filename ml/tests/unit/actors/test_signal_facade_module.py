from __future__ import annotations

from ml.actors.signal_facade import MLSignalActorFacade
from ml.actors.signal_facade_impl import MLSignalActorFacade as ImplFacade


def test_signal_facade_exports_impl() -> None:
    assert MLSignalActorFacade is ImplFacade
