from __future__ import annotations

from ml.actors.signal import ModelSwapper


def test_model_swapper_execute_swap() -> None:
    swapper = ModelSwapper()
    # Set initial model
    swapper.set_current_model(model={"v": 1}, metadata={"version": "1"})
    # Prepare new
    swapper.prepare_swap(model={"v": 2}, metadata={"version": "2"})
    assert swapper.swap_pending is True
    assert swapper.load_error is None
    # Execute swap
    changed = swapper.execute_swap()
    assert changed is True
    assert swapper.swap_pending is False
    assert swapper.current_model == {"v": 2}
    assert swapper.current_metadata == {"version": "2"}
