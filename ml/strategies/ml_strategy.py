"""
ML Trading Strategy implementation with multi-model support.

This module provides a production-ready ML strategy that can:
- Handle signals from multiple ML models
- Filter signals by model_id
- Aggregate signals for consensus trading
- Track model performance over time

"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ml.actors.base import MLSignal
from ml.common.metrics_bootstrap import get_counter
from ml.config.base import ShortEntryPolicy
from ml.strategies.base import BaseMLStrategy
from ml.strategies.common.model_exit_policy import PositionSideProtocol
from ml.strategies.common.model_exit_policy import evaluate_model_exit
from ml.strategies.common.model_exit_policy import resolve_time_in_trade_ns
from nautilus_trader.model.enums import OrderSide


if TYPE_CHECKING:  # typing-only imports to avoid runtime coupling
    from nautilus_trader.model.identifiers import ClientOrderId
    from nautilus_trader.model.position import Position

    from nautilus_trader.model.events import OrderFilled


model_exit_total = get_counter(
    "ml_model_exit_total",
    "Total model-driven exits",
    labels=["action", "reason"],
)

short_entry_blocked_total = get_counter(
    "ml_strategy_short_entry_blocked_total",
    "Total short-entry signals blocked by policy",
    labels=["strategy_id", "policy"],
)


class MLTradingStrategy(BaseMLStrategy):
    """
    Production ML trading strategy with multi-model support.

    This strategy can:
    - Filter signals by specific model IDs
    - Aggregate signals from multiple models
    - Track performance per model
    - Execute trades based on consensus or individual signals

    Configuration options (via MLStrategyConfig):
    - target_model_ids: List of model IDs to listen to (None = all)
    - aggregation_mode: "voting", "weighted_average", or None
    - required_models: Number of models needed for aggregation
    - time_window_ms: Time window for signal aggregation
    - conflict_resolution: How to handle conflicting signals
    - model_weights: Dict of model_id -> weight for weighted averaging
    - track_performance: Whether to track per-model performance

    """

    def _resolve_exit_policy_config(
        self,
    ) -> tuple[float, float, int | None]:
        """
        Resolve exit policy thresholds from config.

        Returns
        -------
        tuple[float, float, int | None]
            Stop loss pct, take profit pct, max holding ms.

        """
        exit_policy = getattr(self._config, "exit_policy_config", None)
        if exit_policy is not None:
            return (
                float(exit_policy.stop_loss_pct),
                float(exit_policy.take_profit_pct),
                exit_policy.max_holding_ms,
            )
        return (
            float(getattr(self._config, "stop_loss_pct", 0.0)),
            float(getattr(self._config, "take_profit_pct", 0.0)),
            None,
        )

    def _timestamp_ns(self) -> int:
        """
        Return a nanosecond timestamp with clock fallback.
        """
        try:
            if hasattr(self, "clock"):
                return int(self.clock.timestamp_ns())
        except Exception as exc:
            self.log.debug(
                "ml_strategy.clock_timestamp_failed",
                exc_info=True,
                error=str(exc),
            )
        return time.time_ns()

    def _update_returns_from_signal(self, signal: MLSignal) -> None:
        """
        Update returns buffers using the latest signal timestamp.
        """
        updater = getattr(self, "_returns_updater", None)
        if updater is None:
            return
        should_update = getattr(updater, "should_update_from_signal", None)
        if callable(should_update) and not should_update():
            return
        cache = self.cache if hasattr(self, "cache") else None
        reference_ts = signal.ts_event or self._timestamp_ns()
        try:
            updater.update_from_signal(
                signal,
                cache=cache,
                reference_ts=reference_ts,
            )
        except Exception as exc:
            self.log.debug(
                "ml_strategy.returns_update_failed",
                exc_info=True,
                error=str(exc),
            )

    def _position_entry_price(self, position: object) -> float | None:
        """
        Resolve a position entry price for exit calculations.
        """
        for attr in ("avg_px_open", "avg_px", "avg_price", "entry_price"):
            value = getattr(position, attr, None)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if price > 0.0:
                return price
        return None

    def _time_in_trade_ns(self, position: object, now_ns: int) -> int | None:
        """
        Compute time-in-trade for the position if available.
        """
        return resolve_time_in_trade_ns(cast(PositionSideProtocol, position), now_ns)

    def _build_exit_metadata(
        self,
        *,
        reason: str,
        trigger_price: float | None,
        time_in_trade_ns: int | None,
    ) -> dict[str, object]:
        """
        Build exit metadata payload for persistence and intents.
        """
        return {
            "reason": reason,
            "trigger_price": trigger_price,
            "time_in_trade_ns": time_in_trade_ns,
        }

    def _exit_side_for_position(self, position: object) -> OrderSide | None:
        """
        Determine exit side from a position.
        """
        side_name = getattr(getattr(position, "side", object()), "name", "")
        if side_name == "LONG":
            return OrderSide.SELL
        if side_name == "SHORT":
            return OrderSide.BUY
        return None

    def _evaluate_exit_policy(
        self,
        position: object,
        *,
        instrument_id: object,
    ) -> dict[str, object] | None:
        """
        Evaluate exit policy for the current position.
        """
        stop_loss_pct, take_profit_pct, max_holding_ms = self._resolve_exit_policy_config()
        if stop_loss_pct <= 0.0 and take_profit_pct <= 0.0 and not max_holding_ms:
            return None

        try:
            current_price = self._resolve_market_price(instrument_id)
        except AttributeError:
            current_price = None
        entry_price = self._position_entry_price(position)
        now_ns = self._timestamp_ns()
        time_in_trade_ns = self._time_in_trade_ns(position, now_ns)
        side_name = getattr(getattr(position, "side", object()), "name", "")

        if entry_price is not None and current_price is not None:
            if side_name == "LONG":
                if stop_loss_pct > 0.0 and current_price <= entry_price * (1.0 - stop_loss_pct):
                    return self._build_exit_metadata(
                        reason="stop_loss",
                        trigger_price=current_price,
                        time_in_trade_ns=time_in_trade_ns,
                    )
                if take_profit_pct > 0.0 and current_price >= entry_price * (1.0 + take_profit_pct):
                    return self._build_exit_metadata(
                        reason="take_profit",
                        trigger_price=current_price,
                        time_in_trade_ns=time_in_trade_ns,
                    )
            elif side_name == "SHORT":
                if stop_loss_pct > 0.0 and current_price >= entry_price * (1.0 + stop_loss_pct):
                    return self._build_exit_metadata(
                        reason="stop_loss",
                        trigger_price=current_price,
                        time_in_trade_ns=time_in_trade_ns,
                    )
                if take_profit_pct > 0.0 and current_price <= entry_price * (1.0 - take_profit_pct):
                    return self._build_exit_metadata(
                        reason="take_profit",
                        trigger_price=current_price,
                        time_in_trade_ns=time_in_trade_ns,
                    )

        if max_holding_ms and time_in_trade_ns is not None:
            max_holding_ns = int(max_holding_ms) * 1_000_000
            if time_in_trade_ns >= max_holding_ns:
                return self._build_exit_metadata(
                    reason="timeout",
                    trigger_price=current_price,
                    time_in_trade_ns=time_in_trade_ns,
                )

        return None

    def _process_ml_signal(self, signal: MLSignal) -> None:
        """
        Process ML signal and execute trading logic.

        This implementation:
        1. Checks current position
        2. Determines trade direction from signal
        3. Manages position changes
        4. Tracks model performance if enabled
        5. Persists decisions to StrategyStore

        Parameters
        ----------
        signal : MLSignal
            The ML signal to process (may be aggregated).

        """
        current_position = self._get_current_position()

        # Determine decision and target side based on canonical prediction surface.
        # Using 0.5 as threshold for binary classification.
        # Some unit tests call this method on a lightweight dummy instance which
        # may not inherit BaseMLStrategy. Fall back to the base helpers to keep
        # behavior consistent without requiring full initialization.
        try:
            decision = self.decision_from_prediction(signal.prediction)
        except AttributeError:
            from ml.strategies.base import BaseMLStrategy as _Base

            decision = _Base.decision_from_prediction(self, signal.prediction)
        try:
            target_side = self.target_side_from_prediction(signal.prediction, 0.5)
        except AttributeError:
            from ml.strategies.base import BaseMLStrategy as _Base

            target_side = _Base.target_side_from_prediction(self, signal.prediction, 0.5)

        if decision == "HOLD":
            signal_direction = "HOLD"
        elif target_side == OrderSide.BUY:
            signal_direction = "LONG"
        else:
            signal_direction = "SHORT"
        decision_type = "BUY" if target_side == OrderSide.BUY else "SELL"
        short_entry_policy = self._resolve_short_entry_policy()
        short_policy_value = getattr(short_entry_policy, "value", str(short_entry_policy))
        persist_hold_on_block = bool(
            getattr(self._config, "persist_hold_on_short_entry_block", False),
        )

        self._update_returns_from_signal(signal)

        # Log signal details
        model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id", "unknown")
        self.log.info(
            f"Processing ML signal from {model_id}: "
            f"prediction={signal.prediction:.3f}, "
            f"confidence={signal.confidence:.3f}, "
            f"direction={signal_direction}",
        )

        # Calculate position size for persistence
        position_size = self._calculate_position_size()

        # Prepare risk metrics
        risk_metrics = {
            "confidence": float(signal.confidence),
            "prediction": float(signal.prediction),
            "active_positions": float(self._active_positions),
            "has_position": 1.0 if current_position is not None else 0.0,
        }

        # Prepare execution params
        execution_params = {
            "target_side": target_side.name,
            "model_id": model_id,
            "action": None,  # Will be set based on decision
        }

        if decision == "HOLD" and current_position is None:
            execution_params["action"] = "hold"
            execution_params["reason"] = "neutral_band"
            self._persist_strategy_decision(
                signal=signal,
                decision_type="HOLD",
                position_size=None,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
            )
            self.log.info(
                "Neutral-band signal; holding flat",
            )
            return

        # Check if we need to change position
        if current_position is None:
            if target_side == OrderSide.SELL and short_entry_policy is not ShortEntryPolicy.ALLOW:
                execution_params["action"] = "hold"
                execution_params["short_entry_policy"] = short_policy_value
                execution_params["reason"] = "short_entry_blocked"
                self._persist_strategy_decision(
                    signal=signal,
                    decision_type="HOLD",
                    position_size=None,
                    risk_metrics=risk_metrics,
                    execution_params=execution_params,
                    persist_hold=persist_hold_on_block,
                )
                try:
                    short_entry_blocked_total.labels(
                        strategy_id=str(self.id),
                        policy=short_policy_value,
                    ).inc()
                except Exception as exc:
                    self.log.debug(
                        "ml_strategy.short_entry_blocked_metric_failed",
                        strategy_id=str(self.id),
                        exc_info=True,
                        error=str(exc),
                    )
                self.log.info(
                    f"Short entry blocked by policy ({short_policy_value}); holding flat",
                )
                return
            # No position, enter new one
            execution_params["action"] = "enter"
            self._persist_strategy_decision(
                signal=signal,
                decision_type=decision_type,
                position_size=position_size,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
            )
            if self._config.execute_trades:
                self._enter_position(target_side, signal)
            else:
                self._dry_run_trades += 1
                self.log.info(
                    f"[DRY RUN] Would enter {target_side.name} position "
                    f"(execute_trades=False) - Total dry run trades: {self._dry_run_trades}",
                )
            return

        exit_payload = self._evaluate_exit_policy(
            current_position,
            instrument_id=signal.instrument_id,
        )
        if exit_payload is not None:
            exit_side = self._exit_side_for_position(current_position)
            execution_params["action"] = "exit"
            execution_params["exit"] = exit_payload
            self._persist_strategy_decision(
                signal=signal,
                decision_type=exit_side.name if exit_side is not None else decision_type,
                position_size=current_position.quantity,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
            )
            if self._config.execute_trades:
                if exit_side is None:
                    self.log.warning("Exit requested but position side is unknown")
                    return
                self._set_exit_intent_metadata(exit_payload)
                self._place_market_order(
                    exit_side,
                    current_position.quantity,
                    reduce_only=True,
                )
            else:
                self._dry_run_trades += 1
                self.log.info(
                    "[DRY RUN] Would exit %s position due to %s (execute_trades=False)",
                    current_position.side.name,
                    exit_payload.get("reason"),
                )
            return

        model_exit_config = getattr(self._config, "model_exit_config", None)
        if model_exit_config is not None:
            try:
                current_price = self._resolve_market_price(signal.instrument_id)
            except AttributeError:
                current_price = None
            model_exit = evaluate_model_exit(
                position=current_position,
                signal=signal,
                config=model_exit_config,
                now_ns=self._timestamp_ns(),
                trigger_price=current_price,
            )
            if model_exit is not None:
                model_exit_total.labels(action=model_exit.action, reason=model_exit.reason).inc()
                exit_payload = model_exit.to_metadata()
                exit_payload["exit_on_flip"] = model_exit_config.exit_on_flip
                exit_payload["reverse_on_flip"] = model_exit_config.reverse_on_flip
                if model_exit_config.exit_confidence_threshold is not None:
                    exit_payload["exit_confidence_threshold"] = float(
                        model_exit_config.exit_confidence_threshold,
                    )
                if model_exit_config.exit_prediction_band > 0.0:
                    exit_payload["exit_prediction_band"] = float(
                        model_exit_config.exit_prediction_band,
                    )
                if model_exit_config.min_hold_ms is not None:
                    exit_payload["min_hold_ms"] = int(model_exit_config.min_hold_ms)
                execution_params["action"] = model_exit.action
                execution_params["exit"] = exit_payload
                if model_exit.action == "reverse":
                    if target_side == OrderSide.SELL and short_entry_policy is not ShortEntryPolicy.ALLOW:
                        exit_side = self._exit_side_for_position(current_position)
                        exit_payload["short_entry_policy"] = short_policy_value
                        exit_payload["blocked_action"] = "reverse"
                        execution_params["action"] = "exit"
                        execution_params["exit"] = exit_payload
                        self._persist_strategy_decision(
                            signal=signal,
                            decision_type=exit_side.name if exit_side is not None else decision_type,
                            position_size=current_position.quantity,
                            risk_metrics=risk_metrics,
                            execution_params=execution_params,
                        )
                        if self._config.execute_trades:
                            if exit_side is None:
                                self.log.warning("Exit requested but position side is unknown")
                                return
                            self._set_exit_intent_metadata(exit_payload)
                            self._place_market_order(
                                exit_side,
                                current_position.quantity,
                                reduce_only=True,
                            )
                        else:
                            self._dry_run_trades += 1
                            self.log.info(
                                "[DRY RUN] Would exit %s position due to short-entry policy (execute_trades=False)",
                                current_position.side.name,
                            )
                        return
                    execution_params["current_side"] = current_position.side.name
                    self._persist_strategy_decision(
                        signal=signal,
                        decision_type=decision_type,
                        position_size=position_size,
                        risk_metrics=risk_metrics,
                        execution_params=execution_params,
                    )
                    if self._config.execute_trades:
                        self._set_exit_intent_metadata(exit_payload)
                        self._reverse_position(current_position, target_side, signal)
                    else:
                        self._dry_run_trades += 1
                        self.log.info(
                            "[DRY RUN] Would reverse %s position due to %s (execute_trades=False)",
                            current_position.side.name,
                            model_exit.reason,
                        )
                    return

                exit_side = self._exit_side_for_position(current_position)
                self._persist_strategy_decision(
                    signal=signal,
                    decision_type=exit_side.name if exit_side is not None else decision_type,
                    position_size=current_position.quantity,
                    risk_metrics=risk_metrics,
                    execution_params=execution_params,
                )
                if self._config.execute_trades:
                    if exit_side is None:
                        self.log.warning("Exit requested but position side is unknown")
                        return
                    self._set_exit_intent_metadata(exit_payload)
                    self._place_market_order(
                        exit_side,
                        current_position.quantity,
                        reduce_only=True,
                    )
                else:
                    self._dry_run_trades += 1
                    self.log.info(
                        "[DRY RUN] Would exit %s position due to %s (execute_trades=False)",
                        current_position.side.name,
                        model_exit.reason,
                    )
                return

        if self._should_reverse_position(current_position, target_side):
            # Position exists but signal suggests opposite direction
            execution_params["action"] = "reverse"
            execution_params["current_side"] = current_position.side.name
            try:
                current_price = self._resolve_market_price(signal.instrument_id)
            except AttributeError:
                current_price = None
            time_in_trade_ns = self._time_in_trade_ns(current_position, self._timestamp_ns())
            execution_params["exit"] = self._build_exit_metadata(
                reason="reverse",
                trigger_price=current_price,
                time_in_trade_ns=time_in_trade_ns,
            )
            if target_side == OrderSide.SELL and short_entry_policy is not ShortEntryPolicy.ALLOW:
                exit_side = self._exit_side_for_position(current_position)
                execution_params["action"] = "exit"
                execution_params["exit"] = self._build_exit_metadata(
                    reason="short_entry_policy_exit_only",
                    trigger_price=current_price,
                    time_in_trade_ns=time_in_trade_ns,
                )
                execution_params["short_entry_policy"] = short_policy_value
                self._persist_strategy_decision(
                    signal=signal,
                    decision_type=exit_side.name if exit_side is not None else decision_type,
                    position_size=current_position.quantity,
                    risk_metrics=risk_metrics,
                    execution_params=execution_params,
                )
                if self._config.execute_trades:
                    if exit_side is None:
                        self.log.warning("Exit requested but position side is unknown")
                        return
                    self._set_exit_intent_metadata(execution_params["exit"])
                    self._place_market_order(
                        exit_side,
                        current_position.quantity,
                        reduce_only=True,
                    )
                else:
                    self._dry_run_trades += 1
                    self.log.info(
                        "[DRY RUN] Would exit %s position due to short-entry policy (execute_trades=False)",
                        current_position.side.name,
                    )
                return
            self._persist_strategy_decision(
                signal=signal,
                decision_type=decision_type,
                position_size=position_size,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
            )
            if self._config.execute_trades:
                self._set_exit_intent_metadata(execution_params["exit"])
                self._reverse_position(current_position, target_side, signal)
            else:
                self._dry_run_trades += 1
                self.log.info(
                    f"[DRY RUN] Would reverse position from {current_position.side.name} "
                    f"to {target_side.name} (execute_trades=False) - "
                    f"Total dry run trades: {self._dry_run_trades}",
                )

        else:
            # Position aligns with signal - HOLD
            execution_params["action"] = "hold"
            self.log.debug(
                (
                    f"Position already {current_position.side.name}, aligns with signal "
                    f"direction {signal_direction}"
                ),
            )

            # Persist HOLD decision if configured
            self._persist_strategy_decision(
                signal=signal,
                decision_type="HOLD",
                position_size=None,
                risk_metrics=risk_metrics,
                execution_params=execution_params,
            )

            # Could add logic here to increase position size or adjust stops

    def _enter_position(self, side: OrderSide, signal: MLSignal) -> None:
        """
        Enter a new position based on ML signal.

        Parameters
        ----------
        side : OrderSide
            The side to enter (BUY or SELL).
        signal : MLSignal
            The signal triggering the entry.

        """
        if side == OrderSide.SELL and self._resolve_short_entry_policy() is not ShortEntryPolicy.ALLOW:
            self.log.info(
                "Short entry blocked by policy; skipping entry",
            )
            return
        if self._should_block_entry_orders():
            self.log.info(
                "Intent entry suppressed due to max_positions; reduce-only orders only",
            )
            return
        # Determine quantity via sizer + risk gate (fallback to legacy sizing for test doubles)
        try:
            quantity = self.size_and_validate(signal)
        except AttributeError:
            quantity = self._calculate_position_size()
        if quantity is None:
            reject_reason = self._get_sizing_reject_reason() or "unknown"
            persist_hold_on_reject = bool(
                getattr(self._config, "persist_hold_on_sizing_reject", False),
            )
            execution_params = {
                "action": "hold",
                "reason": "sizing_rejected",
                "sizing_reject_reason": reject_reason,
                "target_side": side.name,
                "intended_action": "enter",
            }
            self._persist_strategy_decision(
                signal=signal,
                decision_type="HOLD",
                position_size=None,
                risk_metrics=None,
                execution_params=execution_params,
                persist_hold=persist_hold_on_reject,
            )
            self.log.warning(
                f"Cannot enter position due to sizing failure for {signal.instrument_id} "
                f"(reason={reject_reason})",
            )
            return

        # Check if trading is enabled
        if not self._config.execute_trades:
            self.log.info(
                f"[DRY RUN] Would place {side.name} order for {quantity} units (execute_trades=False)",
            )
            # Still update position counter for tracking purposes
            self._active_positions += 1
            return

        # Place the order
        try:
            order_id = self._submit_smart_order(side, quantity, signal)
        except AttributeError:
            try:
                order_id = self._place_market_order(side, quantity)
            except AttributeError:
                order_id = None
        if order_id is None:
            self.log.error("Order submission failed")
            return
        self._active_positions += 1

        # Track the signal that triggered this trade
        model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id", "unknown")
        self.log.info(
            f"Entering {side.name} position: {quantity} units based on signal from {model_id}",
        )

        # Store signal info for performance tracking
        if self.track_performance:
            self._track_trade_entry(model_id, signal, order_id)

        # Note: Decision already persisted in _process_ml_signal

    def _reverse_position(
        self,
        current_position: Position,
        target_side: OrderSide,
        signal: MLSignal,
    ) -> None:
        """
        Reverse an existing position based on ML signal.

        Parameters
        ----------
        current_position : Position
            The current position to reverse.
        target_side : OrderSide
            The new target side.
        signal : MLSignal
            The signal triggering the reversal.

        """
        # Close current position first
        close_side = OrderSide.SELL if current_position.side.name == "LONG" else OrderSide.BUY

        self.log.info(f"Reversing position from {current_position.side.name} to {target_side.name}")

        # Check if trading is enabled
        if not self._config.execute_trades:
            quantity = self._calculate_position_size()
            self.log.info(
                "[DRY RUN] Would close %s position and open %s for %s units (execute_trades=False)",
                current_position.side.name,
                target_side.name,
                quantity,
            )
            return

        # Close existing position
        self._place_market_order(
            close_side,
            current_position.quantity,
            reduce_only=True,
        )

        # Open new position in opposite direction
        if self._should_block_entry_orders():
            self.log.info(
                "Intent entry suppressed due to max_positions; reduce-only orders only",
            )
            return
        try:
            quantity = self.size_and_validate(signal)
        except AttributeError:
            quantity = self._calculate_position_size()
        if quantity is None:
            self.log.warning("Closed position but cannot open new one due to sizing failure")
            return

        try:
            order_id = self._submit_smart_order(target_side, quantity, signal)
        except AttributeError:
            try:
                order_id = self._place_market_order(target_side, quantity)
            except AttributeError:
                order_id = None
        if order_id is None:
            self.log.error("Order submission failed during reversal")
            return

        # Track the reversal
        model_id = getattr(signal, "model_id", None) or signal.metadata.get("model_id", "unknown")
        if self.track_performance:
            self._track_trade_entry(model_id, signal, order_id)

    def _should_reverse_position(
        self,
        current_position: Position,
        target_side: OrderSide,
    ) -> bool:
        """
        Check if position should be reversed.

        Parameters
        ----------
        current_position : Position
            The current position.
        target_side : OrderSide
            The target side from signal.

        Returns
        -------
        bool
            True if position should be reversed.

        """
        # Some unit tests bind this method onto lightweight doubles without
        # a custom should_reverse implementation. Guard the call and fall back
        # to a simple heuristic: reverse when LONG->SELL or SHORT->BUY.
        try:
            fn = getattr(self, "should_reverse")
            if callable(fn):
                return bool(fn(current_position, target_side))
        except Exception as exc:
            # Non-fatal: fallback to heuristic below and log at debug level
            self.log.debug("should_reverse hook failed; using heuristic: %s", exc)
        side_name = getattr(getattr(current_position, "side", object()), "name", "")
        if side_name == "LONG" and target_side == OrderSide.SELL:
            return True
        if side_name == "SHORT" and target_side == OrderSide.BUY:
            return True
        return False

    def _track_trade_entry(
        self,
        model_id: str,
        signal: MLSignal,
        order_id: ClientOrderId,
    ) -> None:
        """
        Track trade entry for performance analysis.

        Parameters
        ----------
        model_id : str
            The model that generated the signal.
        signal : MLSignal
            The signal that triggered the trade.
        order_id : ClientOrderId
            The order ID for tracking.

        """
        # Store mapping of order_id to model_id for later performance tracking
        if not hasattr(self, "_order_to_model"):
            self._order_to_model = {}

        self._order_to_model[str(order_id)] = {
            "model_id": model_id,
            "signal": signal,
            "entry_time": self.clock.timestamp_ns(),
        }

    def on_order_filled(self, event: OrderFilled) -> None:
        """
        Handle order fills and track model performance.

        Parameters
        ----------
        event : OrderFilled
            The order filled event.

        """
        super().on_order_filled(event)

        # Track performance if this was a closing order
        if hasattr(self, "_order_to_model") and self.track_performance:
            order_id = str(event.client_order_id)

            # Check if this is a tracked order
            if order_id in self._order_to_model:
                order_info = self._order_to_model[order_id]
                model_id = order_info["model_id"]

                # Calculate P&L (simplified - real implementation would use actual fills)
                # This is just for demonstration
                if event.order_side.name == "SELL":
                    # Assuming we're closing a long position
                    pnl = float(event.last_px.as_double()) - float(event.avg_px.as_double())
                else:
                    # Assuming we're closing a short position
                    pnl = float(event.avg_px.as_double()) - float(event.last_px.as_double())

                # Update model performance
                self._update_model_performance(model_id, pnl)

                self.log.info(f"Trade completed for model {model_id}: P&L = {pnl:.2f}")

                # Clean up tracking
                del self._order_to_model[order_id]


class MultiModelMLStrategy(MLTradingStrategy):
    """
    Extended ML strategy specifically designed for multi-model aggregation.

    This strategy extends the base MLTradingStrategy with additional
    features for handling multiple models:
    - Weighted consensus decisions
    - Model performance-based dynamic weighting
    - Conflict resolution strategies

    """

    def __init__(self, config: Any, stores: object | None = None) -> None:
        """
        Initialize multi-model strategy with dependency injection support.

        Parameters
        ----------
        config : MLStrategyConfig
            The strategy configuration.
        stores : ActorStoresRegistries, optional
            Container with all 4 stores and 4 registries from init_ml_stores_and_registries.
            If not provided, stores may be initialized based on config.

        """
        super().__init__(config, stores)

        # Enable performance tracking for dynamic weighting
        self.track_performance = True
        self.use_dynamic_weights = getattr(config, "use_dynamic_weights", False)

    def _get_dynamic_model_weights(self) -> dict[str, float]:
        """
        Calculate dynamic weights based on model performance.

        Returns
        -------
        dict[str, float]
            Model weights based on historical performance.

        """
        if not self._model_performance:
            # No performance data yet, use equal weights
            return {}

        weights = {}
        for model_id, perf in self._model_performance.items():
            # Weight based on accuracy and total profit
            accuracy = perf.get("accuracy", 0.5)
            total_profit = perf.get("total_profit", 0.0)
            total_trades = perf.get("total_trades", 1)

            # Combine accuracy and profit per trade
            profit_per_trade = total_profit / max(total_trades, 1)

            # Simple weighting formula (can be customized)
            weight = accuracy * (1.0 + np.tanh(profit_per_trade / 100.0))
            weights[model_id] = max(weight, 0.1)  # Minimum weight of 0.1

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _aggregate_signal(self, signal: MLSignal) -> None:
        """
        Override aggregation to use dynamic weights if enabled.

        Parameters
        ----------
        signal : MLSignal
            The signal to aggregate.

        """
        # Update model weights if using dynamic weighting
        if self.use_dynamic_weights:
            self.model_weights = self._get_dynamic_model_weights()

        # Call parent aggregation
        super()._aggregate_signal(signal)
