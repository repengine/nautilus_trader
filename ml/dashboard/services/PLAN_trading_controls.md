# Trading Control Buttons Implementation Plan

## Overview

This document analyzes the dashboard's top-bar trading control buttons and provides a comprehensive implementation plan for connecting them to Nautilus Trader backend functionality.

## Current UI Elements Analysis

### 1. "🔌 Connect System" Button

**Purpose**: Establishes connection to Nautilus Trader system components
**Current Implementation**: Basic connectivity check via `/api/control/status`

**Required Backend Functionality**:
```python
# ml/dashboard/services/system_connector.py
class SystemConnector:
    """Manages system-wide connections and health checks."""

    def __init__(self):
        self.nautilus_node: TradingNode | None = None
        self.data_engine_connected = False
        self.exec_engine_connected = False
        self.cache_connected = False

    async def connect_system(self) -> dict[str, Any]:
        """Connect to all Nautilus components."""
        try:
            # Initialize Nautilus TradingNode
            from nautilus_trader.live.node import TradingNode
            from nautilus_trader.config import TradingNodeConfig

            config = TradingNodeConfig.from_env()
            self.nautilus_node = TradingNode(config=config)

            # Connect data engine
            if self.nautilus_node.kernel.data_engine:
                self.nautilus_node.kernel.data_engine.connect()
                self.data_engine_connected = True

            # Connect execution engine
            if self.nautilus_node.kernel.exec_engine:
                self.nautilus_node.kernel.exec_engine.connect()
                self.exec_engine_connected = True

            # Verify cache connection
            self.cache_connected = bool(self.nautilus_node.kernel.cache)

            return {
                "success": True,
                "status": "connected",
                "components": {
                    "data_engine": self.data_engine_connected,
                    "exec_engine": self.exec_engine_connected,
                    "cache": self.cache_connected,
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
```

**Data Sources**:
- Nautilus TradingNode state
- Data/Execution engine connection status
- Cache health status

**Update Frequency**: On-demand (button click + periodic health checks every 30s)

**Safety Requirements**:
- Connection timeout (30s max)
- Graceful degradation if components unavailable
- Clear error messages for connection failures

### 2. "🟢 LIVE TRADING" Button

**Purpose**: Controls live trading state (PAPER/LIVE toggle)
**Current Implementation**: Basic toggle via `toggleLiveTrading()` JavaScript function

**Required Backend Functionality**:
```python
# ml/dashboard/services/trading_controller.py
class TradingController:
    """Controls trading state and execution modes."""

    def __init__(self, trading_node: TradingNode):
        self.trading_node = trading_node
        self.trading_state = "STOPPED"  # STOPPED, PAPER, LIVE

    async def toggle_trading_mode(self) -> dict[str, Any]:
        """Toggle between PAPER and LIVE trading modes."""
        try:
            if self.trading_state == "STOPPED":
                # Start in PAPER mode first for safety
                await self._start_paper_trading()
                self.trading_state = "PAPER"
                return {
                    "success": True,
                    "mode": "PAPER",
                    "message": "Paper trading started"
                }
            elif self.trading_state == "PAPER":
                # Require explicit confirmation for LIVE mode
                await self._start_live_trading()
                self.trading_state = "LIVE"
                return {
                    "success": True,
                    "mode": "LIVE",
                    "message": "⚠️ LIVE TRADING ACTIVE",
                    "warning": True
                }
            else:  # LIVE
                await self._stop_trading()
                self.trading_state = "STOPPED"
                return {
                    "success": True,
                    "mode": "STOPPED",
                    "message": "Trading stopped"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _start_live_trading(self):
        """Enable live trading with all safety checks."""
        # Pre-flight safety checks
        await self._run_safety_checks()

        # Set execution mode to LIVE
        if self.trading_node.kernel.exec_engine:
            # Configure for live execution
            pass

    async def _run_safety_checks(self):
        """Run comprehensive safety checks before live trading."""
        checks = [
            self._check_account_balance(),
            self._check_risk_limits(),
            self._check_market_hours(),
            self._check_model_health(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)
        failures = [r for r in results if isinstance(r, Exception)]

        if failures:
            raise RuntimeError(f"Safety checks failed: {failures}")
```

**Data Sources**:
- TradingNode execution state
- Account balances and positions
- Risk management parameters
- Model performance metrics

**Update Frequency**: Real-time state changes + UI updates every 1s

**Safety Requirements**:
- MANDATORY safety checks before LIVE mode
- Account balance verification
- Risk limit validation
- Model performance thresholds
- Market hours verification
- Position size limits

### 3. "🛑 Emergency Stop" Button

**Purpose**: Immediately halts all trading activity and cancels orders
**Current Implementation**: Calls `/api/control/emergency_stop`

**Required Backend Functionality**:
```python
# ml/dashboard/services/emergency_controller.py
class EmergencyController:
    """Handles emergency stop procedures."""

    def __init__(self, trading_node: TradingNode):
        self.trading_node = trading_node
        self.stop_in_progress = False

    async def emergency_stop(self) -> dict[str, Any]:
        """Execute emergency stop sequence."""
        if self.stop_in_progress:
            return {"success": False, "error": "Emergency stop already in progress"}

        self.stop_in_progress = True
        stop_time = datetime.now(UTC)

        try:
            # 1. Cancel all open orders (highest priority)
            cancelled_orders = await self._cancel_all_orders()

            # 2. Close all open positions (if configured)
            closed_positions = await self._close_positions_if_configured()

            # 3. Stop all ML actors
            stopped_actors = await self._stop_all_actors()

            # 4. Disconnect from exchanges
            await self._disconnect_exchanges()

            # 5. Set trading node to stopped state
            if hasattr(self.trading_node, 'stop_async'):
                await self.trading_node.stop_async()

            return {
                "success": True,
                "stop_time": stop_time.isoformat(),
                "cancelled_orders": cancelled_orders,
                "closed_positions": closed_positions,
                "stopped_actors": stopped_actors,
                "message": "🛑 Emergency stop completed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Emergency stop failed: {str(e)}",
                "partial_stop": True
            }
        finally:
            self.stop_in_progress = False

    async def _cancel_all_orders(self) -> list[str]:
        """Cancel all open orders across all venues."""
        cancelled = []

        if self.trading_node.kernel.exec_engine:
            # Get all open orders from cache
            cache = self.trading_node.kernel.cache
            open_orders = cache.orders_open()

            for order in open_orders:
                try:
                    # Submit cancel command
                    cancel_command = CancelOrder(
                        trader_id=self.trading_node.trader_id,
                        strategy_id=order.strategy_id,
                        instrument_id=order.instrument_id,
                        client_order_id=order.client_order_id,
                        venue_order_id=order.venue_order_id,
                        command_id=UUID4(),
                        ts_init=self.trading_node.kernel.clock.timestamp_ns(),
                    )

                    self.trading_node.kernel.exec_engine.execute(cancel_command)
                    cancelled.append(str(order.client_order_id))

                except Exception as e:
                    self.trading_node.kernel.logger.error(
                        f"Failed to cancel order {order.client_order_id}: {e}"
                    )

        return cancelled
```

**Data Sources**:
- Open orders from cache
- Active positions
- Running ML actors
- Exchange connection status

**Update Frequency**: Immediate execution (< 1s response time)

**Safety Requirements**:
- Order cancellation takes absolute priority
- Timeout protection (max 10s total execution)
- Partial stop state handling
- Comprehensive logging
- No exceptions should prevent order cancellation

### 4. Market Ticker Displays (SPY, QQQ, VIX, BTC)

**Purpose**: Real-time market data display with prices and % changes
**Current Implementation**: Static data with periodic JavaScript updates

**Required Backend Functionality**:
```python
# ml/dashboard/services/market_data_service.py
class MarketDataService:
    """Provides real-time market data for dashboard tickers."""

    def __init__(self, trading_node: TradingNode):
        self.trading_node = trading_node
        self.subscribed_symbols = ["SPY", "QQQ", "VIX", "BTCUSD"]
        self.last_prices = {}
        self.daily_changes = {}
        self.websocket_connections = {}

    async def start_market_data_feed(self):
        """Initialize real-time market data subscriptions."""
        data_engine = self.trading_node.kernel.data_engine

        for symbol in self.subscribed_symbols:
            try:
                # Subscribe to real-time quotes/trades
                instrument_id = InstrumentId.from_str(f"{symbol}.VENUE")

                # Subscribe to quotes for real-time pricing
                subscribe_quote = Subscribe(
                    client_id=ClientId("DATA_CLIENT"),
                    venue=Venue("VENUE"),
                    data_type=QuoteTick,
                    metadata={"instrument_id": instrument_id},
                    command_id=UUID4(),
                    ts_init=self.trading_node.kernel.clock.timestamp_ns(),
                )

                data_engine.execute(subscribe_quote)

            except Exception as e:
                self.trading_node.kernel.logger.error(
                    f"Failed to subscribe to {symbol}: {e}"
                )

    def get_ticker_data(self) -> dict[str, dict[str, Any]]:
        """Get current ticker data for UI display."""
        ticker_data = {}

        for symbol in self.subscribed_symbols:
            price = self.last_prices.get(symbol, 0.0)
            daily_change = self.daily_changes.get(symbol, 0.0)

            ticker_data[symbol.lower()] = {
                "price": self._format_price(symbol, price),
                "change_pct": f"{daily_change:+.1f}%",
                "direction": "up" if daily_change >= 0 else "down",
                "timestamp": datetime.now(UTC).isoformat()
            }

        return ticker_data

    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to asset type."""
        if symbol == "BTC":
            return f"${price:,.0f}"
        elif symbol in ["SPY", "QQQ"]:
            return f"${price:.2f}"
        else:  # VIX
            return f"{price:.2f}"
```

**Data Sources**:
- Real-time market data feeds (Databento, exchange APIs)
- Historical daily open prices for % change calculation
- WebSocket connections for live updates

**Update Frequency**: Real-time (< 100ms latency for price updates)

**Safety Requirements**:
- Fallback to delayed data if real-time feed fails
- Rate limiting to prevent API quota exhaustion
- Error handling for missing/stale data
- Circuit breaker for failing data sources

## WebSocket Implementation for Live Updates

```python
# ml/dashboard/services/websocket_service.py
class DashboardWebSocketService:
    """Manages WebSocket connections for real-time dashboard updates."""

    def __init__(self):
        self.connections: set[WebSocket] = set()
        self.update_tasks: dict[str, asyncio.Task] = {}

    async def start_live_updates(self):
        """Start all real-time update tasks."""
        self.update_tasks["market_data"] = asyncio.create_task(
            self._market_data_update_loop()
        )
        self.update_tasks["trading_status"] = asyncio.create_task(
            self._trading_status_update_loop()
        )

    async def _market_data_update_loop(self):
        """Send market data updates to all connected clients."""
        while True:
            try:
                ticker_data = self.market_data_service.get_ticker_data()
                await self._broadcast({
                    "type": "market_update",
                    "data": ticker_data
                })
                await asyncio.sleep(1.0)  # 1 second updates
            except Exception as e:
                logger.error(f"Market data update failed: {e}")
                await asyncio.sleep(5.0)  # Back off on errors

    async def _broadcast(self, message: dict):
        """Broadcast message to all connected WebSocket clients."""
        if not self.connections:
            return

        # Remove closed connections
        disconnected = set()
        for ws in self.connections.copy():
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)

        self.connections -= disconnected
```

## Integration with Existing Dashboard Architecture

### 1. Enhanced Control Panel

```python
# ml/dashboard/control_enhanced.py (extend existing)
class EnhancedControlPanel(SimpleControlPanel):
    """Extended control panel with Nautilus integration."""

    def __init__(self):
        super().__init__()
        self.system_connector = SystemConnector()
        self.trading_controller: TradingController | None = None
        self.emergency_controller: EmergencyController | None = None
        self.market_data_service: MarketDataService | None = None

    async def initialize_nautilus_integration(self):
        """Initialize full Nautilus Trader integration."""
        # Connect to Nautilus system
        result = await self.system_connector.connect_system()

        if result["success"] and self.system_connector.nautilus_node:
            node = self.system_connector.nautilus_node

            # Initialize controllers
            self.trading_controller = TradingController(node)
            self.emergency_controller = EmergencyController(node)
            self.market_data_service = MarketDataService(node)

            # Start market data feed
            await self.market_data_service.start_market_data_feed()

        return result
```

### 2. New API Endpoints

```python
# ml/dashboard/app.py (additions to existing endpoints)

@app.post("/api/control/system/connect")
def control_connect_system() -> tuple[Any, int]:
    """Connect to Nautilus Trader system."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    import asyncio
    from ml.dashboard.control_enhanced import EnhancedControlPanel

    control_panel = EnhancedControlPanel.from_env()
    result = asyncio.run(control_panel.initialize_nautilus_integration())
    return jsonify(result), 200 if result.get("success") else 400

@app.post("/api/control/trading/toggle")
def control_toggle_trading() -> tuple[Any, int]:
    """Toggle trading mode between PAPER and LIVE."""
    if not _require_token():
        return jsonify({"error": "unauthorized"}), 401

    # Implementation with enhanced control panel
    # ... (as defined above)

@app.get("/api/control/market/tickers")
def control_market_tickers() -> tuple[Any, int]:
    """Get current market ticker data."""
    from ml.dashboard.control_enhanced import EnhancedControlPanel

    control_panel = EnhancedControlPanel.from_env()
    if control_panel.market_data_service:
        data = control_panel.market_data_service.get_ticker_data()
        return jsonify(data), 200

    return jsonify({"error": "market_data_unavailable"}), 503
```

## Frontend JavaScript Enhancements

```javascript
// Enhanced button implementations
async function connectToSystem() {
    try {
        showLoadingState('🔄 Connecting...');
        const response = await fetch('/api/control/system/connect', {
            method: 'POST',
            headers: {'X-ML-DASHBOARD-TOKEN': getAuthToken()}
        });
        const data = await response.json();

        if (data.success) {
            updateConnectionStatus('connected', data.components);
            startLiveDataFeeds();  // Begin real-time updates
            showSuccessMessage('✅ Connected to Nautilus Trader');
        } else {
            showErrorMessage(`❌ Connection failed: ${data.error}`);
        }
    } catch (error) {
        showErrorMessage(`❌ Connection error: ${error.message}`);
    } finally {
        hideLoadingState();
    }
}

async function toggleLiveTrading() {
    // Safety confirmation for LIVE mode
    const currentMode = getCurrentTradingMode();
    if (currentMode === 'PAPER') {
        const confirmed = await showConfirmationDialog(
            '⚠️ Switch to LIVE Trading?',
            'This will execute trades with real money. Are you sure?'
        );
        if (!confirmed) return;
    }

    try {
        const response = await fetch('/api/control/trading/toggle', {
            method: 'POST',
            headers: {'X-ML-DASHBOARD-TOKEN': getAuthToken()}
        });
        const data = await response.json();

        if (data.success) {
            updateTradingModeUI(data.mode);
            if (data.warning) {
                showWarningMessage(data.message);
            }
        }
    } catch (error) {
        showErrorMessage(`Trading mode toggle failed: ${error.message}`);
    }
}

// WebSocket client for real-time updates
class DashboardWebSocket {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onclose = () => {
            this.scheduleReconnect();
        };
    }

    handleMessage(data) {
        switch (data.type) {
            case 'market_update':
                updateMarketTickers(data.data);
                break;
            case 'trading_status':
                updateTradingStatus(data.data);
                break;
            case 'emergency_alert':
                showEmergencyAlert(data.message);
                break;
        }
    }
}
```

## Implementation Priority & Timeline

### Phase 1 (Week 1): Core Infrastructure
1. System connector with basic Nautilus integration
2. Enhanced control panel base architecture
3. Safety validation framework

### Phase 2 (Week 2): Trading Controls
1. Trading mode toggle with safety checks
2. Emergency stop functionality
3. Enhanced API endpoints

### Phase 3 (Week 3): Market Data Integration
1. Real-time market data service
2. WebSocket infrastructure
3. Live ticker updates

### Phase 4 (Week 4): UI Enhancements & Testing
1. Enhanced JavaScript client
2. Error handling and fallbacks
3. Comprehensive testing

## Risk Mitigation

1. **Trading Safety**: Multiple confirmation layers before LIVE mode
2. **Emergency Stop**: Order cancellation takes absolute priority
3. **Data Reliability**: Fallback mechanisms for all data sources
4. **Connection Resilience**: Automatic reconnection with exponential backoff
5. **Performance**: All critical operations complete within safety timeouts

This implementation plan provides a robust foundation for connecting the dashboard's trading controls to Nautilus Trader's powerful trading infrastructure while maintaining the highest safety standards for live trading operations.