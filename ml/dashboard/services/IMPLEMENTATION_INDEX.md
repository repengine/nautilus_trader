# Nautilus ML Dashboard Services – Implementation Index

## Overview

This directory groups the planning material for connecting the Nautilus ML
dashboard UI to backend functionality. Nothing in this index is implemented yet;
use it to understand the intended scope before writing code. The actual Python
package currently exposes only stubs (see `README.md`).

## Planned Architecture

The future service layer is expected to live in `ml/dashboard/services/` with a
module-per-domain layout. The structure below is **proposed**, not present today:

```
ml/dashboard/services/
├── __init__.py                       # Service layer initialization (future)
├── actors_service.py                 # Actor lifecycle management (planned)
├── pipelines_service.py              # Pipeline orchestration (planned)
├── metrics_service.py                # Live metrics aggregation (planned)
├── trading_service.py                # Trading operations (planned)
├── features_service.py               # Feature engineering (planned)
├── models_service.py                 # Model management (planned)
├── api_service.py                    # API documentation & testing (planned)
└── terminal_service.py               # Terminal & configuration (planned)
```

## Implementation Plans

### 1. Trading Controls (21KB)
**File:** `PLAN_trading_controls.md`
**UI Components:**
- 🔌 Connect System button
- 🟢 LIVE TRADING button
- 🛑 Emergency Stop button
- Market ticker displays

**Key Backend Connections:**
- Nautilus TradingNode integration
- Real-time market data feeds
- Emergency stop procedures with order cancellation
- WebSocket infrastructure for live updates

---

### 2. Actor Management (28KB)
**File:** `PLAN_actor_management.md`
**UI Components:**
- Model Performance & P&L table
- Action buttons (Pause/Resume, Config, Promote, Stop)
- Deploy New Actor section
- Hot reload functionality
- A/B testing indicators

**Key Backend Connections:**
- BaseMLInferenceActor lifecycle management
- ModelRegistry integration
- Real-time health monitoring
- Performance metrics aggregation

---

### 3. Pipeline Orchestration (16KB)
**File:** `PLAN_pipeline_orchestration.md`
**UI Components:**
- Dataset Building section
- Model Training section
- Hyperparameter Tuning section
- Scheduled Pipelines table
- Pipeline progress monitoring

**Key Backend Connections:**
- MLPipelineOrchestrator integration
- Job queue management with PostgreSQL
- Async execution with worker pools
- WebSocket-based progress tracking

---

### 4. Metrics & Monitoring (13KB)
**File:** `PLAN_metrics_monitoring.md`
**UI Components:**
- KPI cards (P&L, Sharpe, Win Rate, etc.)
- Live Data Ingestion Monitor
- Portfolio & Active Positions
- System Health & Resources
- Active Experiments tracking

**Key Backend Connections:**
- Store integration (DataStore, ModelStore, StrategyStore)
- Prometheus metrics collection
- WebSocket + polling hybrid updates
- Progressive fallback chains

---

### 5. Feature Engineering (38KB)
**File:** `PLAN_feature_engineering.md`
**UI Components:**
- Feature Designer with checkboxes
- Technical Indicators selection
- Custom Feature Code editor
- Feature Analysis section

**Key Backend Connections:**
- FeatureEngineer and FeatureConfig
- FeatureRegistry integration
- Sandboxed code execution
- Hot/cold path optimization

---

### 6. Strategy Builder (62KB)
**File:** `PLAN_strategy_builder.md`
**UI Components:**
- Strategy Builder form
- Risk parameter fields
- Strategy Logic code editor
- Validate/Backtest/Deploy buttons
- Performance chart

**Key Backend Connections:**
- MLTradingStrategy compilation
- Nautilus BacktestEngine
- Secure code validation
- Multi-stage deployment pipeline

---

### 7. API Explorer (37KB)
**File:** `PLAN_api_explorer.md`
**UI Components:**
- Endpoint documentation sections
- API Tester with method selection
- Request/Response editors
- Interactive testing interface

**Key Backend Connections:**
- OpenAPI/Swagger generation
- Request validation and auth
- Rate limiting and logging
- API versioning strategy

---

### 8. Terminal & Settings (36KB)
**File:** `PLAN_terminal_settings.md`
**UI Components:**
- Terminal command interface
- Auto-completion system
- Settings configuration sections
- Save/Apply/Reset controls

**Key Backend Connections:**
- Secure command execution
- Configuration management
- Real-time config propagation
- Multi-tier persistence

## Proposed Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2, TBD)
- Base service layer architecture
- WebSocket infrastructure
- Authentication and security framework
- Basic store integration

### Phase 2: Essential Services (Weeks 3-4, TBD)
- Trading controls and safety systems
- Actor management basics
- Metrics collection and monitoring
- API documentation framework

### Phase 3: Advanced Features (Weeks 5-6, TBD)
- Pipeline orchestration
- Feature engineering UI
- Strategy builder with validation
- Terminal command execution

### Phase 4: Production Readiness (Weeks 7-8, TBD)
- Performance optimization
- Comprehensive testing
- Security hardening
- Documentation and deployment

## Common Patterns Across All Services

### 1. Progressive Fallback Chains
```
PostgreSQL + Redis → PostgreSQL Only → DummyStore → Cached Data
```

### 2. Update Mechanisms
- Primary: WebSocket for real-time updates
- Fallback: Polling with configurable intervals
- Caching: Multi-tier (memory, Redis, database)

### 3. Security Layers
- Authentication: Token-based with expiration
- Authorization: Role-based access control
- Validation: Schema-based with sanitization
- Sandboxing: Restricted execution environments

### 4. Store Integration Pattern
All services follow the mandatory 4-Store + 4-Registry pattern:
- **Stores:** FeatureStore, ModelStore, StrategyStore, DataStore
- **Registries:** FeatureRegistry, ModelRegistry, StrategyRegistry, DataRegistry

### 5. Performance Requirements
- **Hot Path:** <5ms P99 latency, zero allocations
- **Cold Path:** Batch operations, heavy I/O allowed
- **Caching:** TTL-based with progressive fallback

## Next Steps

1. **Prioritization:** Review plans and prioritize based on user needs
2. **Service Creation:** Create service modules following the patterns
3. **API Implementation:** Build REST endpoints in `app.py`
4. **WebSocket Layer:** Implement real-time communication
5. **Testing:** Comprehensive integration tests
6. **Documentation:** API docs and user guides

## Success Metrics

- **Functional:** All UI elements connected to real backend
- **Performance:** Meet hot/cold path requirements
- **Reliability:** 99.9% uptime with graceful degradation
- **Security:** Pass security audit with no critical issues
- **User Experience:** <100ms UI response time

## Total Analysis: 251KB of Implementation Plans
Each plan provides production-ready specifications for building a robust, scalable dashboard that seamlessly integrates with the Nautilus Trader ML infrastructure while maintaining all architectural patterns and performance requirements.
