# Pipeline Orchestration Implementation Plan

## Overview

This document outlines the implementation plan for the dashboard's "⚙️ Pipelines" tab orchestration system, integrating with the existing `MLPipelineOrchestrator` and providing real-time pipeline management capabilities through the dashboard UI.

## Current Dashboard Pipeline UI Analysis

### 1. Dataset Building Section
**UI Elements:**
- Symbols input field (`SPY,QQQ,IWM`)
- Date range inputs (start/end dates)
- "Build Dataset" button

**Mapping to Orchestrator:**
- Maps to `MLPipelineOrchestrator.build_dataset()` method
- Uses `DatasetBuildConfig` for configuration
- Triggers data ingestion and feature engineering pipeline

### 2. Model Training Section
**UI Elements:**
- Model type dropdown (Teacher/Student/Ensemble)
- Algorithm selection (Transformer/XGBoost/LSTM/CNN)
- "Start Training" button

**Mapping to Orchestrator:**
- Maps to `MLPipelineOrchestrator.train_teacher()` and distillation methods
- Uses existing CLI wrappers (`teacher_main`, `hpo_main`)
- Integrates with model registry for artifact management

### 3. Hyperparameter Tuning Section
**UI Elements:**
- Search method dropdown (Optuna/Grid/Random/Bayesian)
- Trials input field
- "Optimize" button

**Mapping to Orchestrator:**
- Maps to `MLPipelineOrchestrator.run_hpo()` method
- Configures HPO through existing CLI integration
- Tracks optimization progress and best parameters

### 4. Scheduled Pipelines Table
**UI Elements:**
- Pipeline name, schedule, last/next run, status columns
- Shows recurring tasks (Daily Feature Engineering, Weekly Model Retraining)

**Current Implementation:**
- Static table with hardcoded entries
- No actual scheduler integration

### 5. Pipeline Progress Monitoring
**UI Elements:**
- Real-time status updates
- Progress indicators
- Terminal-style output display

**Current Implementation:**
- Basic status display
- No real progress tracking

## Implementation Architecture

### 1. Job Queue Management Strategy

#### Queue System Design
```python
@dataclass(slots=True)
class PipelineJob:
    job_id: str
    job_type: PipelineJobType  # DATASET_BUILD, MODEL_TRAIN, HPO, SCHEDULED
    status: JobStatus  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    config: dict[str, Any]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: float = 0.0
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

#### Queue Backend Options
1. **In-Memory Queue (Development)**
   - Use `asyncio.Queue` for simple implementation
   - Jobs persist only during service lifetime
   - Suitable for single-instance deployments

2. **PostgreSQL-Based Queue (Production)**
   - Leverage existing database infrastructure
   - Persistent job storage with ACID guarantees
   - Support for distributed deployments
   - Job history and audit trail

3. **Redis Queue (Scalable)**
   - High-performance distributed queue
   - Built-in pub/sub for real-time updates
   - Horizontal scaling capabilities

#### Recommended Implementation
Start with PostgreSQL-based queue to align with existing ML infrastructure:

```python
class PipelineJobQueue:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._running_jobs: dict[str, asyncio.Task] = {}

    async def enqueue(self, job: PipelineJob) -> str:
        """Add job to queue and return job_id."""

    async def dequeue(self) -> PipelineJob | None:
        """Get next pending job."""

    async def get_job_status(self, job_id: str) -> PipelineJob | None:
        """Get current job status."""

    async def update_progress(self, job_id: str, progress: float) -> None:
        """Update job progress."""
```

### 2. Async Execution Strategy

#### Worker Pool Architecture
```python
class PipelineWorkerPool:
    def __init__(
        self,
        orchestrator: MLPipelineOrchestrator,
        max_concurrent_jobs: int = 3,
        max_concurrent_per_type: dict[str, int] | None = None,
    ):
        self.orchestrator = orchestrator
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_concurrent_per_type = max_concurrent_per_type or {
            "DATASET_BUILD": 2,
            "MODEL_TRAIN": 1,  # GPU constraint
            "HPO": 1,  # Resource intensive
            "SCHEDULED": 3,
        }
        self._workers: list[asyncio.Task] = []
        self._job_queue = PipelineJobQueue(...)
```

#### Job Execution Strategy
1. **Resource-Aware Scheduling**
   - Limit concurrent GPU-intensive jobs (training/HPO)
   - Allow multiple I/O-bound jobs (dataset building)
   - Configurable resource constraints per job type

2. **Progress Tracking Integration**
   - Wrap existing CLI tools with progress callbacks
   - Stream output to job metadata
   - Real-time status updates via WebSocket

3. **Error Handling & Retry Logic**
   - Exponential backoff for transient failures
   - Job-specific retry policies
   - Dead letter queue for failed jobs

### 3. Pipeline Orchestrator Service Extension

#### Service Class Enhancement
```python
class PipelineOrchestrationService:
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.orchestrator = self._init_orchestrator()
        self.job_queue = PipelineJobQueue(...)
        self.worker_pool = PipelineWorkerPool(self.orchestrator)
        self.scheduler = PipelineScheduler(...)

    async def start_dataset_build(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> str:
        """Start dataset building job."""

    async def start_model_training(
        self,
        model_type: str,
        algorithm: str,
        dataset_id: str,
        **kwargs
    ) -> str:
        """Start model training job."""

    async def start_hpo(
        self,
        search_method: str,
        trials: int,
        model_config: dict[str, Any],
        **kwargs
    ) -> str:
        """Start hyperparameter optimization job."""

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get job status and progress."""

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel running job."""
```

### 4. Real-Time Progress Tracking

#### WebSocket Integration
```python
class PipelineProgressWebSocket:
    def __init__(self, job_queue: PipelineJobQueue):
        self.job_queue = job_queue
        self.connections: set[WebSocket] = set()

    async def broadcast_job_update(self, job_id: str, update: dict[str, Any]):
        """Broadcast job updates to all connected clients."""

    async def handle_connection(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
```

#### Progress Callback System
```python
class ProgressCallback:
    def __init__(self, job_id: str, job_queue: PipelineJobQueue):
        self.job_id = job_id
        self.job_queue = job_queue

    async def update_progress(
        self,
        progress: float,
        stage: str,
        message: str | None = None
    ):
        """Update job progress and broadcast to clients."""
        await self.job_queue.update_progress(
            self.job_id,
            progress,
            {"stage": stage, "message": message}
        )
```

### 5. Scheduled Pipeline Management

#### Scheduler Integration
```python
class PipelineScheduler:
    def __init__(
        self,
        orchestration_service: PipelineOrchestrationService,
        config_loader: Any,
    ):
        self.orchestration_service = orchestration_service
        self.config_loader = config_loader
        self._scheduled_jobs: dict[str, dict[str, Any]] = {}

    async def register_scheduled_pipeline(
        self,
        pipeline_id: str,
        schedule: str,  # Cron expression
        pipeline_config: dict[str, Any],
    ) -> bool:
        """Register new scheduled pipeline."""

    async def unregister_scheduled_pipeline(self, pipeline_id: str) -> bool:
        """Remove scheduled pipeline."""

    async def get_scheduled_pipelines(self) -> list[dict[str, Any]]:
        """Get all scheduled pipelines with next run times."""

    async def run_scheduled_pipelines(self):
        """Background task to execute scheduled pipelines."""
```

### 6. Resource Management Strategy

#### Database Schema Extensions
```sql
-- Pipeline Jobs Table
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    job_id VARCHAR PRIMARY KEY,
    job_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    config JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    progress FLOAT DEFAULT 0.0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',

    INDEX idx_pipeline_jobs_status (status),
    INDEX idx_pipeline_jobs_type_status (job_type, status),
    INDEX idx_pipeline_jobs_created (created_at)
);

-- Scheduled Pipelines Table
CREATE TABLE IF NOT EXISTS scheduled_pipelines (
    pipeline_id VARCHAR PRIMARY KEY,
    pipeline_name VARCHAR NOT NULL,
    schedule_cron VARCHAR NOT NULL,
    pipeline_config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

#### Memory Management
- Set process limits for worker processes
- Monitor memory usage of long-running jobs
- Implement job cancellation for resource protection

#### Storage Management
- Clean up temporary files after job completion
- Implement retention policies for job history
- Archive old job logs and artifacts

### 7. Integration with Existing ML Pipeline

#### MLPipelineOrchestrator Integration Points
1. **Async Wrapper Layer**
   ```python
   class AsyncMLPipelineOrchestrator:
       def __init__(self, orchestrator: MLPipelineOrchestrator):
           self.orchestrator = orchestrator

       async def build_dataset_async(
           self,
           config: DatasetBuildConfig,
           progress_callback: ProgressCallback,
       ) -> BuildArtifacts:
           """Async wrapper for dataset building."""
           loop = asyncio.get_event_loop()
           return await loop.run_in_executor(
               None,
               self._build_dataset_with_progress,
               config,
               progress_callback,
           )
   ```

2. **CLI Integration**
   - Wrap existing CLI commands with async execution
   - Capture stdout/stderr for progress tracking
   - Handle CLI exit codes and error conditions

3. **Registry Integration**
   - Automatic artifact registration after job completion
   - Metadata propagation from jobs to registry
   - Version tracking for pipeline runs

### 8. API Endpoint Extensions

#### New Dashboard API Endpoints
```python
# Add to ml/dashboard/app.py

@app.post("/api/pipeline/jobs")
async def create_pipeline_job() -> tuple[Any, int]:
    """Create new pipeline job."""

@app.get("/api/pipeline/jobs")
async def list_pipeline_jobs() -> tuple[Any, int]:
    """List pipeline jobs with filtering."""

@app.get("/api/pipeline/jobs/<job_id>")
async def get_pipeline_job(job_id: str) -> tuple[Any, int]:
    """Get specific job details."""

@app.delete("/api/pipeline/jobs/<job_id>")
async def cancel_pipeline_job(job_id: str) -> tuple[Any, int]:
    """Cancel running job."""

@app.post("/api/pipeline/schedule")
async def create_scheduled_pipeline() -> tuple[Any, int]:
    """Create scheduled pipeline."""

@app.get("/api/pipeline/schedule")
async def list_scheduled_pipelines() -> tuple[Any, int]:
    """List scheduled pipelines."""

@app.websocket("/ws/pipeline/progress")
async def pipeline_progress_websocket():
    """WebSocket endpoint for real-time progress updates."""
```

### 9. Implementation Phases

#### Phase 1: Core Infrastructure (2-3 weeks)
- [ ] Implement `PipelineJob` and `PipelineJobQueue` classes
- [ ] Create database schema for jobs and schedules
- [ ] Basic async wrapper for `MLPipelineOrchestrator`
- [ ] Simple job execution without progress tracking

#### Phase 2: Progress Tracking (1-2 weeks)
- [ ] Implement progress callback system
- [ ] Add WebSocket endpoints for real-time updates
- [ ] Enhance UI with live progress indicators
- [ ] Stream job logs to dashboard

#### Phase 3: Advanced Features (2-3 weeks)
- [ ] Implement resource-aware scheduling
- [ ] Add job retry and error handling
- [ ] Create scheduled pipeline management
- [ ] Enhance UI with job history and management

#### Phase 4: Production Features (1-2 weeks)
- [ ] Add comprehensive monitoring and metrics
- [ ] Implement job persistence and recovery
- [ ] Add performance optimization and caching
- [ ] Complete testing and documentation

### 10. Configuration Management

#### Environment Variables
```bash
# Pipeline orchestration settings
PIPELINE_MAX_CONCURRENT_JOBS=3
PIPELINE_MAX_DATASET_JOBS=2
PIPELINE_MAX_TRAINING_JOBS=1
PIPELINE_JOB_TIMEOUT_MINUTES=120
PIPELINE_CLEANUP_RETENTION_DAYS=30

# Queue settings
PIPELINE_QUEUE_BACKEND=postgresql  # postgresql|redis|memory
PIPELINE_QUEUE_POLL_INTERVAL_SECONDS=5

# Resource limits
PIPELINE_MAX_MEMORY_GB=16
PIPELINE_MAX_DISK_TEMP_GB=100
```

#### Dashboard Configuration Extension
```python
@dataclass(slots=True, frozen=True)
class PipelineOrchestrationConfig:
    max_concurrent_jobs: int = 3
    max_concurrent_per_type: dict[str, int] = field(default_factory=dict)
    job_timeout_minutes: int = 120
    queue_backend: str = "postgresql"
    queue_poll_interval_seconds: float = 5.0
    cleanup_retention_days: int = 30
    enable_progress_websocket: bool = True
    enable_scheduled_pipelines: bool = True
```

## Risk Assessment & Mitigation

### 1. Resource Exhaustion
**Risk:** Long-running jobs consuming excessive memory/disk/CPU
**Mitigation:**
- Implement resource monitoring and limits
- Job timeout mechanisms
- Graceful degradation under resource pressure

### 2. Database Contention
**Risk:** High frequency job status updates causing database performance issues
**Mitigation:**
- Batch status updates where possible
- Use connection pooling
- Consider Redis for high-frequency updates

### 3. Job Failure Recovery
**Risk:** System restart causing loss of running jobs
**Mitigation:**
- Persistent job state in database
- Job recovery on service startup
- Idempotent job execution where possible

### 4. UI Responsiveness
**Risk:** Long polling or heavy updates causing UI to become unresponsive
**Mitigation:**
- WebSocket-based real-time updates
- Efficient data serialization
- Client-side state management

## Success Metrics

### 1. Performance Metrics
- Job queue throughput (jobs/minute)
- Average job completion time by type
- Resource utilization efficiency
- UI responsiveness (< 100ms for status updates)

### 2. Reliability Metrics
- Job success rate (> 95%)
- Service uptime (> 99.9%)
- Recovery time after failures (< 5 minutes)
- Data consistency (zero data loss)

### 3. User Experience Metrics
- Time to start job (< 2 seconds)
- Progress update frequency (< 10 seconds)
- UI error rate (< 1%)
- Feature adoption rate

## Conclusion

This implementation plan provides a comprehensive approach to integrating pipeline orchestration with the Nautilus Trader ML dashboard. The design leverages existing infrastructure while adding robust async execution, progress tracking, and resource management capabilities.

The phased approach allows for incremental delivery and testing, ensuring that each component is stable before building upon it. The focus on PostgreSQL integration maintains consistency with the existing ML stack while providing the scalability needed for production deployments.