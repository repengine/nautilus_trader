/**
 * Nautilus ML Dashboard API Client
 *
 * Centralized client for all 37 backend endpoints from 8 services.
 * Follows Universal ML Architecture Patterns and ML Coding Standards.
 *
 * Services:
 * - Metrics Service (Agent 1): 4 endpoints
 * - Trading Service (Agent 2): 4 endpoints
 * - API Explorer Service (Agent 3): 3 endpoints
 * - Terminal Service (Agent 4): 4 endpoints
 * - Actor Service (Agent 5): 6 endpoints
 * - Pipeline Service (Agent 6): 5 endpoints
 * - Feature Service (Agent 7): 4 endpoints
 * - Strategy Service (Agent 8): 7 endpoints
 */

class NautilusAPIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
        this.requestCount = 0;
        this.errorCount = 0;
    }

    /**
     * Base fetch wrapper with error handling and metrics
     * @param {string} endpoint - API endpoint path
     * @param {object} options - Fetch options
     * @returns {Promise<any>} - Parsed JSON response
     */
    async _request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const startTime = performance.now();
        this.requestCount++;

        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers,
                },
                ...options,
            });

            const duration = performance.now() - startTime;
            console.log(`[API] ${options.method || 'GET'} ${endpoint} - ${response.status} (${duration.toFixed(2)}ms)`);

            if (!response.ok) {
                this.errorCount++;
                const error = await response.json().catch(() => ({ error: response.statusText }));
                throw new APIError(
                    error.error || error.message || 'Request failed',
                    response.status,
                    endpoint
                );
            }

            return await response.json();
        } catch (error) {
            this.errorCount++;
            console.error(`[API ERROR] ${endpoint}:`, error);

            if (error instanceof APIError) {
                throw error;
            }

            throw new APIError(
                `Network error: ${error.message}`,
                0,
                endpoint
            );
        }
    }

    /**
     * GET request helper
     */
    async _get(endpoint) {
        return this._request(endpoint, { method: 'GET' });
    }

    /**
     * POST request helper
     */
    async _post(endpoint, data) {
        return this._request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    }

    /**
     * PUT request helper
     */
    async _put(endpoint, data) {
        return this._request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    }

    /**
     * DELETE request helper
     */
    async _delete(endpoint) {
        return this._request(endpoint, { method: 'DELETE' });
    }

    // ========================================================================
    // METRICS SERVICE (Agent 1) - 4 endpoints
    // ========================================================================

    /**
     * Get current metrics snapshot
     * GET /api/metrics/snapshot
     */
    async getMetricsSnapshot() {
        return this._get('/api/metrics/snapshot');
    }

    /**
     * Get portfolio performance metrics
     * GET /api/metrics/portfolio
     */
    async getPortfolioMetrics() {
        return this._get('/api/metrics/portfolio');
    }

    /**
     * Get data ingestion metrics
     * GET /api/metrics/ingestion
     */
    async getIngestionMetrics() {
        return this._get('/api/metrics/ingestion');
    }

    /**
     * Get ML experiments and training runs
     * GET /api/metrics/experiments
     */
    async getExperiments() {
        return this._get('/api/metrics/experiments');
    }

    // ========================================================================
    // TRADING SERVICE (Agent 2) - 4 endpoints
    // ========================================================================

    /**
     * Trigger emergency stop for all trading
     * POST /api/control/emergency/stop
     */
    async emergencyStop() {
        return this._post('/api/control/emergency/stop', {});
    }

    /**
     * Get system health status
     * GET /api/health/system
     */
    async getSystemHealth() {
        return this._get('/api/health/system');
    }

    /**
     * Toggle trading on/off
     * POST /api/trading/toggle
     */
    async toggleTrading(enable, safetyChecks = null) {
        const payload = { enable: Boolean(enable) };
        if (safetyChecks && typeof safetyChecks === 'object') {
            payload.safety_checks = safetyChecks;
        }
        return this._post('/api/trading/toggle', payload);
    }

    /**
     * Get market data status
     * GET /api/trading/market-data
     */
    async getMarketDataStatus() {
        return this._get('/api/trading/market-data');
    }

    // ========================================================================
    // API EXPLORER SERVICE (Agent 3) - 3 endpoints
    // ========================================================================

    /**
     * Get OpenAPI specification
     * GET /api/openapi.json
     */
    async getOpenAPISpec() {
        return this._get('/api/openapi.json');
    }

    /**
     * Get Swagger UI HTML (returns HTML string)
     * GET /api/docs
     */
    async getSwaggerUI() {
        const response = await fetch(`${this.baseURL}/api/docs`);
        return response.text();
    }

    /**
     * Test an API endpoint interactively
     * POST /api/explorer/test
     */
    async testEndpoint(method, endpoint, body = null) {
        return this._post('/api/explorer/test', {
            method,
            endpoint,
            body,
        });
    }

    // ========================================================================
    // TERMINAL SERVICE (Agent 4) - 4 endpoints
    // ========================================================================

    /**
     * Execute terminal command
     * POST /api/terminal/execute
     */
    async executeCommand(command) {
        return this._post('/api/terminal/execute', { command });
    }

    /**
     * Get command execution history
     * GET /api/terminal/history
     */
    async getCommandHistory(limit = 50) {
        return this._get(`/api/terminal/history?limit=${limit}`);
    }

    /**
     * Get dashboard settings
     * GET /api/settings
     */
    async getSettings(section = null) {
        const query = section ? `?section=${encodeURIComponent(section)}` : '';
        return this._get(`/api/settings${query}`);
    }

    /**
     * Update dashboard settings
     * POST /api/settings
     */
    async updateSettings(section, updates, validate = true) {
        return this._post('/api/settings', {
            section,
            updates,
            validate,
        });
    }

    // ========================================================================
    // ACTOR SERVICE (Agent 5) - 6 endpoints
    // ========================================================================

    /**
     * Deploy a new ML actor
     * POST /api/actors/deploy
     */
    async deployActor(config) {
        return this._post('/api/actors/deploy', config);
    }

    /**
     * Get health status of all actors
     * GET /api/actors/health
     */
    async getActorsHealth() {
        return this._get('/api/actors/health');
    }

    /**
     * Hot-reload actor without downtime
     * POST /api/actors/hot-reload
     */
    async hotReloadActor(actorId) {
        return this._post('/api/actors/hot-reload', { actor_id: actorId });
    }

    /**
     * Pause actor execution
     * POST /api/actors/pause
     */
    async pauseActor(actorId) {
        return this._post('/api/actors/pause', { actor_id: actorId });
    }

    /**
     * Resume paused actor
     * POST /api/actors/resume
     */
    async resumeActor(actorId) {
        return this._post('/api/actors/resume', { actor_id: actorId });
    }

    /**
     * Stop actor gracefully
     * POST /api/actors/stop
     */
    async stopActor(actorId) {
        return this._post('/api/actors/stop', { actor_id: actorId });
    }

    // ========================================================================
    // PIPELINE SERVICE (Agent 6) - 5 endpoints
    // ========================================================================

    /**
     * Build dataset from raw data
     * POST /api/pipeline/build-dataset
     */
    async buildDataset(config) {
        return this._post('/api/pipeline/build-dataset', config);
    }

    /**
     * Train ML model
     * POST /api/pipeline/train-model
     */
    async trainModel(config) {
        return this._post('/api/pipeline/train-model', config);
    }

    /**
     * Run hyperparameter optimization
     * POST /api/pipeline/run-hpo
     */
    async runHPO(config) {
        return this._post('/api/pipeline/run-hpo', config);
    }

    /**
     * Get all pipeline jobs
     * GET /api/pipeline/jobs
     */
    async getPipelineJobs(status = null) {
        const params = status ? `?status=${status}` : '';
        return this._get(`/api/pipeline/jobs${params}`);
    }

    /**
     * Get job progress
     * GET /api/pipeline/jobs/{job_id}/progress
     */
    async getJobProgress(jobId) {
        return this._get(`/api/pipeline/jobs/${jobId}/progress`);
    }

    /**
     * Cancel running job
     * POST /api/pipeline/jobs/{job_id}/cancel
     */
    async cancelJob(jobId) {
        return this._post(`/api/pipeline/jobs/${jobId}/cancel`, {});
    }

    // ========================================================================
    // FEATURE SERVICE (Agent 7) - 4 endpoints
    // ========================================================================

    /**
     * Get all feature manifests
     * GET /api/features/manifests
     */
    async getFeatureManifests() {
        return this._get('/api/features/manifests');
    }

    /**
     * Validate feature engineering code
     * POST /api/features/validate-code
     */
    async validateFeatureCode(code) {
        return this._post('/api/features/validate-code', { code });
    }

    /**
     * Analyze feature importance and correlations
     * POST /api/features/analyze
     */
    async analyzeFeatures(config) {
        return this._post('/api/features/analyze', config);
    }

    /**
     * Generate feature code from designer parameters
     * POST /api/features/designer/generate
     */
    async generateFeatures(params) {
        return this._post('/api/features/designer/generate', params);
    }

    // ========================================================================
    // STRATEGY SERVICE (Agent 8) - 7 endpoints
    // ========================================================================

    /**
     * Get all trading strategies
     * GET /api/strategies
     */
    async getStrategies() {
        return this._get('/api/strategies');
    }

    /**
     * Get strategy details by ID
     * GET /api/strategies/{strategy_id}
     */
    async getStrategyDetails(strategyId) {
        return this._get(`/api/strategies/${strategyId}`);
    }

    /**
     * Create new strategy
     * POST /api/strategies
     */
    async createStrategy(config) {
        return this._post('/api/strategies', config);
    }

    /**
     * Backtest strategy
     * POST /api/strategies/backtest
     */
    async backtestStrategy(config) {
        return this._post('/api/strategies/backtest', config);
    }

    /**
     * Optimize strategy parameters
     * POST /api/strategies/optimize
     */
    async optimizeStrategy(config) {
        return this._post('/api/strategies/optimize', config);
    }

    /**
     * Deploy strategy to live trading
     * POST /api/strategies/{strategy_id}/deploy
     */
    async deployStrategy(config) {
        return this._post('/api/strategies/deploy', config);
    }

    /**
     * Get strategy performance metrics
     * GET /api/strategies/{strategy_id}/metrics
     */
    async getStrategyMetrics(strategyId) {
        return this._get(`/api/strategies/${strategyId}/performance`);
    }

    // ========================================================================
    // OBSERVABILITY & REGISTRY ENDPOINTS
    // ========================================================================

    /**
     * Get observability summary
     * GET /api/observability/summary
     */
    async getObservabilitySummary() {
        return this._get('/api/observability/summary');
    }

    /**
     * Get store health status
     * GET /api/observability/stores
     */
    async getStoresHealth() {
        return this._get('/api/observability/stores');
    }

    /**
     * Get dataset registry
     * GET /api/registry/datasets
     */
    async getDatasets() {
        return this._get('/api/registry/datasets');
    }

    /**
     * Get feature registry
     * GET /api/registry/features
     */
    async getFeatureRegistry() {
        return this._get('/api/registry/features');
    }

    /**
     * Get model deployments
     * GET /api/registry/deployments
     */
    async getDeployments() {
        return this._get('/api/registry/deployments');
    }

    // ========================================================================
    // CLIENT METRICS
    // ========================================================================

    /**
     * Get client-side metrics
     */
    getClientMetrics() {
        return {
            total_requests: this.requestCount,
            total_errors: this.errorCount,
            error_rate: this.requestCount > 0
                ? (this.errorCount / this.requestCount * 100).toFixed(2) + '%'
                : '0%',
        };
    }

    /**
     * Reset client metrics
     */
    resetMetrics() {
        this.requestCount = 0;
        this.errorCount = 0;
    }
}

/**
 * Custom API Error class for better error handling
 */
class APIError extends Error {
    constructor(message, status, endpoint) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.endpoint = endpoint;
    }

    toString() {
        return `APIError[${this.status}] ${this.endpoint}: ${this.message}`;
    }
}

// ============================================================================
// GLOBAL INSTANCE
// ============================================================================

// Create global API client instance
const apiClient = new NautilusAPIClient();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NautilusAPIClient, APIError, apiClient };
}
