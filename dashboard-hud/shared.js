const API_BASE = "http://127.0.0.1:8000";

async function apiGet(path) {
    const response = await fetch(`${API_BASE}${path}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

async function apiPost(path, body) {
    const response = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
}

async function checkConnection() {
    try {
        const response = await fetch(`${API_BASE}/backtests/history?limit=1`);
        return response.ok;
    } catch {
        return false;
    }
}

function formatDateTime(isoString) {
    return new Date(isoString).toLocaleString();
}

function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.style.display = 'flex';
}

function hideLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.style.display = 'none';
}

function showElement(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.style.display = 'block';
}

function hideElement(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.style.display = 'none';
}

// ============================================================================
// SSE Streaming Client
// ============================================================================

/**
 * StreamingBacktestClient - Real-time SSE streaming for backtests
 * 
 * Features:
 * - EventSource connection with auto-reconnect
 * - Heartbeat monitoring (10s interval)
 * - Live log rendering
 * - Progress bar updates
 * - Result handling
 * - Cleanup on completion/error
 * 
 * Usage:
 * ```javascript
 * const client = new StreamingBacktestClient(jobId, {
 *     onLog: (data) => console.log(data.message),
 *     onProgress: (data) => updateProgressBar(data.percent),
 *     onResult: (data) => showResults(data.metrics),
 *     onError: (data) => showError(data.message)
 * });
 * client.connect();
 * ```
 */
class StreamingBacktestClient {
    constructor(jobId, callbacks = {}) {
        this.jobId = jobId;
        this.callbacks = {
            onLog: callbacks.onLog || (() => { }),
            onProgress: callbacks.onProgress || (() => { }),
            onResult: callbacks.onResult || (() => { }),
            onError: callbacks.onError || (() => { }),
            onHeartbeat: callbacks.onHeartbeat || (() => { }),
            onConnected: callbacks.onConnected || (() => { }),
            onDisconnect: callbacks.onDisconnect || (() => { })
        };

        this.eventSource = null;
        this.retryCount = 0;
        this.maxRetries = 10;
        this.lastHeartbeat = Date.now();
        this.heartbeatCheckInterval = null;
        this.isComplete = false;
    }

    /**
     * Connect to SSE stream
     */
    connect() {
        if (this.isComplete) return;

        const url = `${API_BASE}/stream/backtest/${this.jobId}`;
        console.log(`[SSE] Connecting to ${url}`);

        try {
            this.eventSource = new EventSource(url);
            console.log(`[SSE] EventSource created, readyState: ${this.eventSource.readyState}`);
        } catch (err) {
            console.error(`[SSE] Failed to create EventSource:`, err);
            this.callbacks.onError({ message: `EventSource creation failed: ${err.message}` });
            return;
        }

        // Connection opened
        this.eventSource.onopen = () => {
            console.log(`[SSE] Connected to job ${this.jobId}, readyState: ${this.eventSource.readyState}`);
            this.retryCount = 0;
            this.startHeartbeatMonitor();
        };

        // Connection event
        this.eventSource.addEventListener('connected', (e) => {
            const data = JSON.parse(e.data);
            console.log(`[SSE] Confirmed connected:`, data);
            this.callbacks.onConnected(data);
        });

        // Log event
        this.eventSource.addEventListener('log', (e) => {
            const data = JSON.parse(e.data);
            this.callbacks.onLog(data);
        });

        // Progress event
        this.eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            this.callbacks.onProgress(data);
        });

        // Heartbeat event
        this.eventSource.addEventListener('heartbeat', (e) => {
            const data = JSON.parse(e.data);
            this.lastHeartbeat = Date.now();
            this.callbacks.onHeartbeat(data);
        });

        // Result event (final)
        this.eventSource.addEventListener('result', (e) => {
            const data = JSON.parse(e.data);
            console.log(`[SSE] Received result:`, data);
            this.isComplete = true;
            this.callbacks.onResult(data);
            this.disconnect();
        });

        // Error event (from server)
        this.eventSource.addEventListener('error', (e) => {
            // Check if it's a server-sent error event with data
            if (e.data) {
                const data = JSON.parse(e.data);
                console.error(`[SSE] Server error:`, data);
                this.isComplete = true;
                this.callbacks.onError(data);
                this.disconnect();
            }
        });

        // Connection error (network issue)
        this.eventSource.onerror = (e) => {
            console.error(`[SSE] Connection error`, e);

            if (this.isComplete) return;

            // Auto-reconnect with exponential backoff
            this.eventSource.close();
            this.retryCount++;

            if (this.retryCount <= this.maxRetries) {
                const delay = Math.min(1000 * Math.pow(2, this.retryCount), 30000);
                console.log(`[SSE] Reconnecting in ${delay}ms (attempt ${this.retryCount})`);
                setTimeout(() => this.connect(), delay);
            } else {
                console.error(`[SSE] Max retries exceeded`);
                this.callbacks.onError({ message: 'Connection lost. Max retries exceeded.' });
                this.callbacks.onDisconnect();
            }
        };
    }

    /**
     * Start heartbeat monitor - detects stale connections
     */
    startHeartbeatMonitor() {
        this.stopHeartbeatMonitor();

        // Check every 30s if we've received a heartbeat
        this.heartbeatCheckInterval = setInterval(() => {
            const timeSinceHeartbeat = Date.now() - this.lastHeartbeat;

            // If no heartbeat for 60s, reconnect
            if (timeSinceHeartbeat > 60000 && !this.isComplete) {
                console.warn(`[SSE] No heartbeat for ${timeSinceHeartbeat}ms, reconnecting...`);
                this.eventSource.close();
                this.connect();
            }
        }, 30000);
    }

    /**
     * Stop heartbeat monitor
     */
    stopHeartbeatMonitor() {
        if (this.heartbeatCheckInterval) {
            clearInterval(this.heartbeatCheckInterval);
            this.heartbeatCheckInterval = null;
        }
    }

    /**
     * Disconnect and cleanup
     */
    disconnect() {
        console.log(`[SSE] Disconnecting from job ${this.jobId}`);

        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        this.stopHeartbeatMonitor();
        this.callbacks.onDisconnect();
    }
}

/**
 * Submit streaming backtest and return client
 * 
 * @param {Object} config - Backtest configuration
 * @param {Object} callbacks - Event callbacks
 * @returns {Promise<StreamingBacktestClient>}
 */
async function submitStreamingBacktest(config, callbacks) {
    // Submit job to streaming endpoint
    const postStart = Date.now();
    const response = await apiPost('/jobs/submit/backtest/stream', {
        config: config.config || 'config/mvp_v1.yaml',
        symbols: [config.symbol],
        start: config.start,
        end: config.end,
        initial_capital: config.capital || 10000
    });
    console.log(`[DEBUG] POST took ${Date.now() - postStart}ms`);

    const { job_id, stream_url } = response;
    console.log(`[Backtest] Submitted job ${job_id}, stream: ${stream_url}`);

    // Create and connect client IMMEDIATELY
    console.log(`[DEBUG] Creating SSE client at ${new Date().toISOString()}`);
    const client = new StreamingBacktestClient(job_id, callbacks);

    // Connect synchronously - don't delay
    console.log(`[DEBUG] Connecting SSE now...`);
    client.connect();
    console.log(`[DEBUG] SSE connect() called at ${new Date().toISOString()}`);

    return client;
}

