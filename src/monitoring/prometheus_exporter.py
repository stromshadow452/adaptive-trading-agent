"""
src/monitoring/prometheus_exporter.py
=====================================
Prometheus metrics exporter for SCOPUS shadow trading monitoring.

Exposes key metrics for scraping by Prometheus at /metrics endpoint.
Integrates with the FastAPI app via prometheus_client.

Install:
    pip install prometheus-client

Wire to FastAPI in src/api/main.py:
    from prometheus_client import make_asgi_app
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
"""
from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Gauge, Counter, Histogram, REGISTRY
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("[PrometheusExporter] prometheus_client not installed — metrics disabled")


# ---------------------------------------------------------------------------
# Metric definitions  (only register if prometheus_client is available)
# ---------------------------------------------------------------------------

if _PROMETHEUS_AVAILABLE:
    PROFIT_FACTOR      = Gauge("scopus_profit_factor",
                               "Rolling profit factor (gross profit / gross loss)")
    WIN_RATE           = Gauge("scopus_win_rate",
                               "Rolling win rate (0.0–1.0)")
    MAX_DRAWDOWN_PCT   = Gauge("scopus_max_drawdown_pct",
                               "Maximum drawdown since shadow start (%)")
    SHARPE_RATIO       = Gauge("scopus_sharpe_ratio",
                               "Annualized Sharpe ratio")
    NET_PNL_USD        = Gauge("scopus_net_pnl_usd",
                               "Net cumulative PnL in USD")
    AVG_SLIPPAGE_PIPS  = Gauge("scopus_avg_slippage_pips",
                               "Average fill slippage in pips")
    P95_LATENCY_MS     = Gauge("scopus_p95_latency_ms",
                               "P95 signal-to-fill latency in milliseconds")
    SHADOW_DAYS        = Gauge("scopus_shadow_days",
                               "Days elapsed in shadow trading run")
    OPEN_POSITIONS     = Gauge("scopus_open_positions",
                               "Number of currently open simulated positions")
    EQUITY             = Gauge("scopus_equity_usd",
                               "Current shadow equity in USD")

    # ── Week 9: Drift & Retrain Gauges ────────────────────────────────────
    FEATURE_PSI_MAX    = Gauge("scopus_feature_psi_max",
                               "Highest PSI across all monitored features")
    FEATURE_PSI_WARN   = Gauge("scopus_feature_psi_warn_count",
                               "Number of features with PSI in [0.10, 0.25)")
    FEATURE_PSI_CRIT   = Gauge("scopus_feature_psi_crit_count",
                               "Number of features with PSI >= 0.25 (drift alert)")
    LAST_RETRAIN_EPOCH = Gauge("scopus_last_retrain_epoch",
                               "Unix timestamp of the last completed model retrain")
    # ──────────────────────────────────────────────────────────────────────

    TRADES_TOTAL       = Counter("scopus_trades_total",
                                 "Total number of completed trades",
                                 ["strategy", "regime"])
    CB_TRIPS_TOTAL     = Counter("scopus_cb_trips_total",
                                 "Circuit breaker trip events",
                                 ["symbol"])
    SIGNAL_ERRORS      = Counter("scopus_signal_errors_total",
                                 "Pipeline errors per bar")

    SLIPPAGE_HIST      = Histogram("scopus_slippage_pips_hist",
                                   "Slippage distribution in pips",
                                   buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    LATENCY_HIST       = Histogram("scopus_latency_ms_hist",
                                   "Signal latency distribution in ms",
                                   buckets=[10, 50, 100, 200, 500, 1000])



class PrometheusExporter:
    """
    Updates Prometheus Gauges and Counters from ShadowMetrics snapshots.

    Wire up in a background thread or FastAPI startup event:
        exporter = PrometheusExporter(fills_log="logs/shadow/fills.jsonl")
        asyncio.create_task(exporter.update_loop(interval_seconds=60))
    """

    def __init__(self, fills_log: str = "logs/shadow/fills.jsonl"):
        self.fills_log = fills_log
        self._available = _PROMETHEUS_AVAILABLE
        if not self._available:
            logger.warning("[PrometheusExporter] Prometheus unavailable — metrics not exported")

    def update(self, metrics_snapshot: dict, open_positions: int = 0, equity: float = 0.0):
        """
        Push a ShadowMetrics.compute() snapshot to Prometheus gauges.
        Call this once per period (e.g. minute / 5-min / bar).
        """
        if not self._available:
            return
        try:
            PROFIT_FACTOR.set(metrics_snapshot.get("profit_factor", 0.0))
            WIN_RATE.set(metrics_snapshot.get("win_rate", 0.0))
            MAX_DRAWDOWN_PCT.set(metrics_snapshot.get("max_drawdown_pct", 0.0))
            SHARPE_RATIO.set(metrics_snapshot.get("sharpe_annualized") or 0.0)
            NET_PNL_USD.set(metrics_snapshot.get("net_pnl_usd", 0.0))
            AVG_SLIPPAGE_PIPS.set(metrics_snapshot.get("avg_slippage_pips", 0.0))
            P95_LATENCY_MS.set(metrics_snapshot.get("p95_latency_ms", 0.0))
            SHADOW_DAYS.set(metrics_snapshot.get("shadow_days", 0.0))
            OPEN_POSITIONS.set(open_positions)
            EQUITY.set(equity)
            logger.debug("[PrometheusExporter] Metrics pushed to Prometheus")
        except Exception as e:
            logger.error(f"[PrometheusExporter] Failed to update metrics: {e}")

    def record_fill(self, strategy: str, regime: str,
                    slippage_pips: float, latency_ms: float):
        """Call this once per trade fill to update histograms and counters."""
        if not self._available:
            return
        try:
            TRADES_TOTAL.labels(strategy=strategy, regime=regime).inc()
            SLIPPAGE_HIST.observe(slippage_pips)
            LATENCY_HIST.observe(latency_ms)
        except Exception as e:
            logger.debug(f"[PrometheusExporter] record_fill error: {e}")

    def record_cb_trip(self, symbol: str):
        """Call when circuit breaker is tripped."""
        if not self._available:
            return
        try:
            CB_TRIPS_TOTAL.labels(symbol=symbol).inc()
            logger.warning(f"[PrometheusExporter] CB trip recorded for {symbol}")
        except Exception as e:
            logger.debug(f"[PrometheusExporter] record_cb_trip error: {e}")

    def record_error(self):
        """Call once per pipeline error/exception caught."""
        if not self._available:
            return
        try:
            SIGNAL_ERRORS.inc()
        except Exception:
            pass

    async def update_loop(self, interval_seconds: int = 60):
        """
        Async background task: refresh Prometheus metrics every N seconds.
        Start with: asyncio.create_task(exporter.update_loop())
        """
        import asyncio
        from src.monitoring.metrics import ShadowMetrics
        metrics = ShadowMetrics(self.fills_log)
        while True:
            try:
                snapshot = metrics.compute()
                self.update(snapshot)
            except Exception as e:
                logger.error(f"[PrometheusExporter] update_loop error: {e}")
            await asyncio.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Docker Compose stub for monitoring stack
# ---------------------------------------------------------------------------
DOCKER_COMPOSE_MONITORING = """
# monitoring/docker-compose.yml
# Run with: docker compose up -d
version: "3.8"
services:
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alert_rules.yml:/etc/prometheus/alert_rules.yml

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=scopus2026
    volumes:
      - grafana_data:/var/lib/grafana

  alertmanager:
    image: prom/alertmanager:latest
    ports: ["9093:9093"]
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  grafana_data:
"""
