# JARVIS Remediation Plan Template

**Incident:** [Describe Gate Tripped]
**Timestamp:** [ISO8601]
**Component:** [Stage Name]

## Immediate Actions
1.  [ ] **Trip Circuit Breaker:** Run `python -c "from src.risk.circuit_breaker import CircuitBreaker; CircuitBreaker().trip(symbol='AFFECTED_SYMBOL', reason='Incident')"`
2.  [ ] **Dump State:** Check `logs/crash_dump_*.json`
3.  [ ] **Revert Model:** If ML related, run `./tools/rollback_finrl.sh`

## Root Cause Analysis
*   [ ] **Feature Drift:** Check `logs/finrl/feature_stats.json` vs Training Distribution.
*   [ ] **Latency:** Check `logs/finrl/*_latency.jsonl` for spikes > 200ms.
*   [ ] **Integrity:** Verify `models/registry/*.sig` matches files.

## Resolution
*   [ ] Patch applied to `src/`
*   [ ] Unit tests passed: `pytest tests/unit/test_jarvis.py`
*   [ ] Circuit Breaker reset: Manually edit `config/circuit_breakers.json` to set `tripped: false`.
