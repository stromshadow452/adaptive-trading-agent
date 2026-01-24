#!/bin/bash
# Usage: ./rollback_finrl.sh
set -e

echo "ROLLING BACK JARVIS PROTOCOL..."

# 1. Trip Global Circuit Breaker
echo "Tripping Global Circuit Breaker..."
python -c "from src.risk.circuit_breaker import CircuitBreaker; CircuitBreaker().trip(reason='Manual Rollback')"

# 2. Restore Previous Model
echo "Restoring Previous Model..."
# Logic to symlink 'champion_latest.json' to 'champion_prev.json' would go here

# 3. Reset Circuit Breaker (Manual confirmation needed usually, but for script we might reset if safe)
# echo "Resetting Circuit Breaker..."
# python -c "from src.risk.circuit_breaker import CircuitBreaker; cb=CircuitBreaker(); cb._save_state({'global_trip': False, 'symbols': {}})"

echo "Rollback Complete. System Halted."
