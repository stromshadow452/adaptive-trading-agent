import os
import json
import math
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("RiskEngine")

class RealTimeRiskEngine:
    def __init__(self, state_file="logs/risk_state.json", initial_equity=10000.0):
        self.state_file = state_file
        self.initial_equity = initial_equity
        self.state = {
            "total_pnl": 0.0,
            "last_50_trades": [],  # List of dicts with essential info
            "slippage_ema": {},
            "loss_streak": 0,
            "sym_loss_streak": {},
            "sym_cooldowns": {},
            "regime_cooldowns": {},
            "global_cooldown_until": None,
            "slippage_strikes": {},
            "trend_hold_times": [],
            "consecutive_trend_exceeds": 0
        }
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.state.update(data)
                logger.info(f"[RiskEngine] Loaded state from {self.state_file}")
            except Exception as e:
                logger.error(f"[RiskEngine] Failed to load state: {e}")

    def _save_state(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"[RiskEngine] Failed to save state: {e}")

    def _calculate_pf(self, trades: list) -> float:
        gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] <= 0))
        return gross_profit / max(gross_loss, 0.01)

    def _get_currency_exposure(self, symbol: str):
        symbol = symbol.upper().replace("/", "")
        if len(symbol) == 6:
            return symbol[:3], symbol[3:]
        return symbol, None

    def evaluate_pre_trade(self, signal: dict, open_positions: dict, current_time: datetime):
        """
        Evaluate all risk rules in order.
        Returns: (allow_trade: bool, size_multiplier: float, reason: str)
        """
        try:
            return self._evaluate_internal(signal, open_positions, current_time)
        except Exception as e:
            logger.error(f"[RiskEngine] Error in pre_trade evaluation: {e}")
            return True, 1.0, f"error_fallback: {e}"

    def _evaluate_internal(self, signal: dict, open_positions: dict, current_time: datetime):
        sym = signal.get("symbol", "")
        if not sym:
            return False, 0.0, "missing_symbol"

        # Initialize defaults
        size_mult = 1.0
        n_trades = len(self.state["last_50_trades"])

        # 1. Global Circuit Breaker
        # Calculate 24h PnL
        pnl_last_24h = 0.0
        for t in self.state["last_50_trades"]:
            try:
                close_time = datetime.fromisoformat(t["close_time"])
                if (current_time - close_time).total_seconds() <= 86400:
                    pnl_last_24h += t["pnl"]
            except:
                pass
        
        current_equity = self.initial_equity + self.state["total_pnl"]
        equity_24h_ago = current_equity - pnl_last_24h
        pnl_pct_24h = pnl_last_24h / max(equity_24h_ago, 1.0)
        
        if pnl_pct_24h < -0.035:
            return False, 0.0, f"circuit_breaker: 24h PnL={pnl_pct_24h*100:.2f}%"

        # 2. Loss Streak Protection (Soft Penalty)
        # Instead of a global 12h hard block which starves the agent,
        # we apply a 50% size penalty if the global loss streak is severe.
        if self.state["loss_streak"] >= 5 and n_trades >= 10:
            pf_10 = self._calculate_pf(self.state["last_50_trades"][-10:])
            if pf_10 < 0.8:
                size_mult *= 0.5

        # 3. Symbol Cool-down
        sym_cd = self.state["sym_cooldowns"].get(sym)
        if sym_cd:
            if current_time < datetime.fromisoformat(sym_cd):
                return False, 0.0, "symbol_cooldown"
            else:
                del self.state["sym_cooldowns"][sym]

        # 4. Correlation Control
        base, quote = self._get_currency_exposure(sym)
        corr_count = 0
        for osym in open_positions:
            obase, oquote = self._get_currency_exposure(osym)
            if base in (obase, oquote) or (quote and quote in (obase, oquote)):
                corr_count += 1
        
        if corr_count >= 3:
            size_mult *= 0.5
        elif corr_count == 2:
            size_mult *= 0.7
        elif corr_count == 1:
            size_mult *= 0.85

        # 5. Rolling PF Monitor (Grace period: 20 trades)
        if n_trades >= 30:
            pf_15 = self._calculate_pf(self.state["last_50_trades"][-15:])
            pf_30 = self._calculate_pf(self.state["last_50_trades"][-30:])
            if pf_15 < 0.8 and pf_30 < 1.0:
                size_mult *= 0.5

        # 6. Spread check removed — spread cost is already embedded in the
        #    PaperExecutor's bid/ask fill model. The hardcoded default spread
        #    value was causing hundreds of false blocks on low-ATR bars.

        # 7. Regime Failure Detection
        regime = signal.get("regime", "UNKNOWN").upper()
        if regime == "TREND":
            regime_cd = self.state["regime_cooldowns"].get("TREND")
            if regime_cd:
                if current_time < datetime.fromisoformat(regime_cd):
                    return False, 0.0, "trend_regime_cooldown"
                else:
                    del self.state["regime_cooldowns"]["TREND"]
            
            exceeds = self.state.get("consecutive_trend_exceeds", 0)
            if exceeds >= 5:
                self.state["regime_cooldowns"]["TREND"] = (current_time + timedelta(hours=24)).isoformat()
                return False, 0.0, "trend_regime_failure_trigger"
            elif exceeds >= 3:
                size_mult *= 0.5

        # 8. Sizing Floor
        size_mult = max(size_mult, 0.5)

        return True, round(size_mult, 3), "ok"

    def update_post_trade(self, fill):
        """
        Takes a SimulatedFill object or dict equivalent and updates state.
        """
        try:
            self._update_internal(fill)
            self._save_state()
        except Exception as e:
            logger.error(f"[RiskEngine] Error in post_trade update: {e}")

    def _update_internal(self, fill):
        if not hasattr(fill, "symbol"):
            # If it's a dict
            sym = fill.get("symbol")
            pnl = fill.get("pnl_usd", 0.0)
            close_time = fill.get("ts") or fill.get("filled_at")
            slippage_usd = fill.get("slippage_usd", 0.0)
            hold_minutes = fill.get("hold_minutes", 0.0)
            regime = fill.get("regime", "UNKNOWN").upper()
            metadata = fill.get("metadata", {})
        else:
            sym = fill.symbol
            pnl = fill.pnl_usd or 0.0
            close_time = getattr(fill, "ts", getattr(fill, "filled_at", datetime.now(timezone.utc).isoformat()))
            slippage_usd = fill.slippage_usd or 0.0
            hold_minutes = fill.hold_minutes or 0.0
            regime = fill.regime.upper() if fill.regime else "UNKNOWN"
            metadata = fill.metadata or {}

        # Add to trades
        self.state["total_pnl"] += pnl
        self.state["last_50_trades"].append({
            "symbol": sym,
            "pnl": pnl,
            "close_time": close_time,
            "regime": regime
        })
        if len(self.state["last_50_trades"]) > 50:
            self.state["last_50_trades"].pop(0)

        n_trades = len(self.state["last_50_trades"])

        # Loss Streak updates
        if pnl < 0:
            self.state["loss_streak"] += 1
            self.state["sym_loss_streak"][sym] = self.state["sym_loss_streak"].get(sym, 0) + 1
            if self.state["sym_loss_streak"][sym] >= 3:
                try:
                    ct = datetime.fromisoformat(close_time)
                except:
                    ct = datetime.now(timezone.utc)
                self.state["sym_cooldowns"][sym] = (ct + timedelta(hours=24)).isoformat()
                self.state["sym_loss_streak"][sym] = 0
        else:
            self.state["loss_streak"] = 0
            self.state["sym_loss_streak"][sym] = 0

        # Slippage EMA (Grace period 20 trades)
        if n_trades >= 20:
            atr_pctile = metadata.get("atr_pctile", 0.0)
            # Ignore slippage spikes during high ATR regimes
            if atr_pctile < 0.8:
                prev_ema = self.state["slippage_ema"].get(sym)
                if prev_ema is None:
                    self.state["slippage_ema"][sym] = slippage_usd
                else:
                    if slippage_usd > 2.0 * prev_ema:
                        strikes = self.state["slippage_strikes"].get(sym, [])
                        strikes.append(close_time)
                        # Filter strikes within last 48h
                        valid_strikes = []
                        try:
                            ct = datetime.fromisoformat(close_time)
                            for s in strikes:
                                if (ct - datetime.fromisoformat(s)).total_seconds() <= 48 * 3600:
                                    valid_strikes.append(s)
                        except:
                            valid_strikes = strikes
                        self.state["slippage_strikes"][sym] = valid_strikes
                        
                        if len(valid_strikes) >= 3:
                            self.state["sym_cooldowns"][sym] = (ct + timedelta(hours=24)).isoformat()
                            self.state["slippage_strikes"][sym] = []
                    
                    # Update EMA
                    self.state["slippage_ema"][sym] = 0.8 * prev_ema + 0.2 * slippage_usd

        # Regime Failure Detection
        if regime == "TREND":
            self.state["trend_hold_times"].append(hold_minutes)
            if len(self.state["trend_hold_times"]) > 20:
                self.state["trend_hold_times"].pop(0)
            
            if len(self.state["trend_hold_times"]) >= 5:
                sorted_times = sorted(self.state["trend_hold_times"])
                median_hold = sorted_times[len(sorted_times) // 2]
                
                if hold_minutes > 3.0 * median_hold:
                    self.state["consecutive_trend_exceeds"] = self.state.get("consecutive_trend_exceeds", 0) + 1
                else:
                    self.state["consecutive_trend_exceeds"] = 0
