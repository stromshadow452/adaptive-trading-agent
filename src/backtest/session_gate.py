"""
MARK-3 SESSION GATE
===================

Session-aware trading module that controls when the agent is allowed to trade.

Sessions (UTC):
- SYDNEY:   21:00–23:00 (selective)
- TOKYO:    00:00–03:00 (exploration)
- LONDON:   07:00–10:00 (full)
- NEW_YORK: 12:00–15:00 (full)
- OFF:      All other times (blocked)

Integration point: Meta-Gating stage (after EDGE_SCORE, before signal emission)
"""

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import yaml

logger = logging.getLogger(__name__)


# ============================================================================
# SESSION DEFINITIONS
# ============================================================================

@dataclass
class SessionWindow:
    """Defines a trading session time window in UTC."""
    name: str
    start_hour: int
    end_hour: int
    policy: str  # FULL, EXPLORATION, SELECTIVE, BLOCK
    size_multiplier: float
    min_edge_score: float
    min_regime_strength: float
    description: str = ""


# Default session windows (UTC)
DEFAULT_SESSIONS = {
    "SYDNEY": SessionWindow(
        name="SYDNEY",
        start_hour=21,
        end_hour=23,
        policy="SELECTIVE",
        size_multiplier=0.85,
        min_edge_score=0.70,
        min_regime_strength=0.75,
        description="Sydney session - very selective"
    ),
    "TOKYO": SessionWindow(
        name="TOKYO",
        start_hour=0,
        end_hour=3,
        policy="EXPLORATION",
        size_multiplier=0.75,
        min_edge_score=0.50,
        min_regime_strength=0.50,
        description="Tokyo session - exploration mode"
    ),
    "LONDON": SessionWindow(
        name="LONDON",
        start_hour=7,
        end_hour=10,
        policy="FULL",
        size_multiplier=1.0,
        min_edge_score=0.40,
        min_regime_strength=0.40,
        description="London session - full trading"
    ),
    "NEW_YORK": SessionWindow(
        name="NEW_YORK",
        start_hour=12,
        end_hour=15,
        policy="FULL",
        size_multiplier=1.0,
        min_edge_score=0.40,
        min_regime_strength=0.40,
        description="New York session - full trading"
    ),
}


# ============================================================================
# SESSION DETECTION
# ============================================================================

def detect_session(timestamp: datetime) -> str:
    """
    Detect the current trading session from a UTC timestamp.
    
    Args:
        timestamp: Candle timestamp (must be timezone-aware or assumed UTC)
        
    Returns:
        Session name: SYDNEY, TOKYO, LONDON, NEW_YORK, or OFF
    """
    # Extract hour (UTC)
    if hasattr(timestamp, 'hour'):
        hour = timestamp.hour
    else:
        # Handle pandas Timestamp
        hour = timestamp.hour if hasattr(timestamp, 'hour') else 0
    
    # Check each session window
    # SYDNEY: 21:00–23:00
    if 21 <= hour < 23:
        return "SYDNEY"
    
    # TOKYO: 00:00–03:00
    if 0 <= hour < 3:
        return "TOKYO"
    
    # LONDON: 07:00–10:00
    if 7 <= hour < 10:
        return "LONDON"
    
    # NEW_YORK: 12:00–15:00
    if 12 <= hour < 15:
        return "NEW_YORK"
    
    # All other times
    return "OFF"


# ============================================================================
# SESSION POLICY OUTPUT
# ============================================================================

@dataclass
class SessionGateOutput:
    """Output from session gate evaluation."""
    session: str
    policy: str
    allowed: bool
    size_multiplier: float
    reason: str
    
    def __str__(self):
        status = "ALLOW" if self.allowed else "BLOCK"
        return f"{self.session} → {status} ({self.reason})"


# ============================================================================
# SESSION GATE MODULE
# ============================================================================

class SessionGate:
    """
    Session-aware trading gate.
    
    Controls trading based on market session and quality filters.
    Integrates at Meta-Gating stage after EDGE_SCORE.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize session gate.
        
        Args:
            config_path: Path to session.yaml config (optional)
        """
        self.enabled = True
        self.sessions = DEFAULT_SESSIONS.copy()
        self.log_allows = True
        self.log_blocks = True
        
        # Load config if provided
        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            # Try default path
            default_path = Path("config/session.yaml")
            if default_path.exists():
                self._load_config(default_path)
    
    def _load_config(self, config_path: Path):
        """Load session configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.enabled = config.get('enabled', True)
            
            # Load session definitions
            for name, session_cfg in config.get('sessions', {}).items():
                if name != "OFF" and 'start_hour' in session_cfg:
                    self.sessions[name] = SessionWindow(
                        name=name,
                        start_hour=session_cfg.get('start_hour', 0),
                        end_hour=session_cfg.get('end_hour', 0),
                        policy=session_cfg.get('policy', 'BLOCK'),
                        size_multiplier=session_cfg.get('size_multiplier', 1.0),
                        min_edge_score=session_cfg.get('min_edge_score', 0.40),
                        min_regime_strength=session_cfg.get('min_regime_strength', 0.40),
                        description=session_cfg.get('description', '')
                    )
            
            # Logging settings
            log_cfg = config.get('logging', {})
            self.log_allows = log_cfg.get('log_allows', True)
            self.log_blocks = log_cfg.get('log_blocks', True)
            
            logger.info(f"SessionGate loaded from {config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load session config: {e}, using defaults")
    
    def evaluate(
        self,
        timestamp: datetime,
        symbol: str,
        edge_score: float = 0.0,
        regime_strength: float = 0.0,
        confidence: float = 0.0,
    ) -> SessionGateOutput:
        """
        Evaluate session gate for a trade signal.
        
        Args:
            timestamp: Candle timestamp (UTC)
            symbol: Trading symbol
            edge_score: EDGE_SCORE from MARK-3 (0.0-1.0)
            regime_strength: Regime strength from MARK-2 (0.0-1.0)
            
        Returns:
            SessionGateOutput with decision and metadata
        """
        # Bypass if disabled
        if not self.enabled:
            return SessionGateOutput(
                session="DISABLED",
                policy="BYPASS",
                allowed=True,
                size_multiplier=1.0,
                reason="Session gate disabled"
            )
        
        # Detect current session
        session = detect_session(timestamp)
        
        # OFF session - allow only exceptional quality setups
        if session == "OFF":
            if confidence > 0.65 and edge_score > 0.65:
                output = SessionGateOutput(
                    session=session,
                    policy="OFF_OVERRIDE",
                    allowed=True,
                    size_multiplier=1.0,
                    reason=f"Off-session override (conf={confidence:.2f}, edge={edge_score:.2f})"
                )
                if self.log_allows:
                    logger.info(f"[SESSION] {symbol} {output}")
                return output

            output = SessionGateOutput(
                session=session,
                policy="BLOCK",
                allowed=False,
                size_multiplier=0.0,
                reason="Off-session (no trading window)"
            )
            if self.log_blocks:
                logger.info(f"[SESSION] {symbol} {output}")
            return output
        
        # Get session config
        session_cfg = self.sessions.get(session)
        if not session_cfg:
            return SessionGateOutput(
                session=session,
                policy="UNKNOWN",
                allowed=False,
                size_multiplier=0.0,
                reason=f"Unknown session: {session}"
            )
        
        # Apply session policy
        policy = session_cfg.policy
        
        if policy == "BLOCK":
            output = SessionGateOutput(
                session=session,
                policy=policy,
                allowed=False,
                size_multiplier=0.0,
                reason="Session policy is BLOCK"
            )
            if self.log_blocks:
                logger.info(f"[SESSION] {symbol} {output}")
            return output
        
        elif policy == "SELECTIVE":
            # Must meet minimum thresholds
            if edge_score < session_cfg.min_edge_score:
                output = SessionGateOutput(
                    session=session,
                    policy=policy,
                    allowed=False,
                    size_multiplier=0.0,
                    reason=f"EDGE {edge_score:.2f} < {session_cfg.min_edge_score:.2f}"
                )
                if self.log_blocks:
                    logger.info(f"[SESSION] {symbol} {output}")
                return output
            
            if regime_strength < session_cfg.min_regime_strength:
                output = SessionGateOutput(
                    session=session,
                    policy=policy,
                    allowed=False,
                    size_multiplier=0.0,
                    reason=f"Regime {regime_strength:.2f} < {session_cfg.min_regime_strength:.2f}"
                )
                if self.log_blocks:
                    logger.info(f"[SESSION] {symbol} {output}")
                return output
            
            # Passed selective filters
            output = SessionGateOutput(
                session=session,
                policy=policy,
                allowed=True,
                size_multiplier=session_cfg.size_multiplier,
                reason="Passed selective filters"
            )
            if self.log_allows:
                logger.info(f"[SESSION] {symbol} {output}")
            return output
        
        elif policy == "EXPLORATION":
            # Exploration mode - always allow but reduced size
            output = SessionGateOutput(
                session=session,
                policy=policy,
                allowed=True,
                size_multiplier=session_cfg.size_multiplier,
                reason="Exploration mode"
            )
            if self.log_allows:
                logger.info(f"[SESSION] {symbol} {output}")
            return output
        
        elif policy == "FULL":
            # Full trading - apply standard EDGE/MARK-2 rules
            output = SessionGateOutput(
                session=session,
                policy=policy,
                allowed=True,
                size_multiplier=session_cfg.size_multiplier,
                reason="Full trading session"
            )
            if self.log_allows:
                logger.info(f"[SESSION] {symbol} {output}")
            return output
        
        # Fallback - unknown policy, block
        return SessionGateOutput(
            session=session,
            policy=policy,
            allowed=False,
            size_multiplier=0.0,
            reason=f"Unknown policy: {policy}"
        )
    
    def get_session_info(self, timestamp: datetime) -> Dict[str, Any]:
        """Get session info for context enrichment."""
        session = detect_session(timestamp)
        session_cfg = self.sessions.get(session)
        
        return {
            "session": session,
            "policy": session_cfg.policy if session_cfg else "OFF",
            "size_multiplier": session_cfg.size_multiplier if session_cfg else 0.0,
            "description": session_cfg.description if session_cfg else "Off-session"
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_session_to_sizing(
    base_size: float,
    session_output: SessionGateOutput,
) -> Tuple[float, Dict[str, Any]]:
    """
    Apply session gate output to position sizing.
    
    Args:
        base_size: Size from EDGE + MARK-2 pipeline
        session_output: Output from session gate evaluation
        
    Returns:
        (adjusted_size, info_dict)
    """
    if not session_output.allowed:
        return 0.0, {
            'blocked_by_session': True,
            'session': session_output.session,
            'reason': session_output.reason
        }
    
    adjusted_size = base_size * session_output.size_multiplier
    
    return adjusted_size, {
        'blocked_by_session': False,
        'session': session_output.session,
        'session_multiplier': session_output.size_multiplier,
        'size_before_session': base_size,
        'size_after_session': adjusted_size
    }
