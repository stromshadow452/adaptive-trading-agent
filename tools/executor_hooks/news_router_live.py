from __future__ import annotations

import time
import re
from typing import Any, Dict, Callable, Optional, Tuple


_CACHE: Dict[str, Tuple[str, float]] = {}
_LAST_FETCH: Dict[str, float] = {}
_RATE_LIMIT_SECONDS = 15.0
_TTL_SECONDS = 120.0
_MAX_LEN = 280
_WS_RE = re.compile(r"\s+")


def _sanitize(text: str) -> str:
    t = text.strip()
    t = _WS_RE.sub(" ", t)
    # Remove control characters
    t = "".join(ch for ch in t if ch.isprintable())
    return t[:_MAX_LEN]


def _default_fetch(_symbol: str) -> str:
    # Deterministic fallback (shadow-safe, no network)
    table = {
        "EURUSD": "Live feed: euro edges higher amid cautious sentiment after data.",
        "USDJPY": "Live feed: dollar steady; policy expectations anchor risk appetite.",
    }
    return table.get(_symbol, "Live feed: markets steady with balanced risk sentiment.")


def news_router(
    md: Dict[str, Any],
    *,
    use_remote: bool = False,
    fetcher: Optional[Callable[[str], str]] = None,
    rate_limit_seconds: float = _RATE_LIMIT_SECONDS,
    ttl_seconds: float = _TTL_SECONDS,
) -> Any:
    """
    Live headlines router with in-memory cache and rate limit.
    - Shadow-safe by default (use_remote=False): uses deterministic fallback.
    - If use_remote=True, a custom `fetcher(symbol) -> str` can be supplied.
    - Returns sanitized, bounded-length text.
    """
    symbol = (md.get("symbol") or md.get("market") or "GENERIC")
    now = time.monotonic()

    # Serve from cache if fresh
    cached = _CACHE.get(symbol)
    if cached and (now - cached[1] <= ttl_seconds):
        text = _sanitize(cached[0])
        return {
            "text": text,
            "telemetry": {
                "source": "live",
                "cache_hit": True,
                "ttl_ms": int((ttl_seconds - (now - cached[1])) * 1000),
                "fetched_at": cached[1],
            },
        }

    # Rate limit remote calls per symbol
    last = _LAST_FETCH.get(symbol, 0.0)
    if now - last < rate_limit_seconds:
        # If rate-limited and we have stale cache, serve it; else deterministic default
        if cached:
            text = _sanitize(cached[0])
            return {
                "text": text,
                "telemetry": {
                    "source": "live",
                    "cache_hit": True,
                    "ttl_ms": int((ttl_seconds - (now - cached[1])) * 1000),
                    "fetched_at": cached[1],
                },
            }
        text = _sanitize(_default_fetch(symbol))
        return {"text": text, "telemetry": {"source": "live", "cache_hit": False, "ttl_ms": int(ttl_seconds * 1000), "fetched_at": now}}

    # Decide source
    if use_remote and fetcher is not None:
        try:
            text = str(fetcher(symbol) or "")
        except Exception:
            text = _default_fetch(symbol)
    else:
        text = _default_fetch(symbol)

    clean = _sanitize(text)
    _CACHE[symbol] = (clean, now)
    _LAST_FETCH[symbol] = now
    return {
        "text": clean,
        "telemetry": {"source": "live", "cache_hit": False, "ttl_ms": int(ttl_seconds * 1000), "fetched_at": now},
    }

