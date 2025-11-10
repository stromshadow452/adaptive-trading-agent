# src/contracts.py
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, List, Dict, Any

Symbol = str
TF = Literal["M1","M5","M15","M30","H1","H4","Daily"]

class Candidate(BaseModel):
    symbol: Symbol
    tf: TF
    score: float
    price: Optional[float] = None
    atr14: Optional[float] = None
    rsi14: Optional[float] = None
    regime: Optional[Literal["trend","trending","meanrev","range","unknown"]] = "unknown"
    extras: Dict[str, Any] = {}

class CandidatesFile(BaseModel):
    candidates: List[Candidate]

class Plan(BaseModel):
    symbol: Symbol
    tf: TF
    enter: bool
    side: Literal["buy","sell","hold"]
    final_score: float
    price: Optional[float]
    atr: Optional[float]
    size: float
    sl: Optional[float]
    tp: Optional[float]
    sl_type: Literal["atr","pct","none"]
    meta_w: Optional[float] = None
    reason: Dict[str, Any] = {}

class ApprovedAggregate(BaseModel):
    approved: List[Plan]  # list of Plan objects

# Small helpers
def ensure_utf8_sig(path: str) -> str:
    return path  # symbolic; use open(..., encoding="utf-8-sig")
