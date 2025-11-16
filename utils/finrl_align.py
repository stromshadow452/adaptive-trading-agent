from typing import Dict, Any, List

# Replace with the EXACT order used in training/executor — do not rename keys.
# Frozen as of commit: <INSERT_COMMIT_HASH>
ORDER_16 = [
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
    "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16"
]

def align_features_16(candidate: Dict[str, Any]) -> List[float]:
    feats = candidate.get("features") or candidate
    out: List[float] = []
    for k in ORDER_16:
        v = feats.get(k, 0.0)
        out.append(float(v))
    return out