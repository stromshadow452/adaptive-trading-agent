def test_deterministic():
    from src.decision_engine.finrl_adapter import finrl_signal
    class DummyBooster:
        def predict(self, X): return [0.8]  # already prob
    cand = {"features": {f"f{i}": 0.01*i for i in range(1,17)}}
    cfg = {"enable_finrl_adapter": True}
    s1 = finrl_signal(cand, DummyBooster(), cfg)
    s2 = finrl_signal(cand, DummyBooster(), cfg)
    assert s1 == s2

def test_margin_squash():
    from src.decision_engine.finrl_adapter import finrl_signal
    class DummyBooster:
        def predict(self, X): return [2.0]  # margin -> squash
    cand = {"features": {f"f{i}": 0.01*i for i in range(1,17)}}
    cfg = {"enable_finrl_adapter": True}
    s, c, m = finrl_signal(cand, DummyBooster(), cfg)
    assert s == "buy" and 0.99 < c <= 1.0

def test_presence_fallback():
    from src.decision_engine.finrl_adapter import finrl_signal
    cand = {"features": {}}
    s, c, m = finrl_signal(cand, object(), {"enable_finrl_adapter": True})
    assert s == "hold" and abs(c-0.55)<1e-9