class SklearnPolicyAdapter:
    """
    Minimal 'policy' wrapper exposing predict_conf(x_vec)->float in [0,1] for BUY.
    """
    def __init__(self, clf, proba_index=1, meta=None):
        self.clf = clf
        self.proba_index = proba_index
        self.algo = "adapter_sklearn"
        self.meta = meta or {}

    def predict_conf(self, x_vec):
        import numpy as np, math
        X = np.asarray(x_vec, dtype=float).reshape(1, -1)
        if hasattr(self.clf, "predict_proba"):
            p = float(self.clf.predict_proba(X)[0][self.proba_index])
        elif hasattr(self.clf, "decision_function"):
            d = float(self.clf.decision_function(X)[0])
            p = 1.0 / (1.0 + math.exp(-d))
        else:
            y = int(self.clf.predict(X)[0])
            p = 1.0 if y == 1 else 0.0
        return p
