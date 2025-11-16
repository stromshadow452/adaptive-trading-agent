import joblib
class WrappedPolicy:
    """
    Top-level wrapper so joblib can pickle safely.
    Ensures .timeframe and .meta exist and proxies act()/predict()/predict_proba()
    """
    def __init__(self, model):
        self.model = model
        self.timeframe = "M15"
        self.meta = {"timeframe":"M15","version":"wrapped_ppo_v1"}
    def act(self, features):
        return getattr(self.model, "act", lambda f: None)(features)
    def predict(self, X):
        return getattr(self.model, "predict", lambda X: None)(X)
    def predict_proba(self, X):
        return getattr(self.model, "predict_proba", lambda X: None)(X)
