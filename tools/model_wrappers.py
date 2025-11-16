class PrimaryWithMeta:
    def __init__(self, inner):
        self.inner = inner
        self.meta = {'name':'fx_bin_19f_thresh55','timeframe':'M15','features':19}
    def predict(self, X): return self.inner.predict(X)
    def predict_proba(self, X):
        return self.inner.predict_proba(X) if hasattr(self.inner,'predict_proba') else self.inner.predict(X)
