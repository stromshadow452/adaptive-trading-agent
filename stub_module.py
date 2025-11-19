class StubModel:
    def __init__(self):
        self.meta = {"timeframe": "M15"}
    def predict(self, X):
        return [0.80]
