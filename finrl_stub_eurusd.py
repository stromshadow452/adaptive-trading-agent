class StubPolicy:
    def __init__(self): self.version="stub_v1"
    def act(self, features):
        return {"side":"BUY","er":0.0012,"conf":0.85,"costs":0.0001,"policy_id":self.version}
    def predict(self, X):
        return [1]
    def predict_proba(self, X):
        return [[0.1,0.9]]
