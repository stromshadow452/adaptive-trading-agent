class StubPolicyG:
    def __init__(self): self.version="stub_gbp_v1"
    def act(self, features): return {"side":"SELL","er":0.0009,"conf":0.80,"costs":0.0001,"policy_id":self.version}
    def predict(self,X): return [1]
    def predict_proba(self,X): return [[0.2,0.8]]
