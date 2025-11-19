import joblib, traceback
class StubModel:
    def __init__(self):
        self.meta = {"timeframe": "M15"}
    def predict(self, X):
        return [0.80]
def main():
    try:
        print("joblib version:", joblib.__version__)
        m = StubModel()
        out = "models/fx_bin_19f_thresh55__pack_ok.joblib"
        joblib.dump(m, out)
        print("STUB_CREATED", out)
    except Exception as e:
        print("EXCEPTION creating stub:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
