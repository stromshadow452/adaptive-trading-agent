from stub_module import StubModel
import joblib, traceback, os
def main():
    try:
        print("joblib version:", joblib.__version__)
        os.makedirs("models", exist_ok=True)
        m = StubModel()
        out = "models/fx_bin_19f_thresh55__pack_ok.joblib"
        joblib.dump(m, out)
        print("STUB_CREATED", out)
    except Exception as e:
        print("EXCEPTION creating stub:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
