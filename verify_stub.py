import joblib, traceback
def main():
    try:
        m = joblib.load("models/fx_bin_19f_thresh55__pack_ok.joblib")
        print("LOADED TYPE:", type(m))
        if isinstance(m, dict):
            print("LOADED dict keys:", list(m.keys()))
            mm = m.get("model", None)
            meta = m.get("meta", None)
            print("model attr:", getattr(mm, "__class__", None), "meta:", meta)
        else:
            print("meta attr:", getattr(m, "meta", None))
    except Exception as e:
        print("EXCEPTION loading stub:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
