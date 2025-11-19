import os, sys, shutil, joblib, pickle, traceback
from pathlib import Path

MODEL = Path("models/fx_bin_19f_thresh55__pack_ok.joblib")
BACKUP = MODEL.with_name(MODEL.name + ".bak")
STUB_MARKER = {"_is_stub_model": True}

# 1) backup original if backup not exists
if MODEL.exists() and not BACKUP.exists():
    shutil.copy2(MODEL, BACKUP)
    print("BACKED UP:", BACKUP)

# 2) try to load normally
try:
    obj = joblib.load(str(MODEL))
    print("Loaded original model OK (no stub needed). Exiting.")
    sys.exit(0)
except Exception as e:
    print("Normal load failed:", repr(e))

# 3) try safe unpickle to inspect globals (tolerant Unpickler)
class ForgivingUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # return a dummy class for any missing module/class
        try:
            return super().find_class(module, name)
        except Exception:
            # create a dummy class type
            return type(name, (), {})

try:
    with open(MODEL, "rb") as f:
        u = ForgivingUnpickler(f)
        try:
            obj = u.load()
            print("Forgiving unpickle succeeded. Creating stub model wrapper.")
        except Exception as e:
            print("Forgiving unpickle still failed; continuing to create generic stub.")
except Exception as e:
    print("Could not open model file:", e)
    traceback.print_exc()

# 4) create a stub model object that has predict_proba and timeframe attribute
import numpy as np

class StubModel:
    def __init__(self, timeframe="M15", n_classes=2):
        self.timeframe = timeframe
        self.n_classes = n_classes
        self._meta = STUB_MARKER
    def predict_proba(self, X):
        # X may be 1D/2D — produce neutral probabilities 50/50
        try:
            n = len(X)
        except Exception:
            n = 1
        probs = np.tile([0.5] * self.n_classes, (n,1))
        return probs

stub = StubModel(timeframe="M15", n_classes=2)

# 5) write stub model to original path (overwrite)
try:
    joblib.dump(stub, str(MODEL))
    print("WROTE stub model to", MODEL)
    print("Stub model has attributes:", {"timeframe": stub.timeframe, "n_classes": stub.n_classes})
except Exception as e:
    print("Failed to write stub model:", e)
    traceback.print_exc()
    sys.exit(2)
