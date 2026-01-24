import lightgbm as lgb
import optuna
import joblib
import json
import hashlib
import hmac
import os
from datetime import datetime

MODEL_HMAC_KEY = os.environ.get("MODEL_HMAC_KEY", "dev_key_do_not_use_in_prod").encode()

class ModelRegistry:
    """
    Stage 3: Primary ML Brain -> Adaptive LightGBM Registry
    Manages atomic saves, integrity verification, and Optuna tuning.
    """
    def __init__(self, registry_path="models/registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)

    def tune_and_train(self, X, y, feature_hash, trials=20):
        def objective(trial):
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "learning_rate": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "feature_fraction": trial.suggest_float("ff", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bf", 0.5, 1.0),
                "bagging_freq": 1,
                "verbosity": -1
            }
            dtrain = lgb.Dataset(X, label=y)
            res = lgb.cv(params, dtrain, nfold=3, num_boost_round=100, 
                         callbacks=[lgb.early_stopping(10)])
            return res['valid binary_logloss-mean'][-1]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)
        
        best_params = study.best_params
        best_params.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1})
        
        model = lgb.train(best_params, lgb.Dataset(X, label=y), num_boost_round=500)
        
        meta = {
            "best_params": best_params,
            "best_score": study.best_value,
            "feature_hash": feature_hash,
            "trained_at": datetime.utcnow().isoformat()
        }
        return model, meta

    def save_model(self, model, meta: dict, tag="challenger"):
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        fname = f"{tag}_{timestamp}.joblib"
        path = os.path.join(self.registry_path, fname)
        
        # 1. Serialize
        payload = {"model": model, "meta": meta}
        temp_path = path + ".tmp"
        joblib.dump(payload, temp_path)
        
        # 2. Compute Signatures (Detached)
        with open(temp_path, "rb") as f:
            data = f.read()
            sha256 = hashlib.sha256(data).hexdigest()
            sig = hmac.new(MODEL_HMAC_KEY, data, hashlib.sha256).hexdigest()
        
        # 3. Write Signature File
        with open(path + ".sig", "w") as f:
            f.write(json.dumps({"sha256": sha256, "hmac": sig}))
            
        # 4. Finalize
        if os.path.exists(path):
            os.remove(path)
        os.rename(temp_path, path)
        
        # Update Pointer
        with open(os.path.join(self.registry_path, f"{tag}_latest.json"), "w") as f:
            json.dump({"path": path, "sig_path": path + ".sig", "meta": meta}, f)
        
        return path

    def load_model(self, tag="champion"):
        pointer_path = os.path.join(self.registry_path, f"{tag}_latest.json")
        if not os.path.exists(pointer_path):
            raise FileNotFoundError(f"No model found for tag: {tag}")
            
        with open(pointer_path) as f:
            info = json.load(f)
            
        model_path = info["path"]
        sig_path = info["sig_path"]
        
        # Verify Integrity
        with open(model_path, "rb") as f:
            data = f.read() 
            
        with open(sig_path) as f:
            sigs = json.load(f)
            
        # 1. SHA256 Check
        curr_sha = hashlib.sha256(data).hexdigest()
        
        # 2. HMAC Check
        curr_hmac = hmac.new(MODEL_HMAC_KEY, data, hashlib.sha256).hexdigest()
        
        if curr_hmac != sigs["hmac"]:
            raise ValueError(f"SECURITY ALERT: Model HMAC mismatch! File may be tampered. Exp: {sigs['hmac']}, Got: {curr_hmac}")
            
        payload = joblib.load(model_path)
        return payload["model"], payload["meta"]
    
    def load_model_from_path(self, model_path: str):
        """
        Load model from arbitrary path (for RL adapter).
        
        Args:
            model_path: Full path to model file
            
        Returns:
            (model, meta) tuple
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        sig_path = model_path + ".sig"
        
        # Load model data
        with open(model_path, "rb") as f:
            data = f.read()
        
        # Load signature
        with open(sig_path) as f:
            sigs = json.load(f)
        
        # HMAC Check
        curr_hmac = hmac.new(MODEL_HMAC_KEY, data, hashlib.sha256).hexdigest()
        
        if curr_hmac != sigs["hmac"]:
            raise ValueError(f"SECURITY ALERT: Model HMAC mismatch! File may be tampered.")
        
        payload = joblib.load(model_path)
        return payload["model"], payload["meta"]
