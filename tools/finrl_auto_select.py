import json, argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO, A2C
from src.rl.finrl_forex_env import ForexEnv

MEM = Path("config/finrl_memory.json")
ALGOS = {"PPO": PPO, "A2C": A2C}

def regime_probe(data_root, symbol):
    env = ForexEnv(data_root, symbol, "M15","H1","D1")
    df = env.df
    trend = float((df["ema_gap"].abs().rolling(48).mean().iloc[-1]))
    vol = float(df["atr"].rolling(48).mean().iloc[-1])
    if trend > 0.003: return "trending"
    if vol   < 0.002: return "ranging"
    return "mixed"

def eval_model(model, env):
    obs, _ = env.reset()
    done = False; ret = 0.0; rets=[]
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, trunc, info = env.step(act)
        ret += float(rew); rets.append(float(rew))
    score = ret / (np.std(rets)+1e-8)
    return {"ret":ret, "score":score}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="GBPUSD")
    ap.add_argument("--data_root", default="data/raw/forex_kaggle_multiTF")
    ap.add_argument("--models_dir", default="models/finrl")
    args = ap.parse_args()

    env = ForexEnv(args.data_root, args.symbol, "M15","H1","D1")
    regime = regime_probe(args.data_root, args.symbol)
    print("Regime:", regime)

    best=None; best_score=-1e9
    for p in Path(args.models_dir).glob(f"*best_model.zip"):
        algo="PPO" if "PPO" in p.name.upper() else ("A2C" if "A2C" in p.name.upper() else None)
        if not algo: continue
        mdl = ALGOS[algo].load(str(p), env=env)
        met = eval_model(mdl, env)
        if met["score"] > best_score:
            best={"algo":algo,"path":str(p),"metrics":met}; best_score=met["score"]

    mem = json.loads(MEM.read_text()) if MEM.exists() else {}
    prev = mem.get(regime)
    if best and (prev is None or best["metrics"]["score"] > prev["metrics"]["score"]):
        mem[regime]=best
        MEM.parent.mkdir(parents=True, exist_ok=True)
        MEM.write_text(json.dumps(mem, indent=2))
        print("✅ memory updated:", best)
    else:
        print("ℹ️ kept previous or no model found")

if __name__ == "__main__":
    main()
