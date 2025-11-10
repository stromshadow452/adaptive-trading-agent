import json, os
out="reports/daily/approved.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
data = [
  {"symbol":"EURUSD","timeframe":"M15","side":"buy","size":0.05},
  {"symbol":"EURGBP","timeframe":"M15","side":"buy","size":0.05},
  {"symbol":"GBPUSD","timeframe":"M15","side":"buy","size":0.05},
  {"symbol":"USDCAD","timeframe":"M15","side":"buy","size":0.05},
  {"symbol":"USDJPY","timeframe":"M15","side":"buy","size":0.05}
]
with open(out,"w",encoding="utf8") as f:
    json.dump(data,f,indent=2,ensure_ascii=False)
print("Wrote", out, "entries:", len(data))
