import os, json, re, shutil
src_dirs=["data/raw/M15_only","data/raw/forex_backup_2020_2025"]
out="reports/daily/approved.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
if os.path.exists(out):
    shutil.copy2(out, out+".bak")
seen=set()
approved=[]
pat=re.compile(r'^([A-Z]{3,8})_([A-Za-z0-9]+)', re.I)
for d in src_dirs:
    if not os.path.isdir(d): continue
    for root,_,files in os.walk(d):
        for f in files:
            if not f.lower().endswith('.csv'): continue
            m=pat.match(f)
            if m:
                sym=m.group(1).upper()
                tf=m.group(2).upper()
            else:
                parts=f.split('_')
                if len(parts)>=2:
                    sym=parts[0].upper(); tf=parts[1].upper()
                else:
                    continue
            key=(sym,tf)
            if key in seen: continue
            seen.add(key)
            approved.append({"symbol": sym, "timeframe": tf})
with open(out,'w',encoding='utf8') as fp:
    json.dump(approved, fp, indent=2)
print("Wrote", out, "entries:", len(approved))
