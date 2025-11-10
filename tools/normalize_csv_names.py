# tools/normalize_csv_names.py
import os, re, shutil

roots = ["data/raw/M15_only", "data/raw/forex_backup_2020_2025"]
backup_dir = "data/raw/_name_backup"
os.makedirs(backup_dir, exist_ok=True)

def canonical(name):
    # strip path, lower not changing symbol case — keep original symbol case
    n = name.strip()
    # replace " to " variants with _to_
    n = re.sub(r'\s*[Tt][Oo]\s*', '_to_', n)
    # replace multiple spaces with single underscore
    n = re.sub(r'[\s]+', '_', n)
    # replace sequences of underscores with single
    n = re.sub(r'_+', '_', n)
    # remove spaces around underscores
    n = re.sub(r'\s*_\s*', '_', n)
    # ensure .csv lower
    if not n.lower().endswith('.csv'):
        base, ext = os.path.splitext(n)
        n = base + '.csv'
    return n

for root in roots:
    if not os.path.isdir(root):
        print("Missing folder:", root)
        continue
    for dirpath,_,files in os.walk(root):
        for f in files:
            if not f.lower().endswith('.csv'): 
                continue
            new = canonical(f)
            if new != f:
                old_full = os.path.join(dirpath, f)
                new_full = os.path.join(dirpath, new)
                # backup original first
                try:
                    shutil.copy2(old_full, os.path.join(backup_dir, f))
                except Exception as e:
                    print("backup failed for", f, e)
                # avoid overwriting existing file: add suffix if exists
                if os.path.exists(new_full):
                    i = 1
                    base, ext = os.path.splitext(new)
                    while os.path.exists(os.path.join(dirpath, f"{base}_{i}{ext}")):
                        i += 1
                    new = f"{base}_{i}{ext}"
                    new_full = os.path.join(dirpath, new)
                os.rename(old_full, new_full)
                print("RENAMED:", f, "->", new)
print("Done. Backups in", backup_dir)
