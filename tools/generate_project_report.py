import os
import datetime

IGNORE_DIRS = {'.git', '.venv', '.venv311', '__pycache__', 'venv', 'models', 'checkpoints', 'data', 'repo_trash', 'test_output', 'artifacts', '.pytest_cache'}
REPORT_FILE = r'e:\adaptive-trading-agent (2)\adaptive-trading-agent (2)\artifacts\project_scan_report.md'
ROOT_DIR = r'e:\adaptive-trading-agent (2)\adaptive-trading-agent (2)'

def count_loc(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for line in f if line.strip() and not line.strip().startswith('#'))
    except Exception:
        return 0

def scan_project():
    total_files = 0
    total_loc = 0
    python_files = 0
    
    dir_structure = {}
    loc_by_module = {}
    
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        rel_path = os.path.relpath(root, ROOT_DIR)
        top_module = rel_path.split(os.sep)[0] if rel_path != '.' else 'root'
        
        dir_structure[rel_path] = []
        
        if top_module not in loc_by_module:
            loc_by_module[top_module] = {'files': 0, 'loc': 0}
            
        for file in files:
            if file.endswith(('.pyc', '.pkl', '.bin', '.h5', '.onnx', '.pdf', '.exe', '.dll')):
                continue
            
            total_files += 1
            filepath = os.path.join(root, file)
            dir_structure[rel_path].append(file)
            
            if file.endswith('.py'):
                python_files += 1
                loc = count_loc(filepath)
                total_loc += loc
                loc_by_module[top_module]['files'] += 1
                loc_by_module[top_module]['loc'] += loc

    return total_files, total_loc, python_files, dir_structure, loc_by_module

def generate_report():
    total_files, total_loc, python_files, dir_structure, loc_by_module = scan_project()
    
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("# Adaptive Trading Agent - Project Scan Report\n\n")
        f.write(f"**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Total Scanned Files:** {total_files}\n")
        f.write(f"- **Python Files:** {python_files}\n")
        f.write(f"- **Total Lines of Code (Python, excluding blanks/comments):** {total_loc:,}\n\n")
        
        f.write("## 2. Codebase Distribution by Module\n\n")
        f.write("| Module/Directory | Files | Lines of Code |\n")
        f.write("|-----------------|-------|---------------|\n")
        
        sorted_modules = sorted(loc_by_module.items(), key=lambda x: x[1]['loc'], reverse=True)
        for mod, stats in sorted_modules:
            if stats['files'] > 0:
                f.write(f"| `{mod}` | {stats['files']} | {stats['loc']:,} |\n")
        
        f.write("\n## 3. High-Level Directory Overview\n\n")
        for d in sorted(dir_structure.keys()):
            if d == '.' or not dir_structure[d]: continue
            if d.count(os.sep) > 1: continue # Only show top level and secondary
            f.write(f"- **`{d}/`** ({len(dir_structure[d])} files)\n")

    print(f"SUCCESS: Report written to {REPORT_FILE}")

if __name__ == '__main__':
    generate_report()
