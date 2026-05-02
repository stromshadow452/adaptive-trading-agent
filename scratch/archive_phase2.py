import os
import json
import shutil

def main():
    base_dir = os.path.abspath('.')
    archive_dir = os.path.join(base_dir, '_archive_unused')
    
    with open('scratch/classification.json') as f:
        data = json.load(f)
        
    unknown_files = data['unknown']
    
    moved_count = 0
    for file_path in unknown_files:
        if not os.path.exists(file_path):
            continue
            
        rel_path = os.path.relpath(file_path, base_dir)
        dest_path = os.path.join(archive_dir, rel_path)
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.move(file_path, dest_path)
        moved_count += 1
        
    print(f"Phase 2 Archive Complete. Moved {moved_count} UNKNOWN files.")

if __name__ == '__main__':
    main()
