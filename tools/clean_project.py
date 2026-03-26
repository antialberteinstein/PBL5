import os
import shutil
import logging

def clean_project():
    """Clear __pycache__ and other temporary files."""
    base_dir = "."
    logging.info(f"Cleaning project temporary files in: {os.path.abspath(base_dir)}")
    
    for root, dirs, files in os.walk(base_dir):
        # 1. Remove __pycache__
        if "__pycache__" in dirs:
            pycache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                logging.info(f"Removed: {pycache_path}")
            except Exception as e:
                logging.error(f"Failed to remove {pycache_path}: {e}")
                
        # 2. Remove .pyc files (redundant if __pycache__ is removed, but good for older versions)
        for file in files:
            if file.endswith(".pyc") or file.endswith(".pyo"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logging.info(f"Removed: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to remove {file_path}: {e}")
                    
    logging.info("Clean project operation completed successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    clean_project()