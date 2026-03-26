import os
import shutil
import logging

def clean_database():
    db_dir = "database"
    if os.path.exists(db_dir):
        logging.info(f"Cleaning database directory: {db_dir}")
        for filename in os.listdir(db_dir):
            file_path = os.path.join(db_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')
    else:
        logging.warning(f"Database directory {db_dir} does not exist.")
    
    logging.info("Clean database operation completed successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    clean_database()