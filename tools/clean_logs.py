import os
import shutil
import logging

def clean_logs():
    log_dir = "logs"
    if os.path.exists(log_dir):
        logging.info(f"Cleaning logs directory: {log_dir}")
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')
    else:
        logging.warning(f"Logs directory {log_dir} does not exist.")
        
    logging.info("Clean logs operation completed successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    clean_logs()