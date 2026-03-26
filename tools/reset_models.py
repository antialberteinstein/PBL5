import os
import shutil
import logging

def reset_models():
    """Remove saved PCA and Scaler models."""
    models_dir = "models"
    if os.path.exists(models_dir):
        logging.info(f"Resetting models in directory: {models_dir}")
        for filename in os.listdir(models_dir):
            if filename.endswith(".pkl") or filename.endswith(".joblib") or filename.endswith(".json"):
                file_path = os.path.join(models_dir, filename)
                try:
                    os.unlink(file_path)
                    logging.info(f"Removed model file: {filename}")
                except Exception as e:
                    logging.error(f"Failed to delete {file_path}. Reason: {e}")
    else:
        logging.warning(f"Models directory {models_dir} does not exist.")
        
    logging.info("Reset models operation completed successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    reset_models()