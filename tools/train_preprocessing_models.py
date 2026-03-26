import os
import sys
import logging

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from classify.preprocessing import train_preprocessing_models

def main():
    """Train PCA and Scaler models from dataset."""
    logging.info("Starting training of preprocessing models...")
    try:
        pca, scaler = train_preprocessing_models()
        logging.info("Successfully trained and saved PCA & Scaler models.")
    except Exception as e:
        logging.error(f"Training failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()