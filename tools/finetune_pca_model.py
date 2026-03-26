import os
import sys
import logging

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from classify.preprocessing import PCAProcessor

def main():
    """Fine-tune the PCA model."""
    logging.info("Fine-tuning PCA model...")
    pca = PCAProcessor()
    if not pca.load():
        logging.error("Failed to load existing PCA model. Train it first using Option 4.")
        return
        
    success, msg = pca.finetune()
    if success:
        logging.info(msg)
    else:
        logging.error(msg)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()