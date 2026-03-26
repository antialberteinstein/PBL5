import os
import sys
import logging

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from classify.preprocessing import ScalerProcessor

def main():
    """Fine-tune the Scaler model."""
    logging.info("Fine-tuning Scaler model...")
    scaler = ScalerProcessor()
    if not scaler.load():
        logging.error("Failed to load existing Scaler model. Train it first using Option 4.")
        return
        
    success, msg = scaler.finetune()
    if success:
        logging.info(msg)
    else:
        logging.error(msg)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()