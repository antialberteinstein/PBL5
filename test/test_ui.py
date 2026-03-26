# ==============================================================================
#                           SECTION: MANUAL TEST ENTRY POINT
# ==============================================================================
"""
Functional test script for the Face Recognition System.
Allows testing of Register, Verify, and Update Face flows.
Administrative tasks are handled by dedicated scripts in /tools.

[UTILITIES USED FROM src/]:
- recog.face_recognition.InsightFaceDetector
- classify.cosine_classifier.CosineClassifier
- ui.registration_ui.RegistrationUI
- ui.update_face_ui.UpdateFaceUI
- ui.verification_ui.VerificationUI
- camera.opencv_client.OpenCVCamera
- classify.preprocessing.PCAProcessor
- classify.preprocessing.ScalerProcessor
- pipeline.classify.ClassificationPipeline
- pipeline.recog.RecognitionPipeline
"""

import os
import sys
import logging
import warnings
import shutil

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from logging_setup import setup_logging
from recog.face_recognition import InsightFaceDetector
from classify.cosine_classifier import CosineClassifier
from ui.registration_ui import RegistrationUI
from ui.update_face_ui import UpdateFaceUI
from ui.verification_ui import VerificationUI
from camera.opencv_client import OpenCVCamera
from classify.preprocessing import PCAProcessor, ScalerProcessor
from pipeline.classify import ClassificationPipeline
from pipeline.recog import RecognitionPipeline

# ==============================================================================
#                           SECTION: GLOBAL COMPONENTS
# ==============================================================================

face_recognizer = InsightFaceDetector()
classifier = CosineClassifier()
camera = OpenCVCamera()

# ==============================================================================
#                                   SECTION: MAIN
# ==============================================================================

def main():
    setup_logging()
    
    logging.info(
        "\n======================================================\n"
        "[UTILITIES USED FROM src/]:\n"
        "- recog.face_recognition.InsightFaceDetector\n"
        "- classify.cosine_classifier.CosineClassifier\n"
        "- ui.registration_ui.RegistrationUI\n"
        "- ui.update_face_ui.UpdateFaceUI\n"
        "- ui.verification_ui.VerificationUI\n"
        "- camera.opencv_client.OpenCVCamera\n"
        "- classify.preprocessing.PCAProcessor, ScalerProcessor\n"
        "- pipeline.classify.ClassificationPipeline\n"
        "- pipeline.recog.RecognitionPipeline\n"
        "======================================================\n"
    )
    
    face_recognizer.prepare()
    
    pca = PCAProcessor()
    scaler = ScalerProcessor()
    recog_pipeline = None
    
    # Try to load existing models
    if pca.load() and scaler.load():
        logging.info("Preloaded saved preprocessing models successfully.")
        pipeline = ClassificationPipeline(pca, scaler, classifier)
        recog_pipeline = RecognitionPipeline(face_recognizer)
        classify_pipeline = pipeline
    else:
        logging.warning("No saved preprocessing models found! Use 'tools/train_preprocessing_models.py' first.")
    
    while True:
        print("\n--- TEST MENU ---")
        print("  1. Register (New Person)")
        print("  2. Verify (Real-time Recognition)")
        print("  3. Update Face (Add images to existing ID)")
        print("  4. Reload Pipeline (After training/finetune)")
        print("  5. Exit")
        
        choice = input("\nSelect (1-5): ").strip()
        
        if choice == '1':
            if recog_pipeline is None:
                logging.error("Pipeline not initialized. Train models first.")
                continue
            class_id = input("Class ID: ").strip()
            if class_id:
                reg_ui = RegistrationUI(recog_pipeline, classify_pipeline)
                reg_ui.run(class_id, camera)
                
        elif choice == '2':
            if recog_pipeline is None:
                logging.error("Pipeline not initialized. Train models first.")
                continue
            verify_ui = VerificationUI(recog_pipeline, classify_pipeline)
            verify_ui.run(camera)
            
        elif choice == '3':
            if recog_pipeline is None:
                logging.error("Pipeline not initialized. Train models first.")
                continue
            class_id = input("Class ID to update: ").strip()
            if class_id:
                update_ui = UpdateFaceUI(recog_pipeline, classify_pipeline)
                update_ui.run(class_id, camera)
                
        elif choice == '4':
            if pca.load() and scaler.load():
                pipeline = ClassificationPipeline(pca, scaler, classifier)
                recog_pipeline = RecognitionPipeline(face_recognizer)
                classify_pipeline = pipeline
                logging.info("Pipeline reloaded successfully.")
            else:
                logging.error("Failed to reload models.")
                
        elif choice == '5':
            logging.info("Exiting test.")
            camera.release()
            break

if __name__ == "__main__":
    main()
