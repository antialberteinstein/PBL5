# ==============================================================================
#                           SECTION: PREPROCESSING MODULE
# ==============================================================================
"""
Preprocessing module containing PCA and Scaler for face embeddings.
"""

import logging
import os
import threading
from typing import Optional

import numpy as np
import joblib
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


# ==============================================================================
#                           SECTION: CONFIGURATION
# ==============================================================================
# Global default configurations for preprocessing (PCA/Scaler) have been moved 
# into the respective class constructors.


# ==============================================================================
#                           SECTION: PCA PROCESSOR
# ==============================================================================

class PCAProcessor:
    """
    PCA-based dimensionality reduction for face embeddings.
    
    Supports online learning by storing transformed vectors in database
    and automatically finetuning after every FINETUNE_BATCH_SIZE vectors.
    """
    
    def __init__(
        self,
        n_components: int = 128,
        models_dir: str = "models",
        pca_model_path: str = "models/pca_model.joblib",
        pca_vectors_path: str = "models/pca_vectors.bin",
        finetune_min_batch_size: int = 36,
    ):
        """
        Initialize PCA processor.
        
        Args:
            n_components: Number of principal components to keep
            models_dir: Directory to save models
            pca_model_path: Path to PCA model file
            pca_vectors_path: Path to stored vectors for finetuning
            finetune_min_batch_size: Min vectors to trigger finetune
        """
        self.n_components = n_components
        self.models_dir = models_dir
        self.pca_model_path = pca_model_path
        self.pca_vectors_path = pca_vectors_path
        self.finetune_min_batch_size = finetune_min_batch_size
        
        self.pca: Optional[IncrementalPCA] = None
        self._finetune_lock = threading.Lock()
        
        # Ensure models dir exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _append_vectors(self, new_vectors: np.ndarray) -> None:
        """Thread-safe O(1) append of raw vectors bytes to disk."""
        with self._finetune_lock:
            try:
                # Ensure it's float32 or float64 consistently
                vec_bytes = new_vectors.astype(np.float32).tobytes()
                with open(self.pca_vectors_path, "ab") as f:
                    f.write(vec_bytes)
            except Exception as e:
                logging.warning(f"Failed to save PCA vectors: {e}")
    
    def fit(
        self,
        embeddings: np.ndarray,
    ) -> None:
        """
        Fit PCA model on embeddings.
        
        Args:
            embeddings: Training embeddings (shape: [n_samples, embedding_dim])
        """
        ### 1. Initialize PCA model
        self.pca = IncrementalPCA(n_components=self.n_components)
        
        ### 2. Fit on embeddings
        self.pca.fit(embeddings)
        
        logging.info(f"PCA fitted on {len(embeddings)} samples")
    
    def transform(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Apply PCA transformation to embeddings.
        
        Stores ORIGINAL (pre-transform) vectors in database for online learning,
        then triggers finetune after every FINETUNE_BATCH_SIZE transforms.
        
        Args:
            embeddings: Input embeddings (shape: [n_samples, embedding_dim])
            
        Returns:
            Reduced embeddings (shape: [n_samples, n_components])
            
        Raises:
            RuntimeError: If PCA model is not fitted
        """
        if self.pca is None:
            raise RuntimeError("PCA model not fitted. Call fit() first.")
        
        ### Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        ### Store ORIGINAL embeddings BEFORE transformation for online learning
        self._append_vectors(embeddings)
        
        ### Apply transformation AFTER storing originals
        transformed = self.pca.transform(embeddings)
        
        return transformed
    
    def _finetune_async(self) -> None:
        """Background finetune task."""
        # Deprecated: Finetuning is now manually triggered via the UI
        pass
    
    def finetune(self) -> tuple[bool, str]:
        """
        Finetune PCA model using stored vectors from disk.
        
        Retrieves all stored vectors, performs incremental PCA update,
        then clears the file.
        
        Returns:
            Tuple of (success_boolean, message_string)
        """
        if self.pca is None:
            return False, "Cannot finetune: PCA model not initialized"
        
        with self._finetune_lock:
            try:
                if not os.path.exists(self.pca_vectors_path):
                    return False, "No stored vectors found for PCA finetuning"
                    
                # Read raw bytes and reshape to (N, original_dim)
                # Note: PCA input dim is 512 for InsightFace embeddings
                raw_vectors = np.fromfile(self.pca_vectors_path, dtype=np.float32)
                
                if len(raw_vectors) == 0:
                    return False, "Stored vectors file is empty"
                
                # Reshape. We assume input to PCA is 512-dim (Buffalo_l default)
                dim = 512
                if len(raw_vectors) % dim != 0:
                    logging.warning(f"Vector byte size mismatch. Total floats: {len(raw_vectors)}")
                    
                vectors = raw_vectors.reshape(-1, dim)
                count = len(vectors)
                
                if count < self.finetune_min_batch_size:
                    return False, f"Not enough vectors to finetune PCA. Have {count}, need {self.finetune_min_batch_size}."
                
                logging.info(f"Finetuning PCA with {count} vectors")
                
                # Perform incremental learning using partial_fit
                self.pca.partial_fit(vectors)
                
                # Clear the file after successful finetune
                os.remove(self.pca_vectors_path)
                
                # Save the finetuned model
                self.save()
                
                msg = f"PCA model finetuned with {count} new vectors"
                logging.info(msg)
                return True, msg
                
            except Exception as e:
                err = f"Error during finetune: {e}"
                logging.error(err)
                return False, err
            
    def save(self, path: Optional[str] = None) -> None:
        """Save the PCA model to disk."""
        if path is None:
            path = self.pca_model_path
            
        if self.pca is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.pca, path)
            logging.info(f"PCA model saved to {path}")
            
    def load(self, path: Optional[str] = None) -> bool:
        """Load the PCA model from disk."""
        if path is None:
            path = self.pca_model_path
            
        if os.path.exists(path):
            try:
                self.pca = joblib.load(path)
                logging.info(f"PCA model loaded from {path}")
                return True
            except Exception as e:
                logging.error(f"Failed to load PCA model: {e}")
        return False


# ==============================================================================
#                           SECTION: SCALER PROCESSOR CLASS
# ==============================================================================

class ScalerProcessor:
    """
    StandardScaler-based normalization for face embeddings.
    
    Supports online learning by storing transformed vectors in database
    and automatically finetuning after every FINETUNE_BATCH_SIZE vectors.
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        scaler_model_path: str = "models/scaler_model.joblib",
        scaler_vectors_path: str = "models/scaler_vectors.bin",
        finetune_min_batch_size: int = 36,
        pca_n_components: int = 128,
    ):
        """
        Initialize scaler processor.
        
        Args:
            models_dir: Directory to save models
            scaler_model_path: Path to scaler model file
            scaler_vectors_path: Path to stored vectors for finetuning
            finetune_min_batch_size: Min vectors to trigger finetune
            pca_n_components: Expected input dimension (matching PCA output)
        """
        self.models_dir = models_dir
        self.scaler_model_path = scaler_model_path
        self.scaler_vectors_path = scaler_vectors_path
        self.finetune_min_batch_size = finetune_min_batch_size
        self.pca_n_components = pca_n_components
        
        self.scaler: Optional[StandardScaler] = None
        self._finetune_lock = threading.Lock()
        
        # Ensure models dir exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _append_vectors(self, new_vectors: np.ndarray) -> None:
        """Thread-safe O(1) append of raw vector bytes to disk."""
        with self._finetune_lock:
            try:
                vec_bytes = new_vectors.astype(np.float32).tobytes()
                with open(self.scaler_vectors_path, "ab") as f:
                    f.write(vec_bytes)
            except Exception as e:
                logging.warning(f"Failed to save Scaler vectors: {e}")
    
    def fit(
        self,
        embeddings: np.ndarray,
    ) -> None:
        """
        Fit scaler on embeddings.
        
        Args:
            embeddings: Training embeddings (shape: [n_samples, embedding_dim])
        """
        ### 1. Initialize scaler
        self.scaler = StandardScaler()
        
        ### 2. Fit on embeddings
        self.scaler.fit(embeddings)
        
        logging.info(f"Scaler fitted on {len(embeddings)} samples")
    
    def transform(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Apply normalization to embeddings.
        
        Stores ORIGINAL (pre-scaling) vectors in database for online learning,
        then triggers finetune after every FINETUNE_BATCH_SIZE transforms.
        
        Args:
            embeddings: Input embeddings (shape: [n_samples, embedding_dim])
            
        Returns:
            Normalized embeddings (same shape as input)
            
        Raises:
            RuntimeError: If scaler is not fitted
        """
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        ### Ensure 2D array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        ### Store ORIGINAL embeddings BEFORE scaling for online learning
        self._append_vectors(embeddings)
        
        ### Apply transformation AFTER storing originals
        transformed = self.scaler.transform(embeddings)
        
        return transformed
    
    def _finetune_async(self) -> None:
        """Background finetune task."""
        # Deprecated: Finetuning is now manually triggered via the UI
        pass
    
    def finetune(self) -> tuple[bool, str]:
        """
        Finetune scaler model using stored vectors from disk.
        
        Retrieves all stored vectors, performs incremental update,
        then clears the file.
        
        Returns:
            Tuple of (success_boolean, message_string)
        """
        if self.scaler is None:
            return False, "Cannot finetune: Scaler model not initialized"
        
        with self._finetune_lock:
            try:
                if not os.path.exists(self.scaler_vectors_path):
                    return False, "No stored vectors found for Scaler finetuning"
                    
                # Read raw bytes
                # Scaler input is PCA_N_COMPONENTS (128-dim)
                raw_vectors = np.fromfile(self.scaler_vectors_path, dtype=np.float32)
                
                if len(raw_vectors) == 0:
                    return False, "Stored vectors file is empty"
                    
                dim = self.pca_n_components
                vectors = raw_vectors.reshape(-1, dim)
                count = len(vectors)
                
                if count < self.finetune_min_batch_size:
                    return False, f"Not enough vectors to finetune Scaler. Have {count}, need {self.finetune_min_batch_size}."
                
                logging.info(f"Finetuning Scaler with {count} vectors")
                
                # Perform incremental learning using partial_fit
                self.scaler.partial_fit(vectors)
                
                # Clear the file after successful finetune
                os.remove(self.scaler_vectors_path)
                
                # Save the finetuned model
                self.save()
                
                msg = f"Scaler model finetuned with {count} new vectors"
                logging.info(msg)
                return True, msg
                
            except Exception as e:
                err = f"Error during finetune: {e}"
                logging.error(err)
                return False, err
            
    def save(self, path: Optional[str] = None) -> None:
        """Save the Scaler model to disk."""
        if path is None:
            path = self.scaler_model_path
            
        if self.scaler is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.scaler, path)
            logging.info(f"Scaler model saved to {path}")
            
    def load(self, path: Optional[str] = None) -> bool:
        """Load the Scaler model from disk."""
        if path is None:
            path = self.scaler_model_path
            
        if os.path.exists(path):
            try:
                self.scaler = joblib.load(path)
                logging.info(f"Scaler model loaded from {path}")
                return True
            except Exception as e:
                logging.error(f"Failed to load Scaler model: {e}")
        return False


# ==============================================================================
#                           SECTION: TRAINING FUNCTIONS
# ==============================================================================

def train_preprocessing_models(
    lfw_dataset_path: str = "dataset/lfw.npz",
    pca_n_components: int = 128
) -> tuple[PCAProcessor, ScalerProcessor]:
    """
    Train PCA and Scaler models on LFW dataset embeddings.
    
    Args:
        lfw_dataset_path: Path to LFW embeddings npz file
        pca_n_components: Number of components for PCA
    
    Returns:
        Tuple of (pca, scaler) trained instances
    """
    logging.info("Starting preprocessing model training on LFW dataset...")
    
    ### 1. Load LFW dataset
    if not os.path.exists(lfw_dataset_path):
        raise FileNotFoundError(
            f"LFW dataset not found at {lfw_dataset_path}. "
            "Please ensure the dataset file exists."
        )
    
    logging.info(f"Loading embeddings from {lfw_dataset_path}...")
    data = np.load(lfw_dataset_path)
    
    # Extract embeddings (assuming key is 'embeddings' or 'data')
    if 'embeddings' in data:
        embeddings = data['embeddings']
    elif 'data' in data:
        embeddings = data['data']
    else:
        # Try to get the first array in the file
        embeddings = data[list(data.keys())[0]]
    
    if embeddings.size == 0:
        raise ValueError("LFW dataset is empty")
    
    logging.info(f"Loaded {len(embeddings)} embeddings from LFW dataset")
    logging.info(f"Embedding shape: {embeddings.shape}")
    
    ### 2. Train PCA first (dimensionality reduction: 512 -> 128)
    logging.info(f"Training PCA (n_components={pca_n_components})...")
    pca = PCAProcessor(n_components=pca_n_components)
    pca.fit(embeddings)
    logging.info("PCA model trained.")
    
    ### 3. Apply PCA transformation
    reduced_embeddings = pca.transform(embeddings)
    logging.info(f"Reduced embedding shape: {reduced_embeddings.shape}")
    
    ### 4. Train Scaler on PCA-reduced embeddings
    logging.info("Training StandardScaler on PCA-reduced embeddings...")
    scaler = ScalerProcessor(pca_n_components=pca_n_components)
    scaler.fit(reduced_embeddings)
    logging.info("Scaler trained.")
    
    ### 5. Save models to disk
    pca.save()
    scaler.save()
    
    logging.info("Preprocessing models trained successfully on LFW dataset.")
    
    return pca, scaler
