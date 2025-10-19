import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Lightweight embedding model for RAG.
    Uses sentence-transformers (all-MiniLM-L6-v2 by default).
    NO API calls - runs locally!
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence-transformer model
                - all-MiniLM-L6-v2: Fast, 384-dim, best for speed (~50MB)
                - all-mpnet-base-v2: Better quality, 768-dim (~430MB)
                - all-distilroberta-v1: Balanced, 768-dim (~265MB)
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model (cached locally after first download)"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def encode(self, texts: list | str, batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: Single string or list of strings to embed
            batch_size: Process in batches for memory efficiency
        
        Returns:
            numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Encode with batching for efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding
        
        Args:
            text: Text to embed
        
        Returns:
            numpy array of shape [embedding_dim]
        """
        embedding = self.encode([text], batch_size=1)[0]
        return embedding
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0 to 1, higher = more similar)
        """
        # Normalize embeddings
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-10)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-10)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(similarity)
    
    def similarities_batch(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate similarities between query and multiple documents (vectorized)
        
        Args:
            query_embedding: Single query embedding [embedding_dim]
            document_embeddings: Multiple document embeddings [n_docs, embedding_dim]
        
        Returns:
            Array of similarity scores [n_docs]
        """
        # Normalize
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        doc_norms = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Batch cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities
    
    def get_config(self) -> dict:
        """Get model configuration"""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "cpu"
        }


# Singleton instance (cached in memory)
_embedding_instance = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    """
    Get or create embedding model instance (singleton pattern)
    This ensures we only load the model once
    
    Args:
        model_name: Name of the embedding model
    
    Returns:
        EmbeddingModel instance
    """
    global _embedding_instance
    
    if _embedding_instance is None:
        _embedding_instance = EmbeddingModel(model_name)
    
    return _embedding_instance


def reset_embedding_model():
    """Reset the embedding model instance (for testing)"""
    global _embedding_instance
    _embedding_instance = None