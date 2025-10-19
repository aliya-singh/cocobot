import logging
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from models.embeddings import get_embedding_model
from utils.document_processor import DocumentProcessor, DocumentStore
from utils.cache import get_cache_manager, EmbeddingCache

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval-Augmented Generation Engine
    Manages document embeddings and semantic search
    """
    
    def __init__(self, vector_db_path: str = "./data/vector_db", chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize RAG engine
        
        Args:
            vector_db_path: Path to store vector embeddings
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embedding_model = get_embedding_model()
        self.cache_manager = get_cache_manager()
        self.embedding_cache = EmbeddingCache(self.cache_manager)
        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.doc_store = DocumentStore()
        
        # In-memory index
        self.embeddings = np.array([])  # Shape: [n_docs, embedding_dim]
        self.documents = []  # List of document chunks
        
        # Load existing index if available
        self._load_index()
        
        logger.info("RAG Engine initialized")
    
    def _save_index(self):
        """Save embeddings index to disk"""
        try:
            index_file = self.vector_db_path / "index.pkl"
            data = {
                'embeddings': self.embeddings,
                'documents': self.documents
            }
            
            with open(index_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Index saved: {self.embeddings.shape[0]} embeddings")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _load_index(self):
        """Load embeddings index from disk"""
        try:
            index_file = self.vector_db_path / "index.pkl"
            
            if index_file.exists():
                with open(index_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.embeddings = data['embeddings']
                self.documents = data['documents']
                
                logger.info(f"Index loaded: {self.embeddings.shape[0]} embeddings")
            else:
                logger.info("No existing index found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
    
    def add_documents(self, file_paths: List[str], doc_ids: List[str] = None) -> Dict[str, Any]:
        """
        Add documents to RAG system
        
        Args:
            file_paths: List of document file paths
            doc_ids: Optional document IDs (defaults to file names)
        
        Returns:
            Summary of added documents
        """
        if doc_ids is None:
            doc_ids = [Path(f).stem for f in file_paths]
        
        all_chunks = []
        embeddings_list = []
        
        # Process documents
        for file_path, doc_id in zip(file_paths, doc_ids):
            try:
                logger.info(f"Processing document: {file_path}")
                chunks = self.doc_processor.process_document(file_path)
                all_chunks.extend(chunks)
                
                # Embed chunks
                texts = [chunk['content'] for chunk in chunks]
                chunk_embeddings = self.embedding_model.encode(texts, batch_size=32)
                embeddings_list.append(chunk_embeddings)
                
                self.doc_store.add_document(doc_id, chunks)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        if embeddings_list:
            # Combine with existing embeddings
            new_embeddings = np.vstack(embeddings_list)
            
            if len(self.embeddings) > 0:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            else:
                self.embeddings = new_embeddings
            
            self.documents.extend(all_chunks)
            
            # Save updated index
            self._save_index()
        
        return {
            "documents_added": len(doc_ids),
            "chunks_added": len(all_chunks),
            "total_chunks": len(self.documents),
            "total_embeddings": len(self.embeddings)
        }
    
    def search(self, query: str, top_k: int = 5, confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            confidence_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of retrieved chunks with scores
        """
        if len(self.documents) == 0:
            logger.warning("No documents in RAG system")
            return []
        
        try:
            # Embed query
            query_embedding = self.embedding_model.encode_single(query)
            
            # Calculate similarities
            similarities = self.embedding_model.similarities_batch(query_embedding, self.embeddings)
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                
                if score >= confidence_threshold:
                    results.append({
                        "content": self.documents[idx]['content'],
                        "source": self.documents[idx]['source'],
                        "score": score
                    })
            
            logger.info(f"RAG search returned {len(results)} results for query: {query[:50]}...")
            return results
        
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []
    
    def get_search_confidence(self, query: str) -> float:
        """
        Get confidence score for query (0-1)
        Based on best match in documents
        
        Args:
            query: Search query
        
        Returns:
            Confidence score
        """
        results = self.search(query, top_k=1, confidence_threshold=0.0)
        
        if results:
            return results[0]['score']
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        return {
            "total_documents": len(self.doc_store.metadata),
            "total_chunks": len(self.documents),
            "embeddings_shape": self.embeddings.shape if len(self.embeddings) > 0 else "Empty",
            "embedding_model": self.embedding_model.get_config(),
            "vector_db_path": str(self.vector_db_path)
        }
    
    def clear(self):
        """Clear all documents and embeddings"""
        self.embeddings = np.array([])
        self.documents = []
        self.doc_store.clear()
        
        # Delete index file
        index_file = self.vector_db_path / "index.pkl"
        if index_file.exists():
            index_file.unlink()
        
        logger.info("RAG engine cleared")


# Global RAG instance
_rag_instance = None


def get_rag_engine(vector_db_path: str = "./data/vector_db") -> RAGEngine:
    """Get or create RAG engine (singleton)"""
    global _rag_instance
    
    if _rag_instance is None:
        _rag_instance = RAGEngine(vector_db_path)
    
    return _rag_instance


def reset_rag_engine():
    """Reset RAG engine"""
    global _rag_instance
    _rag_instance = None


# Convenience functions
def add_documents_to_rag(file_paths: List[str]) -> Dict[str, Any]:
    """Quick document addition"""
    rag = get_rag_engine()
    return rag.add_documents(file_paths)


def search_rag(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Quick RAG search"""
    rag = get_rag_engine()
    return rag.search(query, top_k)