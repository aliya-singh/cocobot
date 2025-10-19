import logging
from pathlib import Path
from typing import List, Dict, Any
import re

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process and chunk documents for RAG"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks (for context continuity)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @staticmethod
    def load_text_file(file_path: str) -> str:
        """
        Load text from file
        
        Args:
            file_path: Path to text file
        
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise
    
    @staticmethod
    def load_pdf_file(file_path: str) -> str:
        """
        Load text from PDF file
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Extracted text as string
        """
        try:
            import PyPDF2
            
            text = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
            
            return "\n".join(text)
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise
    
    @staticmethod
    def load_markdown_file(file_path: str) -> str:
        """Load markdown file (just plain text read)"""
        return DocumentProcessor.load_text_file(file_path)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Fix common issues
        text = text.replace('\n', ' ').strip()
        
        return text
    
    def chunk_text(self, text: str, doc_name: str = "Unknown") -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            doc_name: Document name for tracking
        
        Returns:
            List of chunk dicts with 'content' and 'source'
        """
        # Clean text
        text = self.clean_text(text)
        
        if len(text) <= self.chunk_size:
            return [{"content": text, "source": doc_name}]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for last period, question mark, or exclamation mark
                last_sentence = max(
                    text.rfind('.', start, end),
                    text.rfind('?', start, end),
                    text.rfind('!', start, end)
                )
                
                if last_sentence > start + self.chunk_size // 2:
                    end = last_sentence + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "source": doc_name
                })
            
            # Move start position (with overlap)
            start = end - self.chunk_overlap
        
        logger.info(f"Chunked {doc_name} into {len(chunks)} chunks")
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a document
        
        Args:
            file_path: Path to document file
        
        Returns:
            List of chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Load based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self.load_pdf_file(str(file_path))
        elif file_path.suffix.lower() in ['.md', '.txt']:
            text = self.load_text_file(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Chunk the text
        chunks = self.chunk_text(text, doc_name=file_path.stem)
        
        return chunks
    
    def process_batch(self, file_paths: List[str]) -> List[Dict[str, str]]:
        """
        Process multiple documents
        
        Args:
            file_paths: List of file paths
        
        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                # Continue with next file
        
        logger.info(f"Processed {len(file_paths)} files into {len(all_chunks)} total chunks")
        return all_chunks


class DocumentStore:
    """Simple in-memory document store with metadata"""
    
    def __init__(self):
        """Initialize document store"""
        self.documents = []
        self.metadata = {}
    
    def add_document(self, doc_id: str, chunks: List[Dict[str, str]], metadata: Dict[str, Any] = None):
        """
        Add processed document
        
        Args:
            doc_id: Unique document ID
            chunks: List of text chunks
            metadata: Optional metadata dict
        """
        self.documents.extend(chunks)
        self.metadata[doc_id] = {
            "chunk_count": len(chunks),
            "total_chars": sum(len(chunk['content']) for chunk in chunks),
            "metadata": metadata or {}
        }
        
        logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            "total_documents": len(self.metadata),
            "total_chunks": len(self.documents),
            "total_characters": sum(len(doc['content']) for doc in self.documents),
            "documents": self.metadata
        }
    
    def clear(self):
        """Clear the store"""
        self.documents = []
        self.metadata = {}
        logger.info("Document store cleared")


# Convenience functions
def process_document(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, str]]:
    """Quick document processing"""
    processor = DocumentProcessor(chunk_size, chunk_overlap)
    return processor.process_document(file_path)


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100, doc_name: str = "text") -> List[Dict[str, str]]:
    """Quick text chunking"""
    processor = DocumentProcessor(chunk_size, chunk_overlap)
    return processor.chunk_text(text, doc_name)