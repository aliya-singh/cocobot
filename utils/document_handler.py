"""Document handling and text extraction"""
import logging
from typing import Dict, Tuple
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from config.config import Config

logger = logging.getLogger(__name__)

class DocumentHandler:
    """Handle various document types"""
    
    @staticmethod
    def extract_pdf_text(file) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text[:Config.MAX_DOCUMENT_SIZE]
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return ""
    
    @staticmethod
    def extract_text_file(file) -> str:
        """Extract text from TXT/MD file"""
        try:
            return file.getvalue().decode('utf-8')[:Config.MAX_DOCUMENT_SIZE]
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return ""
    
    @staticmethod
    def extract_from_file(file, file_type: str) -> Tuple[str, bool]:
        """Extract text based on file type"""
        if file_type == 'application/pdf':
            text = DocumentHandler.extract_pdf_text(file)
        else:
            text = DocumentHandler.extract_text_file(file)
        
        success = len(text) > 0
        return text, success
    
    @staticmethod
    def fetch_website_content(url: str) -> Tuple[str, bool]:
        """Fetch and extract text from website"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:Config.MAX_DOCUMENT_SIZE], True
        except Exception as e:
            logger.error(f"Website fetch error: {e}")
            return "", False
    
    @staticmethod
    def get_domain_name(url: str) -> str:
        """Extract domain name from URL"""
        try:
            domain = urlparse(url).netloc
            return domain or url
        except:
            return url