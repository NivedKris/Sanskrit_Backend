import os
from PyPDF2 import PdfReader

class PDFProcessor:
    """
    Processor for handling PDF documents, extracting text,
    and preparing it for the LLM context.
    """
    
    def __init__(self):
        pass
    
    def extract_text(self, pdf_path):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            if not text:
                raise ValueError("No text could be extracted from the PDF")
            
            return text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def clean_text(self, text):
        """
        Clean extracted text for better processing.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            str: Cleaned text
        """
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        # Remove multiple newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
            
        return text 