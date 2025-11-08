# src/pdf_processor.py

import fitz  # PyMuPDF
import io

def extract_text_from_pdf(pdf_file_object):
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_file_object: A file-like object (e.g., from st.file_uploader)
                         containing the PDF data.
                         
    Returns:
        A single string containing all extracted text, or None if extraction fails.
    """
    try:
        # Read the bytes from the file object
        pdf_bytes = pdf_file_object.read()
        
        # Create a file-like object in memory for PyMuPDF
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        full_text = ""
        
        # Iterate through each page and extract text
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
            
        pdf_document.close()
        
        if not full_text.strip():
            # Handle case where PDF might be scanned/image-based
            return None 
            
        return full_text
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None