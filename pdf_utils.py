import fitz  # PyMuPDF

def read_pdf(file) -> tuple[str, dict]:
    """
    Reads a PDF file stream and returns the extracted text along with document metadata.
    """
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text_chunks = []
    num_pages = len(pdf)
    
    for page in pdf:
        page_text = page.get_text("text")
        if page_text.strip():
            text_chunks.append(page_text)
            
    full_text = "\n\n".join(text_chunks)
    word_count = len(full_text.split())
    
    metadata = {
        "num_pages": num_pages,
        "word_count": word_count,
        "char_count": len(full_text)
    }
    
    return full_text, metadata


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """
    Splits text into overlapping chunks for RAG indexing.
    """
    if not text or not text.strip():
        return []
        
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks