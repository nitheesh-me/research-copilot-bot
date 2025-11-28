import fitz  # PyMuPDF
import hashlib
from moya.tools.tool import Tool

def extract_text_chunks(pdf_path: str, chunk_size: int = 1200, stride: int = 300):
    """
    Extract text from a PDF and split it into chunks.

    :param pdf_path: Path to the PDF file.
    :param chunk_size: Size of each chunk in characters.
    :param stride: Stride for overlapping chunks.
    :return: A tuple containing the list of chunks and metadata.
    """
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text:
            full_text.append((page_num+1, text))

    joined = "\n".join(f"[PAGE {p}]\n{t}" for p,t in full_text)

    chunks = []
    i = 0
    L = len(joined)
    while i < L:
        part = joined[i:i+chunk_size]
        chunks.append({"text": part, "start": i, "end": min(i+chunk_size, L)})
        i += (chunk_size - stride)

    meta = {"pages": doc.page_count, "chars": L, "sha256": hashlib.sha256(joined.encode()).hexdigest()}
    return chunks, meta

pdf_tool = Tool(
    name="extract_text_chunks",
    description="Extract text from a PDF and split it into chunks.",
    function=extract_text_chunks
)
