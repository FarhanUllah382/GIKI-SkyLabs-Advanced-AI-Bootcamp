import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

class DocumentParsingError(Exception):
    """Raised when a document cannot be loaded or parsed into text."""
    pass


def load_pdf(file_path: str) -> List[Document]:

    path = Path(file_path)
    if not path.exists():
        raise DocumentParsingError(f"File not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise DocumentParsingError(
            f"Expected a .pdf file, got: {path.suffix} (file: {file_path})"
        )

    logger.info(f"Loading PDF: {file_path}")

    try:
        loader = PyMuPDFLoader(file_path)
        raw_documents = loader.load()
    except Exception as e:
       
        raise DocumentParsingError(
            f"Failed to parse PDF '{file_path}': {e}"
        ) from e

    if not raw_documents:
        raise DocumentParsingError(
            f"No pages were extracted from '{file_path}'. File may be corrupted."
        )

    cleaned_documents: List[Document] = []
    empty_page_count = 0

    for doc in raw_documents:
        cleaned_text = _clean_text(doc.page_content)

        if not cleaned_text.strip():
            empty_page_count += 1
            logger.warning(
                f"Page {doc.metadata.get('page', '?')} of '{file_path}' "
                f"produced no extractable text (possibly a scanned image page)."
            )

        cleaned_documents.append(
            Document(
                page_content=cleaned_text,
                metadata=doc.metadata,  # preserves source + page number
            )
        )

    if empty_page_count == len(cleaned_documents):
       
        raise DocumentParsingError(
            f"'{file_path}' contains no extractable text on any page. "
            f"This may be a scanned PDF requiring OCR, which is out of scope "
            f"for this loader."
        )

    logger.info(
        f"Successfully parsed '{file_path}': {len(cleaned_documents)} pages "
        f"({empty_page_count} empty)."
    )

    return cleaned_documents


def _clean_text(text: str) -> str:
   
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = " ".join(text.split())

    return text.strip()
