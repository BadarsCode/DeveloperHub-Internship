from pathlib import Path
from typing import List

from pypdf import PdfReader

try:
    from .documents import SimpleDocument
except ImportError:
    from documents import SimpleDocument


def process_all_pdfs(pdf_directory: str | Path) -> List[SimpleDocument]:
    """Load all PDFs recursively and attach simple source metadata."""
    all_documents: List[SimpleDocument] = []
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files for processing")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        try:
            reader = PdfReader(str(pdf_file))
            page_count = 0
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                all_documents.append(
                    SimpleDocument(
                        page_content=text,
                        metadata={
                            "source_file": pdf_file.name,
                            "file_type": "pdf",
                            "page": i + 1,
                        },
                    )
                )
                page_count += 1
            print(f"Loaded {page_count} pages")
        except Exception as exc:
            print(f"Failed to load {pdf_file}: {exc}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents
