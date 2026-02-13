import re
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

try:
    from .documents import SimpleDocument
except ImportError:
    from documents import SimpleDocument


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def load_documents_from_uploaded_files(
    uploaded_files: Iterable[object], temp_dir: str | Path
) -> List[SimpleDocument]:
    """
    Convert Streamlit uploaded files into LangChain Documents.
    Supports: .pdf, .txt
    """
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    all_documents: List[SimpleDocument] = []
    for uploaded in uploaded_files:
        file_name = getattr(uploaded, "name", "uploaded_file")
        safe_name = _sanitize_filename(file_name)
        suffix = Path(safe_name).suffix.lower()
        raw_bytes = uploaded.getvalue()

        if suffix == ".pdf":
            local_path = temp_path / safe_name
            local_path.write_bytes(raw_bytes)
            reader = PdfReader(str(local_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                all_documents.append(
                    SimpleDocument(
                        page_content=text,
                        metadata={
                            "source_file": safe_name,
                            "file_type": "pdf",
                            "page": i + 1,
                        },
                    )
                )
            continue

        if suffix == ".txt":
            text = raw_bytes.decode("utf-8", errors="ignore")
            doc = SimpleDocument(
                page_content=text,
                metadata={"source_file": safe_name, "file_type": "text"},
            )
            all_documents.append(doc)
            continue

        print(f"Skipping unsupported file type: {file_name}")

    return all_documents


def load_wikipedia_topics(
    topics: List[str], max_docs_per_topic: int = 2, lang: str = "en"
) -> List[SimpleDocument]:
    """Load documents from Wikipedia topics."""
    try:
        import wikipedia
        from wikipedia import exceptions as wiki_exceptions
    except Exception as exc:
        raise ImportError(
            "Wikipedia support requires the 'wikipedia' package. Install it with: pip install wikipedia"
        ) from exc

    all_documents: List[SimpleDocument] = []
    wikipedia.set_lang(lang)
    for topic in topics:
        clean_topic = topic.strip()
        if not clean_topic:
            continue
        try:
            candidate_titles = wikipedia.search(clean_topic, results=max_docs_per_topic)
            if not candidate_titles:
                continue

            for title in candidate_titles[:max_docs_per_topic]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                except wiki_exceptions.DisambiguationError as disamb:
                    if not disamb.options:
                        continue
                    page = wikipedia.page(disamb.options[0], auto_suggest=False)
                except wiki_exceptions.PageError:
                    continue

                content = page.content or ""
                all_documents.append(
                    SimpleDocument(
                        page_content=content,
                        metadata={
                            "source_file": f"wikipedia:{page.title}",
                            "file_type": "wikipedia",
                            "url": page.url,
                        },
                    )
                )
        except Exception as exc:
            print(f"Failed to load topic '{clean_topic}': {exc}")
    return all_documents
