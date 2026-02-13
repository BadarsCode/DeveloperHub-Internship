from typing import Any, List

try:
    from .documents import SimpleDocument
except ImportError:
    from documents import SimpleDocument


def _fallback_split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks: List[str] = []
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - max(0, chunk_overlap))
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + chunk_size)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start += step
    return chunks


def split_documents(
    documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Any]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        split_docs = text_splitter.split_documents(documents)
    except Exception:
        split_docs = []
        for doc in documents:
            text = getattr(doc, "page_content", "") or ""
            metadata = dict(getattr(doc, "metadata", {}) or {})
            for idx, chunk_text in enumerate(
                _fallback_split_text(text, chunk_size, chunk_overlap)
            ):
                chunk_meta = dict(metadata)
                chunk_meta["chunk_index"] = idx
                split_docs.append(
                    SimpleDocument(page_content=chunk_text, metadata=chunk_meta)
                )

    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    if split_docs:
        print(f"Example chunk: {split_docs[0].page_content[:200]}")
    return split_docs
