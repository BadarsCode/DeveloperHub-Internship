import argparse
from pathlib import Path

from .chunker import split_documents
from .config import DEFAULT_DATA_DIR, DEFAULT_VECTOR_STORE_DIR
from .embedding_manager import EmbeddingManager
from .llm import get_llm
from .loader import process_all_pdfs
from .pipeline import AdvancedRagPipeline
from .retriever import RAGRetriever
from .vector_store import VectorStore


def build_vector_store_if_needed(
    vector_store: VectorStore,
    embedding_manager: EmbeddingManager,
    data_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool,
) -> None:
    if rebuild:
        vector_store.clear()

    if vector_store.count() > 0:
        print("Existing vector store found. Skipping rebuild.")
        return

    documents = process_all_pdfs(data_dir)
    if not documents:
        raise ValueError(f"No PDF documents found under: {data_dir}")

    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(chunks, embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Notebook-based RAG chatbot built from document.ipynb logic."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_STORE_DIR)
    parser.add_argument("--collection", type=str, default="pdf_documents")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--question", type=str, default=None)
    return parser.parse_args()


def run_chatbot() -> None:
    args = parse_args()

    embedding_manager = EmbeddingManager(model_name=args.embedding_model)
    vector_store = VectorStore(
        collection_name=args.collection,
        persist_directory=str(args.vector_dir),
    )
    build_vector_store_if_needed(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild,
    )

    retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager)
    llm = get_llm(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    pipeline = AdvancedRagPipeline(retriever=retriever, llm=llm)

    if args.question:
        result = pipeline.query(
            args.question,
            top_k=args.top_k,
            min_score=args.min_score,
            summarize=False,
            stream=False,
        )
        print(f"\nAssistant: {result['answer']}\n")
        return

    print("Chatbot is ready. Type 'exit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        result = pipeline.query(
            question,
            top_k=args.top_k,
            min_score=args.min_score,
            summarize=False,
            stream=False,
        )
        print(f"\nAssistant: {result['answer']}")


if __name__ == "__main__":
    run_chatbot()
