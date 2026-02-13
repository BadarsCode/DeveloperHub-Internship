import os
import uuid
from typing import Any, List

import numpy as np

try:
    import chromadb
except Exception:
    chromadb = None


class InMemoryCollection:
    """Minimal Chroma-compatible collection for local fallback."""

    def __init__(self, name: str, metadata: dict | None = None):
        self.name = name
        self.metadata = metadata or {}
        self._ids: List[str] = []
        self._metadatas: List[dict] = []
        self._documents: List[str] = []
        self._embeddings: np.ndarray | None = None

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids = []
        self._metadatas = []
        self._documents = []
        self._embeddings = None

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
        documents: List[str],
    ) -> None:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if self._embeddings is None:
            self._embeddings = arr
        else:
            if arr.shape[1] != self._embeddings.shape[1]:
                raise ValueError("Embedding dimension mismatch")
            self._embeddings = np.vstack([self._embeddings, arr])

        self._ids.extend(ids)
        self._metadatas.extend(metadatas)
        self._documents.extend(documents)

    def query(self, query_embeddings: List[List[float]], n_results: int = 5) -> dict:
        if self._embeddings is None or self.count() == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]],
            }

        query_vec = np.asarray(query_embeddings, dtype=np.float32)[0]
        db = self._embeddings

        if query_vec.shape[0] != db.shape[1]:
            # Align dimensions if different embedding backends are used.
            aligned = np.zeros((db.shape[1],), dtype=np.float32)
            overlap = min(query_vec.shape[0], db.shape[1])
            aligned[:overlap] = query_vec[:overlap]
            query_vec = aligned

        db_norm = np.linalg.norm(db, axis=1) + 1e-12
        q_norm = np.linalg.norm(query_vec) + 1e-12
        similarity = (db @ query_vec) / (db_norm * q_norm)
        ranked_idx = np.argsort(-similarity)[: max(0, int(n_results))]

        ids = [self._ids[i] for i in ranked_idx]
        metadatas = [self._metadatas[i] for i in ranked_idx]
        documents = [self._documents[i] for i in ranked_idx]
        distances = [float(1.0 - similarity[i]) for i in ranked_idx]

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
            "ids": [ids],
        }


class VectorStore:
    """Manage document embeddings in ChromaDB or local in-memory fallback."""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.backend = "chromadb" if chromadb is not None else "in_memory"
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        os.makedirs(self.persist_directory, exist_ok=True)
        if chromadb is not None:
            try:
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "PDF document embeddings for RAG"},
                )
                print(f"Vector store initialized. Collection: {self.collection_name}")
                print(f"Existing documents in collection: {self.collection.count()}")
                self.backend = "chromadb"
                return
            except Exception as exc:
                print(
                    "chromadb is installed but failed to initialize persistent storage. "
                    f"Falling back to in-memory store. Error: {exc}"
                )

        self.collection = InMemoryCollection(
            name=self.collection_name,
            metadata={"description": "In-memory vector store fallback"},
        )
        self.backend = "in_memory"
        if chromadb is None:
            print(
                "chromadb not installed. "
                "Using in-memory vector store fallback (non-persistent)."
            )
        else:
            print(
                "Using in-memory vector store fallback (non-persistent). "
                "(chromadb initialization failed)."
            )

    def count(self) -> int:
        return int(self.collection.count())

    def clear(self) -> None:
        if self.backend == "chromadb":
            if self.client is None:
                raise ValueError("Vector store client is not initialized")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"},
            )
        else:
            self.collection.clear()
        print(f"Collection '{self.collection_name}' cleared")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store")
        ids: List[str] = []
        metadatas: List[dict] = []
        document_text: List[str] = []
        embedding_list: List[list[float]] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            document_text.append(doc.page_content)
            embedding_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                metadatas=metadatas,
                documents=document_text,
            )
            print(f"Successfully added {len(documents)} documents")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as exc:
            print(f"Error adding documents to vector store: {exc}")
            raise
