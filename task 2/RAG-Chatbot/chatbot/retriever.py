from typing import Any, Dict, List

try:
    from .embedding_manager import EmbeddingManager
    from .vector_store import VectorStore
except ImportError:
    from embedding_manager import EmbeddingManager
    from vector_store import VectorStore


class RAGRetriever:
    """Handle query-based retrieval from the vector store."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self, query: str, top_k: int = 5, score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: {query}")
        print(f"top_k: {top_k}, score_threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            result = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
            )

            retrieved_docs: List[Dict[str, Any]] = []
            if result["documents"] and result["documents"][0]:
                documents = result["documents"][0]
                metadatas = result["metadatas"][0]
                distances = result["distances"][0]
                ids = result["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )
                print(f"Retrieved {len(retrieved_docs)} documents after filtering")
            else:
                print("No documents found")
            return retrieved_docs
        except Exception as exc:
            print(f"Error during retrieval: {exc}")
            return []
