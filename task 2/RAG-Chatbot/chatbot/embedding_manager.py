from typing import List

import numpy as np


class EmbeddingManager:
    """Handle document embedding generation.

    Preferred backend: SentenceTransformer
    Fallback backend: local hash-based embeddings (no external model needed)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.backend = "sentence_transformer"
        self.fallback_dim = 384
        self._load_model()

    def _load_model(self) -> None:
        try:
            print(f"Loading embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {dim}")
        except Exception as exc:
            self.backend = "hash_fallback"
            self.model = None
            print(
                f"Could not load SentenceTransformer ({exc}). "
                f"Falling back to hash embeddings (dim={self.fallback_dim})."
            )

    def _hash_embed(self, text: str, dim: int) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for token in text.lower().split():
            idx = hash(token) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"Generating embeddings for {len(texts)} texts...")
        if self.backend == "sentence_transformer" and self.model is not None:
            embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            embeddings = np.vstack(
                [self._hash_embed(text=text or "", dim=self.fallback_dim) for text in texts]
            )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
