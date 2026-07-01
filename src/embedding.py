"""Sentence embedding and vector search utilities."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

LOGGER = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper around Sentence Transformers with cosine search helpers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_transformer: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model = None
        self.uses_transformer = False
        if not use_transformer:
            LOGGER.info("Using TF-IDF similarity instead of Sentence Transformers.")
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.uses_transformer = True
        except Exception as exc:
            LOGGER.warning(
                "Could not load Sentence Transformer model %s: %s. Falling back "
                "to TF-IDF similarity for local demo execution.",
                model_name,
                exc,
            )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode text into normalized vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        if self.model is None:
            raise RuntimeError("Transformer model is unavailable in fallback mode.")
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 250,
        )
        return vectors.astype(np.float32)

    def semantic_scores(self, job_text: str, candidate_texts: list[str]) -> np.ndarray:
        """Return cosine similarity scores between one job and many candidates."""
        if not self.uses_transformer:
            return self._tfidf_scores(job_text, candidate_texts)
        candidate_vectors = self.encode(candidate_texts)
        job_vector = self.encode([job_text])
        if len(candidate_vectors) == 0:
            return np.array([], dtype=np.float32)
        return cosine_similarity(job_vector, candidate_vectors)[0].clip(0.0, 1.0)

    def _tfidf_scores(self, job_text: str, candidate_texts: list[str]) -> np.ndarray:
        if not candidate_texts:
            return np.array([], dtype=np.float32)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([job_text, *candidate_texts])
        scores = cosine_similarity(matrix[0], matrix[1:])[0]
        return scores.clip(0.0, 1.0).astype(np.float32)


def top_k_with_faiss(
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    top_k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return approximate top-k matches using FAISS when available."""
    try:
        import faiss
    except ImportError:
        LOGGER.info("faiss-cpu is unavailable; using numpy ranking.")
        scores = candidate_vectors @ query_vector.reshape(-1)
        indices = np.argsort(scores)[::-1][:top_k]
        return scores[indices], indices

    index = faiss.IndexFlatIP(candidate_vectors.shape[1])
    index.add(candidate_vectors.astype(np.float32))
    scores, indices = index.search(query_vector.astype(np.float32), top_k)
    return scores[0], indices[0]
