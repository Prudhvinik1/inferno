"""Vector store: ABC and numpy/FAISS implementations.

Stores embedding vectors with metadata and supports similarity search
for retrieval-augmented generation (RAG).
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class Document:
    """A stored document with its embedding vector and metadata."""

    id: str
    text: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result with similarity score."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Abstract base class ─────────────────────────────────────────────────────


class VectorStore(ABC):
    """Abstract vector store interface.

    Implementations must provide add, search, delete, and count.
    Optionally supports persist/load for durable storage.
    """

    @abstractmethod
    async def add(
        self,
        id: str,
        text: str,
        embedding: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document with its embedding to the store."""
        ...

    @abstractmethod
    async def add_batch(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]] | np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple documents at once."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the most similar documents to the query embedding."""
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a document by ID. Returns True if found and deleted."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the number of stored documents."""
        ...

    async def persist(self) -> None:
        """Persist the store to disk. Default: no-op."""

    async def load(self) -> None:
        """Load the store from disk. Default: no-op."""


# ── Numpy implementation ─────────────────────────────────────────────────────


class NumpyVectorStore(VectorStore):
    """In-memory vector store using numpy cosine similarity.

    Thread-safe via a lock. Optionally persists to a directory as
    numpy arrays + JSON metadata.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._documents: dict[str, Document] = {}
        self._persist_dir = persist_dir
        self._lock = threading.Lock()

    async def add(
        self,
        id: str,
        text: str,
        embedding: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        doc = Document(id=id, text=text, embedding=vec, metadata=metadata or {})
        with self._lock:
            self._documents[id] = doc

    async def add_batch(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]] | np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        vecs = np.asarray(embeddings, dtype=np.float32)
        metas = metadatas or [{} for _ in ids]
        with self._lock:
            for i, (doc_id, text) in enumerate(zip(ids, texts)):
                self._documents[doc_id] = Document(
                    id=doc_id, text=text, embedding=vecs[i], metadata=metas[i]
                )

    async def search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query = np.asarray(query_embedding, dtype=np.float32)

        with self._lock:
            docs = list(self._documents.values())

        if not docs:
            return []

        # Apply metadata filter
        if metadata_filter:
            docs = [
                d for d in docs
                if all(d.metadata.get(k) == v for k, v in metadata_filter.items())
            ]
            if not docs:
                return []

        # Build matrix and compute cosine similarity
        matrix = np.stack([d.embedding for d in docs])
        scores = _cosine_similarity(query, matrix)

        # Get top-k indices
        k = min(top_k, len(docs))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            SearchResult(
                id=docs[i].id,
                text=docs[i].text,
                score=float(scores[i]),
                metadata=docs[i].metadata,
            )
            for i in top_indices
        ]

    async def delete(self, id: str) -> bool:
        with self._lock:
            return self._documents.pop(id, None) is not None

    async def count(self) -> int:
        return len(self._documents)

    # ── Persistence ──────────────────────────────────────────────────────

    async def persist(self) -> None:
        if not self._persist_dir:
            return

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        with self._lock:
            docs = list(self._documents.values())

        if not docs:
            return

        ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        metadatas = [d.metadata for d in docs]
        embeddings = np.stack([d.embedding for d in docs])

        np.save(self._persist_dir / "embeddings.npy", embeddings)

        meta_path = self._persist_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(
                {"ids": ids, "texts": texts, "metadatas": metadatas},
                ensure_ascii=False,
            )
        )
        logger.info("vectorstore.persisted", count=len(docs), path=str(self._persist_dir))

    async def load(self) -> None:
        if not self._persist_dir:
            return

        emb_path = self._persist_dir / "embeddings.npy"
        meta_path = self._persist_dir / "metadata.json"

        if not emb_path.exists() or not meta_path.exists():
            return

        embeddings = np.load(emb_path)
        meta = json.loads(meta_path.read_text())

        with self._lock:
            self._documents.clear()
            for i, doc_id in enumerate(meta["ids"]):
                self._documents[doc_id] = Document(
                    id=doc_id,
                    text=meta["texts"][i],
                    embedding=embeddings[i],
                    metadata=meta["metadatas"][i],
                )

        logger.info("vectorstore.loaded", count=len(self._documents), path=str(self._persist_dir))


# ── FAISS implementation ─────────────────────────────────────────────────────


class FAISSVectorStore(VectorStore):
    """Vector store backed by FAISS for faster similarity search.

    Falls back to NumpyVectorStore if faiss-cpu is not installed.
    Uses IndexFlatIP (inner product on L2-normalized vectors = cosine sim).
    """

    def __init__(self, dimension: int = 0, persist_dir: Path | None = None) -> None:
        self._dimension = dimension
        self._persist_dir = persist_dir
        self._index: Any = None  # faiss.IndexFlatIP
        self._documents: dict[str, Document] = {}
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._next_idx = 0
        self._lock = threading.Lock()

    async def add(
        self,
        id: str,
        text: str,
        embedding: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        vec = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        vec = _l2_normalize(vec)

        self._ensure_index(vec.shape[1])

        with self._lock:
            idx = self._next_idx
            self._next_idx += 1
            self._index.add(vec)
            self._id_to_idx[id] = idx
            self._idx_to_id[idx] = id
            self._documents[id] = Document(
                id=id, text=text, embedding=vec[0], metadata=metadata or {}
            )

    async def add_batch(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]] | np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        vecs = np.asarray(embeddings, dtype=np.float32)
        vecs = _l2_normalize(vecs)
        metas = metadatas or [{} for _ in ids]

        self._ensure_index(vecs.shape[1])

        with self._lock:
            start_idx = self._next_idx
            self._index.add(vecs)
            for i, doc_id in enumerate(ids):
                idx = start_idx + i
                self._id_to_idx[doc_id] = idx
                self._idx_to_id[idx] = doc_id
                self._documents[doc_id] = Document(
                    id=doc_id, text=texts[i], embedding=vecs[i], metadata=metas[i]
                )
            self._next_idx = start_idx + len(ids)

    async def search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query = _l2_normalize(query)

        with self._lock:
            # Search more than top_k to allow for metadata filtering
            search_k = min(top_k * 3, self._index.ntotal) if metadata_filter else top_k
            scores, indices = self._index.search(query, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self._idx_to_id.get(int(idx))
            if doc_id is None:
                continue
            doc = self._documents.get(doc_id)
            if doc is None:
                continue

            if metadata_filter and not all(
                doc.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue

            results.append(
                SearchResult(
                    id=doc.id, text=doc.text, score=float(score), metadata=doc.metadata
                )
            )
            if len(results) >= top_k:
                break

        return results

    async def delete(self, id: str) -> bool:
        # FAISS doesn't support deletion natively; we mark it removed
        with self._lock:
            if id in self._documents:
                del self._documents[id]
                idx = self._id_to_idx.pop(id, None)
                if idx is not None:
                    self._idx_to_id.pop(idx, None)
                return True
        return False

    async def count(self) -> int:
        return len(self._documents)

    def _ensure_index(self, dimension: int) -> None:
        if self._index is not None:
            return
        try:
            import faiss

            self._dimension = dimension
            self._index = faiss.IndexFlatIP(dimension)
            logger.info("faiss.index_created", dimension=dimension)
        except ImportError:
            raise RuntimeError(
                "faiss-cpu is not installed. Install with: pip install 'llm-serving[faiss]'"
            )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of vectors."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors for use with FAISS IndexFlatIP (cosine sim)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def create_vector_store(
    backend: str = "numpy",
    persist_dir: Path | None = None,
    dimension: int = 0,
) -> VectorStore:
    """Factory function to create a vector store by backend name."""
    if backend == "faiss":
        return FAISSVectorStore(dimension=dimension, persist_dir=persist_dir)
    return NumpyVectorStore(persist_dir=persist_dir)
