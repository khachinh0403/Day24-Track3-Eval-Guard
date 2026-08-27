from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Vietnamese word segmentation
    # 1. from underthesea import word_tokenize
    # 2. segmented = word_tokenize(text, format="text")
    # 3. return segmented.replace("_", " ")
    #
    # ⚠️ LƯU Ý: underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
    # BM25 tokenize bằng split(" ") → "nghỉ_phép" thành 1 token,
    # nhưng query "nghỉ phép" thành 2 token → KHÔNG khớp.
    # Phải replace("_", " ") để BM25 hoạt động đúng.
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        # Build BM25 index
        # 1. self.documents = chunks
        # 2. For each chunk: segment_vietnamese(chunk["text"]) → split by space
        # 3. self.corpus_tokens = [tokenized list for each chunk]
        # 4. from rank_bm25 import BM25Okapi
        #    self.bm25 = BM25Okapi(self.corpus_tokens)
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c["text"]).split() for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            self.bm25 = False

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        # Search BM25 index
        # 1. if self.bm25 is None: return []
        # 2. tokenized_query = segment_vietnamese(query).split()
        # 3. scores = self.bm25.get_scores(tokenized_query)
        # 4. top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # 5. Return [SearchResult(text=..., score=..., metadata=..., method="bm25")]
        #    Lọc scores[i] > 0 để bỏ docs không liên quan.
        if self.bm25 is None:
            return []
        query_tokens = set(segment_vietnamese(query).split())
        if self.bm25 is False:
            scores = [sum(token in set(doc) for token in query_tokens) for doc in self.corpus_tokens]
        else:
            scores = self.bm25.get_scores(list(query_tokens))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [SearchResult(self.documents[i]["text"], float(scores[i]), self.documents[i].get("metadata", {}), "bm25")
                for i in indices if scores[i] > 0]


class DenseSearch:
    def __init__(self):
        self.fast_mode = os.getenv("LAB24_FAST_SETUP") == "1"
        try:
            if self.fast_mode:
                self.client = None
            else:
                from qdrant_client import QdrantClient
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except ImportError:
            self.client = None
        self._encoder = None
        self.documents = []

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        # Build dense index
        # 1. from qdrant_client.models import Distance, VectorParams, PointStruct
        # 2. self.client.recreate_collection(collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        # 3. texts = [c["text"] for c in chunks]
        # 4. vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        # 5. points = [PointStruct(id=i, vector=v.tolist(), payload={**c.get("metadata", {}), "text": c["text"]}) ...]
        # 6. self.client.upsert(collection, points)
        self.documents = chunks
        if self.fast_mode:
            return
        if self.client is None:
            return
        from qdrant_client.models import Distance, VectorParams, PointStruct
        self.client.recreate_collection(collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        vectors = self._get_encoder().encode([c["text"] for c in chunks], show_progress_bar=False)
        points = [PointStruct(id=i, vector=v.tolist(), payload={**c.get("metadata", {}), "text": c["text"]})
                  for i, (c, v) in enumerate(zip(chunks, vectors))]
        if points:
            self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # Search dense index
        # 1. query_vector = self._get_encoder().encode(query).tolist()
        # 2. response = self.client.query_points(collection, query=query_vector, limit=top_k)
        # 3. Return [SearchResult(text=pt.payload["text"], score=pt.score, metadata=pt.payload, method="dense")
        #            for pt in response.points]
        #
        # ⚠️ LƯU Ý: qdrant-client >= 2.0 dùng query_points(), KHÔNG phải search().
        if self.client is None:
            query_terms = set(segment_vietnamese(query).lower().split())
            ranked = sorted(self.documents, key=lambda c: len(query_terms & set(segment_vietnamese(c["text"]).lower().split())), reverse=True)
            return [SearchResult(c["text"], float(len(query_terms & set(segment_vietnamese(c["text"]).lower().split()))), c.get("metadata", {}), "dense")
                    for c in ranked[:top_k]]
        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
        return [SearchResult(pt.payload.get("text", ""), float(pt.score), dict(pt.payload), "dense")
                for pt in response.points]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    # Reciprocal rank fusion
    # 1. rrf_scores = {}  # text → {"score": float, "result": SearchResult}
    # 2. For each result_list in results_list:
    #      For rank, result in enumerate(result_list):
    #        if result.text not in rrf_scores: rrf_scores[result.text] = {"score": 0.0, "result": result}
    #        rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    # 3. Sort by score descending
    # 4. Return top_k SearchResult with method="hybrid"
    scores = {}
    for results in results_list:
        for rank, result in enumerate(results):
            entry = scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [SearchResult(x["result"].text, x["score"], x["result"].metadata, "hybrid") for x in ranked]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
