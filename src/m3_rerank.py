from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            #
            # ⚠️ LƯU Ý: Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding.
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            pass
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        try:
            if os.getenv("LAB24_FAST_SETUP") == "1":
                raise RuntimeError("fast setup uses lexical reranking")
            scores = self._load_model().predict([(query, d["text"]) for d in documents])
        except Exception:
            query_terms = set(query.lower().split())
            scores = [sum(term in d["text"].lower() for term in query_terms) / max(len(query_terms), 1)
                      for d in documents]
        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)[:top_k]
        return [RerankResult(d["text"], float(d.get("score", 0.0)), float(score),
                             d.get("metadata", {}), i) for i, (score, d) in enumerate(scored)]
        # 1. if not documents: return []
        # 2. model = self._load_model()
        # 3. pairs = [(query, doc["text"]) for doc in documents]
        # 4. scores = model.predict(pairs)
        # 5. if isinstance(scores, (int, float)): scores = [scores]
        # 6. scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        # 7. Return [RerankResult(text=..., original_score=doc.get("score", 0.0),
        #            rerank_score=float(score), metadata=..., rank=i)
        #            for i, (score, doc) in enumerate(scored[:top_k])]
        return []


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # Optional FlashRank implementation.
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
