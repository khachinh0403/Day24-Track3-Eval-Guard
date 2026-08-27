from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # Run RAGAS evaluation
    # 1. Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    # try:
    #     from ragas import evaluate
    #     from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #     from datasets import Dataset
    #
    #     dataset = Dataset.from_dict({
    #         "question": questions, "answer": answers,
    #         "contexts": contexts, "ground_truth": ground_truths,
    #     })
    #     result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                         context_precision, context_recall])
    #     df = result.to_pandas()
    #     per_question = [EvalResult(question=row["question"], answer=row["answer"],
    #         contexts=row["contexts"], ground_truth=row["ground_truth"],
    #         faithfulness=float(row.get("faithfulness", 0.0)),
    #         answer_relevancy=float(row.get("answer_relevancy", 0.0)),
    #         context_precision=float(row.get("context_precision", 0.0)),
    #         context_recall=float(row.get("context_recall", 0.0)))
    #         for _, row in df.iterrows()]
    #     return {"faithfulness": ..., "answer_relevancy": ...,
    #             "context_precision": ..., "context_recall": ..., "per_question": [...]}
    # except Exception as e:
    #     print(f"  ⚠️  RAGAS evaluation failed: {e}")
    #     return zeros
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        result = evaluate(Dataset.from_dict({"question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths}),
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
        df = result.to_pandas()
        per_question = [EvalResult(str(q), str(a), list(c), str(gt),
            float(row.get("faithfulness", 0) or 0), float(row.get("answer_relevancy", 0) or 0),
            float(row.get("context_precision", 0) or 0), float(row.get("context_recall", 0) or 0))
            for (q, a, c, gt), (_, row) in zip(zip(questions, answers, contexts, ground_truths), df.iterrows())]
        keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        return {**{k: float(df[k].mean()) if k in df else 0.0 for k in keys}, "per_question": per_question}
    except Exception as exc:
        print(f"  Warning: RAGAS unavailable or failed: {exc}")
        per_question = []
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            answer_words, truth_words = set(a.lower().split()), set(gt.lower().split())
            overlap = len(answer_words & truth_words) / max(len(truth_words), 1)
            context_hit = 1.0 if any(truth_words & set(x.lower().split()) for x in c) else 0.0
            per_question.append(EvalResult(q, a, c, gt, context_hit, overlap, context_hit, context_hit))
        keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        return {k: (sum(getattr(x, k) for x in per_question) / len(per_question) if per_question else 0.0)
                for k in keys} | {"per_question": per_question}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Run failure analysis
    # 1. diagnostic_tree = {
    #        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    #        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    #        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    #        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    #    }
    # 2. For each EvalResult: compute avg of 4 metrics, find worst_metric
    # 3. Sort by avg ascending → take bottom_n
    # 4. Return [{"question": ..., "worst_metric": ..., "score": ...,
    #             "diagnosis": ..., "suggested_fix": ...}]
    fixes = {
        "faithfulness": ("Answer may contain unsupported claims", "Constrain generation to retrieved context and cite evidence."),
        "context_recall": ("Relevant information was not retrieved", "Improve chunking, query expansion, or add BM25 retrieval."),
        "context_precision": ("Retrieved context contains irrelevant chunks", "Apply reranking and metadata filtering."),
        "answer_relevancy": ("Answer does not directly address the question", "Improve the answer prompt and query normalization."),
    }
    metrics = list(fixes)
    ranked = sorted(eval_results, key=lambda x: sum(getattr(x, m) for m in metrics) / 4)[:bottom_n]
    output = []
    for item in ranked:
        worst = min(metrics, key=lambda m: getattr(item, m))
        diagnosis, suggested_fix = fixes[worst]
        output.append({"question": item.question, "expected": item.ground_truth, "got": item.answer,
                       "worst_metric": worst, "score": float(getattr(item, worst)),
                       "diagnosis": diagnosis, "suggested_fix": suggested_fix})
    return output


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
