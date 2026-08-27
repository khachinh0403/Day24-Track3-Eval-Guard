from __future__ import annotations

"""Phase A: evaluate 50 RAG answers and summarize failure patterns."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return sum((self.faithfulness, self.answer_relevancy,
                    self.context_precision, self.context_recall)) / 4

    @property
    def worst_metric(self) -> str:
        scores = {"faithfulness": self.faithfulness,
                  "answer_relevancy": self.answer_relevancy,
                  "context_precision": self.context_precision,
                  "context_recall": self.context_recall}
        return min(scores, key=scores.get)


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run python setup_answers.py first.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(f"Unknown distribution: {distribution!r}")
        groups[distribution].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    if not answers:
        return []
    required = ("id", "distribution", "question", "answer", "contexts", "ground_truth")
    for index, item in enumerate(answers):
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Answer {index} is missing {', '.join(missing)}")
    from src.m4_eval import evaluate_ragas
    raw = evaluate_ragas(
        [item["question"] for item in answers],
        [item["answer"] for item in answers],
        [item["contexts"] for item in answers],
        [item["ground_truth"] for item in answers],
    )
    per_question = raw.get("per_question", [])
    if len(per_question) != len(answers):
        raise RuntimeError("RAGAS must return one result per input answer")
    return [RagasResult(
        question_id=item["id"], distribution=item["distribution"],
        question=item["question"], answer=item["answer"],
        contexts=item["contexts"], ground_truth=item["ground_truth"],
        faithfulness=float(score.faithfulness), answer_relevancy=float(score.answer_relevancy),
        context_precision=float(score.context_precision), context_recall=float(score.context_recall),
    ) for item, score in zip(answers, per_question)]


def bottom_10(results: list[RagasResult]) -> list[dict]:
    output = []
    for rank, result in enumerate(sorted(results, key=lambda item: item.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({"rank": rank, "question_id": result.question_id,
                       "distribution": result.distribution, "question": result.question,
                       "avg_score": round(result.avg_score, 4), "worst_metric": result.worst_metric,
                       "diagnosis": diagnosis, "suggested_fix": suggested_fix})
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    distributions = ("factual", "multi_hop", "adversarial")
    matrix = {metric: {distribution: 0 for distribution in distributions}
              for metric in DIAGNOSTIC_TREE}
    for result in results:
        matrix[result.worst_metric][result.distribution] += 1
    dominant_distribution = max(
        distributions, key=lambda distribution: sum(matrix[metric][distribution] for metric in matrix)
    )
    dominant_metric = max(matrix, key=lambda metric: sum(matrix[metric].values()))
    return {"matrix": matrix, "dominant_failure_distribution": dominant_distribution,
            "dominant_failure_metric": dominant_metric,
            "insight": (f"Distribution '{dominant_distribution}' has the most failures; "
                        f"'{dominant_metric}' is the dominant weak metric. "
                        f"Recommended action: {DIAGNOSTIC_TREE[dominant_metric][1]}.")}


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                        path: str = "reports/ragas_50q.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    per_distribution = {}
    for distribution in ("factual", "multi_hop", "adversarial"):
        subset = [result for result in results if result.distribution == distribution]
        if subset:
            per_distribution[distribution] = {
                "count": len(subset),
                "faithfulness": sum(x.faithfulness for x in subset) / len(subset),
                "answer_relevancy": sum(x.answer_relevancy for x in subset) / len(subset),
                "context_precision": sum(x.context_precision for x in subset) / len(subset),
                "context_recall": sum(x.context_recall for x in subset) / len(subset),
                "avg_score": sum(x.avg_score for x in subset) / len(subset),
            }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"total_questions": len(results), "per_distribution": per_distribution,
                   "failure_clusters": clusters, "bottom_10": bottom_10(results)},
                  handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    results = run_ragas_50q(load_answers())
    save_phase_a_report(results, cluster_analysis(results))
    print(f"Saved Phase A report for {len(results)} questions.")
