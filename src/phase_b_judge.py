from __future__ import annotations

"""Phase B: pairwise LLM judging, swap calibration, and bias analysis."""

import json
import os
import sys
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH, JUDGE_MODEL, OPENAI_API_KEY, TEST_SET_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _valid_scores(scores: object) -> dict[str, float]:
    scores = scores if isinstance(scores, dict) else {}
    return {key: max(0.0, min(1.0, float(scores.get(key, 0.0)))) for key in ("A", "B")}


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Choose the stronger answer; use a deterministic fallback without an API key."""
    prompt = f'''Compare two RAG answers for accuracy, completeness, and conciseness.
Return JSON only: {{"winner":"A|B|tie","reasoning":"...","scores":{{"A":0.0,"B":0.0}}}}.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}'''
    key = OPENAI_API_KEY.strip()
    if key and key != "sk-...":
        try:
            from openai import OpenAI
            response = OpenAI(api_key=key).chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": "Return valid JSON only."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content)
            winner = payload.get("winner", "tie")
            if winner not in {"A", "B", "tie"}:
                raise ValueError("invalid winner")
            return {"winner": winner, "reasoning": str(payload.get("reasoning", "")),
                    "scores": _valid_scores(payload.get("scores"))}
        except Exception as error:
            fallback_note = f"LLM unavailable ({type(error).__name__}); lexical fallback used."
    else:
        fallback_note = "No usable OpenAI API key; lexical fallback used."

    question_terms = set(question.lower().split())
    score_a = len(set(answer_a.lower().split()) & question_terms) / max(1, len(question_terms))
    score_b = len(set(answer_b.lower().split()) & question_terms) / max(1, len(question_terms))
    winner = "tie" if abs(score_a - score_b) < 0.05 else ("A" if score_a > score_b else "B")
    return {"winner": winner, "reasoning": fallback_note,
            "scores": {"A": round(score_a, 3), "B": round(score_b, 3)}}


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    winner_pass2 = {"A": "B", "B": "A", "tie": "tie"}[pass2_raw["winner"]]
    consistent = pass1["winner"] == winner_pass2
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=pass1["winner"] if consistent else "tie",
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=consistent, scores_pass1=_valid_scores(pass1.get("scores")),
        scores_pass2={"A": _valid_scores(pass2_raw.get("scores"))["B"],
                      "B": _valid_scores(pass2_raw.get("scores"))["A"]},
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        return 0.0
    total = len(judge_labels)
    observed = sum(judge == human for judge, human in zip(judge_labels, human_labels)) / total
    labels = set(judge_labels) | set(human_labels)
    expected = sum((judge_labels.count(label) / total) * (human_labels.count(label) / total)
                   for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    inconsistent = sum(not result.position_consistent for result in judge_results)
    decisive = [result for result in judge_results if result.final_winner != "tie"]
    a_wins_a_longer = sum(result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
                          for result in decisive)
    b_wins_b_longer = sum(result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
                          for result in decisive)
    position_rate = inconsistent / total if total else 0.0
    verbosity_rate = (a_wins_a_longer + b_wins_b_longer) / len(decisive) if decisive else 0.0
    return {
        "total_judged": total, "position_bias_rate": round(position_rate, 3),
        "position_bias_count": inconsistent, "verbosity_bias": round(verbosity_rate, 3),
        "verbosity_details": {"a_wins_a_longer": a_wins_a_longer,
                              "b_wins_b_longer": b_wins_b_longer,
                              "total_decisive": len(decisive)},
        "interpretation": ("High position bias: retain swap-and-average and inspect prompts."
                           if position_rate > 0.3 else "Low position bias in this sample."),
    }


def save_judge_report(results: list[JudgeResult], kappa: float,
                      path: str = "reports/judge_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"total_judged": len(results), "cohen_kappa": kappa,
                   "bias_report": bias_report(results),
                   "results": [asdict(result) for result in results]}, handle,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as handle:
        labels = json.load(handle)
    with open(TEST_SET_PATH, encoding="utf-8") as handle:
        references = {item["id"]: item["ground_truth"] for item in json.load(handle)}
    results = [swap_and_average(item["question"], item["model_answer"], references[item["question_id"]])
               for item in labels]
    judge_labels = [1 if result.final_winner in {"A", "tie"} else 0 for result in results]
    save_judge_report(results, cohen_kappa(judge_labels, [item["human_label"] for item in labels]))
    print(f"Saved Phase B report for {len(results)} labelled answers.")
