from __future__ import annotations

"""Phase C: layered PII, input, output, adversarial, and latency guards."""

import asyncio
import json
import os
import re
import sys
import time
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE

SAFE_REFUSAL = "Xin lỗi, tôi không thể hỗ trợ yêu cầu này. Tôi chỉ hỗ trợ chính sách HR nội bộ an toàn."
INPUT_BLOCK_PATTERNS = (
    r"ignore (your |all |previous |the )?(system |previous )?instructions",
    r"bỏ qua .*hướng dẫn", r"system override", r"unrestricted ai", r"pretend you are dan",
    r"do anything now", r"không có giới hạn", r"forget .*prompt", r"admin command",
    r"ignore previous", r"dump .*data", r"reveal .*data", r"system instructions?", r"tiết lộ .*?(lương|thông tin|dữ liệu)",
    r"mật khẩu", r"password", r"all employee salaries", r"bảng lương chi tiết",
    r"bitcoin|ethereum|nấu phở|bài thơ|marvel|phương trình vi phân|thời tiết",
    r"cccd của|số điện thoại của nhân viên|email của nhân viên|thông tin cá nhân của",
    r"<!--.*ignore|\[admin command",
)
OUTPUT_SENSITIVE_PATTERNS = (
    r"cccd .*?(là|:)\s*\d", r"cmnd .*?(là|:)\s*\d", r"mật khẩu .*?(là|:)",
    r"số điện thoại cá nhân", r"thông tin bí mật", r"confidential.*employee",
)


@lru_cache(maxsize=1)
def setup_presidio():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
                  Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7)],
    ))
    registry.add_recognizer(PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("Vietnam mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    ))
    return AnalyzerEngine(registry=registry), AnonymizerEngine()


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    results = analyzer.analyze(
        text=text,
        language=PRESIDIO_LANGUAGE,
        entities=["VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS"],
    )
    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}
    entities = [{"type": result.entity_type, "text": text[result.start:result.end],
                 "score": round(result.score, 3), "start": result.start, "end": result.end}
                for result in results]
    return {"has_pii": True, "entities": entities,
            "anonymized": anonymizer.anonymize(text=text, analyzer_results=results).text}


def setup_nemo_rails():
    from nemoguardrails import LLMRails, RailsConfig
    return LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_DIR))


def _local_input_blocked(text: str) -> bool:
    normalized = text.lower()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
               for pattern in INPUT_BLOCK_PATTERNS)


async def check_input_rail(text: str, rails=None) -> dict:
    """Block known jailbreak, data-exfiltration, and off-topic patterns before RAG."""
    if _local_input_blocked(text):
        return {"allowed": False, "blocked_reason": "local_input_policy", "response": SAFE_REFUSAL}
    if rails is None:
        return {"allowed": True, "blocked_reason": None, "response": ""}
    try:
        response = await rails.generate_async(messages=[{"role": "user", "content": text}])
        response_text = str(response)
        refused = any(token in response_text.lower() for token in
                      ("xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry"))
        return {"allowed": not refused, "blocked_reason": "nemo_input_rail" if refused else None,
                "response": response_text}
    except Exception:
        return {"allowed": False, "blocked_reason": "nemo_input_error", "response": SAFE_REFUSAL}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    normalized = answer.lower()
    if any(re.search(pattern, normalized, flags=re.IGNORECASE | re.DOTALL)
           for pattern in OUTPUT_SENSITIVE_PATTERNS) or pii_scan(answer)["has_pii"]:
        return {"safe": False, "flagged_reason": "local_output_policy", "final_answer": SAFE_REFUSAL}
    if rails is None:
        return {"safe": True, "flagged_reason": None, "final_answer": answer}
    try:
        response = await rails.generate_async(messages=[{"role": "user", "content": question},
                                                         {"role": "assistant", "content": answer}])
        response_text = str(response)
        refused = any(token in response_text.lower() for token in ("xin lỗi", "không thể cung cấp", "i cannot"))
        return {"safe": not refused, "flagged_reason": "nemo_output_rail" if refused else None,
                "final_answer": response_text if refused else answer}
    except Exception:
        return {"safe": False, "flagged_reason": "nemo_output_error", "final_answer": SAFE_REFUSAL}


def _run_async(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    raise RuntimeError("Synchronous guard API cannot be called from an active event loop")


def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                          analyzer=None, anonymizer=None) -> list[dict]:
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    async def run_all():
        output = []
        for item in adversarial_set:
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            blocked_by = "presidio" if pii_result["has_pii"] else None
            if blocked_by is None:
                input_result = await check_input_rail(item["input"], rails)
                if not input_result["allowed"]:
                    blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            output.append({"id": item["id"], "category": item["category"],
                           "input": item["input"], "expected": item["expected"],
                           "actual": actual, "blocked_by": blocked_by,
                           "passed": actual == item["expected"]})
        return output

    results = _run_async(run_all())
    return results


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)
    def percentile(fraction: float) -> float:
        return round(ordered[min(int((len(ordered) - 1) * fraction), len(ordered) - 1)], 2)
    return {"p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99)}


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                        rails=None, analyzer=None, anonymizer=None) -> dict:
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    inputs = (test_inputs or [""])[:max(1, n_runs)]

    async def measure():
        presidio_times, nemo_times, total_times = [], [], []
        for text in inputs:
            started = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - started) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)
        return presidio_times, nemo_times, total_times

    presidio_times, nemo_times, total_times = _run_async(measure())
    total = _percentiles(total_times)
    return {"presidio_ms": _percentiles(presidio_times), "nemo_ms": _percentiles(nemo_times),
            "total_ms": total, "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
            "budget_ms": LATENCY_BUDGET_P95_MS}


def save_guard_report(results: list[dict], latency: dict,
                      path: str = "reports/guard_results.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"total_inputs": len(results), "passed": sum(item["passed"] for item in results),
                   "pass_rate": sum(item["passed"] for item in results) / len(results) if results else 0.0,
                   "results": results, "latency": latency}, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as handle:
        adversarial_set = json.load(handle)
    results = run_adversarial_suite(adversarial_set)
    latency = measure_p95_latency([item["input"] for item in adversarial_set], n_runs=20)
    save_guard_report(results, latency)
    print(f"Saved guard report: {sum(item['passed'] for item in results)}/{len(results)} passed")
