# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Student:** Le Kha Chinh
**Status:** implementation, full RAGAS evaluation, and local guard verification complete.

## Guard Stack Pipeline

| Layer | Tool | Measured P95 | Failure action |
|---|---|---:|---|
| PII detection | Presidio + VN regex recognizers | 13.31 ms | Reject request, redact PII, security log |
| Topic/jailbreak | Local policy pre-filter + NeMo input rail | 0.20 ms local path | 403 and safe refusal |
| RAG pipeline | Day 18 hybrid retrieval + rerank | Not measured yet | Return controlled fallback |
| Output check | Local sensitive-output policy + NeMo output rail | Not measured yet | Replace with safe refusal and log |
| **Input guard total** | Presidio + input rail | **13.34 ms** | **Within 500 ms budget** |

The local pre-filter is deliberately before NeMo. It blocks known prompt-injection and PII-exfiltration patterns even if the LLM provider is unavailable; NeMo provides a second semantic layer when credentials are configured.

## CI Gates

- [ ] RAGAS faithfulness >= 0.75 on the 50-question set.
- [x] Adversarial guard suite >= 90%: **20/20 (100%)**.
- [x] Input guard P95 < 500 ms: **13.34 ms**.
- [x] `python -m pytest tests/ -v`: **40 passed**.

```yaml
- name: RAG quality
  run: python src/phase_a_ragas.py
- name: Guardrail suite
  run: python -m pytest tests/test_phase_c.py -v
- name: Guard latency
  run: python src/phase_c_guard.py
```

## Monitoring

| Metric | Alert threshold | Action |
|---|---|---|
| Faithfulness daily sample | < 0.70 | Inspect retrieval and prompt changes |
| Adversarial block rate | < 90% | Add attack signatures and regression tests |
| Guard P95 | > 500 ms | Profile Presidio/NeMo and scale model service |
| PII detections | >10 per hour | Security alert and audit source |

## Current Evidence

| Metric | Result |
|---|---|
| RAGAS 50q | Avg 0.796; factual 0.871, multi-hop 0.750, adversarial 0.785 |
| Cohen's kappa | 0.286 from deterministic no-key fallback; rerun with LLM judge before release |
| Adversarial pass rate | 20 / 20 |
| Guard P95 latency | 13.34 ms |

For production, run the full RAGAS and judge jobs with managed credentials in CI, save the generated JSON reports as artifacts, and promote only builds that meet the quality gates above.
