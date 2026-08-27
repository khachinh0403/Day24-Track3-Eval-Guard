# LLM Judge Bias Report

## Local verification

`reports/judge_results.json` contains 10 labelled examples evaluated by the implemented swap-and-average flow using the configured LLM judge.

| Metric | Result |
|---|---:|
| Judged examples | 10 |
| Cohen's kappa | 0.000 |
| Position bias rate | 0.0% |
| Verbosity bias rate | 100.0% |

The zero position-bias result is limited by the small evaluation sample. The high verbosity result and kappa of 0.0 show that a reference answer wins nearly every pairwise comparison; pairwise winner alone is too strict to turn into a binary quality label. Production calibration should request a direct pass/fail quality score against the reference, while retaining pairwise swap-and-average for ranking candidates.

## Production procedure

Run `pairwise_judge()` with a real `OPENAI_API_KEY` for the ten human-labelled answers, retain both orderings, convert pass two back to the original A/B coordinate system, and compute kappa from final labels. Treat a kappa below 0.6 as insufficient agreement for an automatic quality gate. Keep the raw reasons and scores as artifacts so disagreements can be audited.

Swap-and-average remains mandatory because a single pairwise pass can favor answer order. Track both position inconsistency and the rate at which the longer answer wins; if either rises after a prompt/model update, hold the release and review a stratified sample.
