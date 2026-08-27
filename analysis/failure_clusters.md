# Failure Cluster Analysis

## Aggregate RAGAS scores

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| Faithfulness | 0.933 | 0.518 | 0.800 |
| Answer relevancy | 0.791 | 0.716 | 0.691 |
| Context precision | 0.933 | 0.929 | 0.933 |
| Context recall | 0.825 | 0.838 | 0.717 |
| **Average** | **0.871** | **0.750** | **0.785** |

## Failure clusters

The failure matrix identifies `answer_relevancy` as the most frequent weakest metric (19 questions), while the factual distribution has the most per-question lowest-metric assignments (20). This does not contradict the aggregate scores: factual answers are generally good, but when a metric is weakest it is usually answer relevancy. Multi-hop has the lowest overall average because faithfulness falls to 0.518.

The most serious examples are question 6 (factual, 0.333), questions 21 and 33 (multi-hop, 0.375), and question 50 (adversarial VPN policy, 0.417). Most bottom-10 failures are faithfulness failures, which indicates unsupported or incorrectly combined policy claims rather than broad retrieval noise.

## Recommended remediation

1. Require an explicit evidence-first answer format for multi-hop calculations and policy combinations.
2. Include policy version metadata in retrieval and instruct the generator to prefer current versions.
3. Add a response validator for numeric calculations and approval thresholds.
4. Improve answer instructions to directly address the question before adding background context.

Adversarial average (0.785) is lower than factual (0.871), as expected, but above multi-hop (0.750). Version conflicts and negation traps are still visible in bottom-10, notably the personal-VPN question.
