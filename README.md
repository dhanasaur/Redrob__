# Redrob Senior AI Engineer Ranker

CPU-only candidate ranker for the Redrob Intelligent Candidate Discovery challenge. Streams 100K profiles, scores them against the Senior AI Engineer JD, and outputs a validated top-100 CSV with grounded reasoning — no GPU, no network, no external LLM calls.

---

## Quick Start

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

Place `candidates.jsonl` (or `.jsonl.gz`) in this folder before running. For a quick smoke test, convert `sample_candidates.json` to JSONL and run on that instead.

**Colab sandbox:** Open `colab_sandbox.ipynb`, set `REPO_URL` to your GitHub repo, and run all cells.

---

## Process

```
candidates.jsonl  →  feature extraction (20 signals + risk)
                  →  pseudo-relevance labels (0.0–4.0)
                  →  tree regressor (80/20 stratified split)
                  →  post-hoc risk/trap penalty tuning
                  →  top-100 rank + factual reasoning  →  submission.csv
```

1. **Extract** — Stream each candidate once; build 20 numeric features from profile, career, skills, and Redrob behavioral signals, plus a risk score and honeypot flag.
2. **Label** — Assign a continuous pseudo-relevance score from JD-weighted heuristics (no manual labels on 100K rows).
3. **Train** — Fit a gradient-boosted tree regressor on 80% of the pool; hold out 20% stratified by label bin.
4. **Tune** — Grid-search risk and honeypot penalties on the holdout to maximize NDCG/AP while keeping trap rate below 10%.
5. **Output** — Select top 100, scale scores monotonically, and write CSV reasoning from actual profile evidence.

---

## Algorithms & Why

| Component | What | Why |
|---|---|---|
| **Rule-based features** | 20 scores: archetype, semantic/title match, career evidence, skill depth, behavior, logistics, trust, retrieval depth, production/eval/vector-DB signals, experience band, hiring readiness, stability, credibility, company quality, NLP/IR focus | Captures JD-specific IR/ranking signals interpretably; runs offline on CPU without embeddings or API calls |
| **Pseudo-labeling** | Heuristic function maps features → continuous relevance (0–4) | Provides a supervised target when ground-truth labels are unavailable at scale |
| **Gradient boosted trees** | LightGBM (preferred) → XGBoost → Random Forest fallback | Learns non-linear feature interactions; fast, deterministic, strong on tabular data |
| **Stratified holdout** | 80/20 split on binned pseudo-labels | Validates ranking quality without leaking train data into penalty tuning |
| **Ranking metrics** | NDCG@10/50, average precision, precision@10 minus trap rate | Optimizes ordered list quality — the actual submission format |
| **Post-hoc penalties** | Subtract tuned weights × risk / hard-trap flags from model scores | Challenge disqualifies submissions with >10% honeypots in top 100; explicit guard against keyword-stuffed or inconsistent profiles |
| **Grounded reasoning** | Template fills from matched terms, skills, logistics — not LLM-generated | Satisfies explainability requirement using facts already in the candidate record |

---

## Repository Layout

```
mlmodel/
├── README.md                 # This file (sole documentation)
├── rank.py                   # Main ranker
├── validate_submission.py    # CSV format validator
├── requirements.txt          # Python dependencies
├── submission_metadata.yaml  # Team & reproducibility metadata
├── colab_sandbox.ipynb       # Colab demo on sample data
├── candidate_schema.json     # Candidate profile schema
├── sample_candidates.json    # 50-candidate test pool
├── sample_submission.csv     # Expected CSV format
└── .gitignore
```

Large/generated files (`candidates.jsonl`, `submission.csv`, diagnostics) are gitignored.

---

## Requirements

- Python 3.10+, 8 GB RAM, any modern CPU
- Fully offline at ranking time
