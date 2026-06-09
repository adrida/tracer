# How TRACER Works

**TRACER** = **T**race-Based **A**daptive **C**ost-**E**fficient **R**outing

## The core idea

You're running an LLM to classify text (intents, harm categories, topics, ...). Most inputs are "easy" -- a cheap ML model could get them right. But you don't know *which* inputs are easy until you try, and you don't want silent quality degradation.

TRACER learns that boundary from your LLM's own outputs. It finds which inputs the LLM handles predictably, builds a fast local surrogate for those, and defers the rest back to the LLM. Coverage grows as you collect more traces.

---

## The pipeline

```
Input text
  │
  ▼
[Embedding]           ← frozen, precomputed
  │
  ▼
[Surrogate model]     ← trained on teacher labels
  │  predict_proba(x)
  ▼
[Acceptor gate]       ← predicts: will surrogate agree with teacher?
  │
  ├── accept_score ≥ τ  →  return surrogate prediction  (FREE)
  └── accept_score < τ  →  defer to teacher LLM          (PAID)
```

### The surrogate

A classifier trained on the teacher's outputs (not ground-truth labels). The surrogate learns to imitate the teacher, not to solve the task from scratch. Simple models (logistic regression, MLP, random forest) work well because the embedding already captures the semantics.

### The acceptor

A second binary classifier that predicts: "will the surrogate's prediction match the teacher on this input?" It's trained on the same data: features are the surrogate's confidence signals (top-1 prob, top-2 prob, margin, entropy). A high acceptor score means "the surrogate is confident and consistent with the teacher here."

### Threshold calibration

The acceptor threshold τ is calibrated on the calibration split (not used for training): find the threshold that maximizes coverage subject to teacher agreement ≥ target. Because τ is *selected* on that split, the agreement measured there is an optimistic, in-sample estimate — so it is **not** the number used to decide deployment. The deploy gate re-measures agreement on a separate held-out `test` split that played no part in surrogate selection or threshold calibration (see [The parity gate](#the-parity-gate)). The approach follows the post-hoc deferral estimator framework of Narasimhan et al. ([ICML 2022](https://proceedings.mlr.press/v162/narasimhan22a.html)).

---

## Three pipeline families

TRACER fits three candidate architectures and picks the one with highest coverage at your target teacher agreement. The L2D and RSB families are grounded in the Learn-to-Defer literature (Mozannar & Sontag, [ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html); Madras et al., [NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/09d37c08f7b129e96277388757530c72-Abstract.html); Mao et al., [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7e1bf09d2e4e6e5c5a0b63b66b47ef42-Abstract.html)).

### Global

```
[Surrogate] → accept all
```

The simplest case. If the surrogate achieves the target teacher agreement on everything (no gating needed), use it. This gives 100% coverage when it works.

### L2D (Learn to Defer)

```
[Surrogate] → [Acceptor gate] → LOCAL or LLM
```

The standard case. The acceptor gates each sample individually. High-confidence inputs are handled by the surrogate; uncertain ones are deferred. Selected for most tasks.

### RSB (Residual Surrogate Boosting)

```
[Surrogate 1] → [Acceptor 1] → LOCAL (easy traffic)
                            ↓
               [Surrogate 2] → [Acceptor 2] → LOCAL (medium traffic)
                                           ↓
                                          LLM (hard traffic)
```

A two-stage cascade. Stage 2 is trained specifically on the inputs that Stage 1 deferred -- the residuals. This gives Stage 2 a specialized view of the "hard" cases and can push coverage higher. Only used when there are enough residual training samples (≥50).

---

## Model zoo

The surrogate model is selected by validation F1 from a pool of candidates. Tree-based models don't use StandardScaler; linear/neural models do.

| Model | Family | When it wins |
|-------|--------|-------------|
| `logreg_c1` | Linear | High-dim, linearly separable embeddings |
| `logreg_c10` | Linear | Same, with tighter regularization |
| `sgd_log` | Linear | Very large datasets (faster than logreg) |
| `mlp_1h` | Neural | Non-linear boundaries, medium data |
| `mlp_2h` | Neural | Complex boundaries, larger data |
| `dt` | Tree | When decision boundaries are axis-aligned |
| `rf` | Tree | Robust ensemble baseline |
| `et` | Tree | Faster than RF, often similar quality |
| `gbt` | Tree | Slower, strong on small datasets (≤4k) |
| `xgb` | Tree | Boosted, optional (install `tracer-llm[xgboost]`) |

---

## Data splits

```
All traces (n)
  │
  ├── Subsample to max_fit_labels (8000) if n > 8000  [stratified]
  │
  ├── 55%  Training      → fit surrogate, fit acceptor
  ├── 15%  Validation    → select best surrogate by val F1
  ├── 15%  Calibration   → calibrate acceptor threshold τ
  └── 15%  Test          → held-out parity gate (deploy decision)
```

Splits are stratified by label. Classes with fewer than 4 samples use a fallback to ensure representation. The `test` split is deliberately disjoint from both surrogate selection (`val`) and threshold calibration (`cal`) so the deploy gate is judged on data it has never seen.

---

## The parity gate

Before any surrogate handles live traffic, TRACER checks: does it actually meet the target teacher agreement on the held-out `test` split — data used for neither surrogate selection nor threshold calibration? This is the parity gate.

If the held-out teacher agreement < target → `selected_method = None`, coverage = 0%. Traffic continues 100% to the teacher. No silent degradation.

The held-out estimate is what guards deployment precisely because the calibration-split agreement is biased upward: τ was chosen to maximise coverage at the target *on that split*, so re-reading agreement there rewards the noise the selection exploited. The candidate summaries keep both numbers (`teacher_agreement_cal_insample` vs the held-out `teacher_agreement_cal_total`) so the gap is visible in `frontier.json`.

### Optional finite-sample certificate

By default the gate compares the held-out *point estimate* against the target. Set `FitConfig(parity_alpha=0.05)` to instead require a one-sided Clopper-Pearson lower bound (at confidence `1 - parity_alpha`) on the held-out agreement to clear the target. Because exactly one threshold is evaluated on the held-out split, this is a genuine finite-sample guarantee with no multiple-comparison correction needed — at the cost of declining to deploy when the held-out split is too small to certify.

---

## Continual learning

Every `tracer.update()` call:
1. Merges new traces with all historical traces
2. Refits from scratch (all three pipeline families)
3. Re-selects best on frontier
4. Re-calibrates threshold
5. Re-checks parity gate

Coverage grows monotonically as more traces accumulate. Easy intents (tight embedding clusters) become self-serve first; ambiguous ones may never fully cross the parity bar.

---

## Qualitative audit

Every fit produces a structured explanation of the routing policy. The slice and boundary-pair design is informed by Slice Finder (Chung et al., [ICDE 2019](https://ieeexplore.ieee.org/document/8731353)) and the contrastive explanation literature (Miller, [AIJ 2019](https://www.sciencedirect.com/science/article/pii/S0004370217301126)).

**Slice summaries**: per-label and per-length-bucket handled/deferred rates. Answers: "which parts of my label space are well-covered?"

**Contrastive boundary pairs**: for each label, one example that was handled and one that was deferred. Answers: "what makes an input 'easy' vs 'hard' for the surrogate?"

**Representative examples**: the most typical handled and deferred inputs (by embedding position). Answers: "what does typical handled/deferred traffic look like?"

**Temporal deltas**: how per-label handled rates changed between the previous fit and this one. Answers: "what's improving and what's plateauing?"

---

## Paper

A research paper is in preparation with formal proofs of the parity guarantees, ablation studies across Banking77, CLINC-150, MNLI, WildGuardMix, and RAGTruth, discussion of limitations, and tooling to reproduce all experiments. Link will appear here upon publication.
