# How TRACER works

TRACER learns a task-specific classifier from teacher-labelled traces. This
local student acts as a System 1 path: it predicts labels directly using
classical ML or a shallow neural classifier. An acceptance
rule and distance guard determine which student answers the caller can use.
For other inputs, the caller keeps using its teacher. This is fixed-label
classification, not free-form generation or zero-shot selection among models.

Embedding, local inference, and fallback calls all have costs; savings and
latency must be measured on the intended workload. See the
[overview](system1.md) for scope and the [API](api.md) for contracts.

## Training and policy selection

The input embeddings are precomputed. TRACER does not fine-tune the encoder.
The supplied embedding model must remain the same at fit and inference time.

First, `fit()` reserves a random 20% of rows for final certification. These rows
are excluded from classifier training, threshold selection, OOD reference data,
and candidate ranking. This split is not stratified, so its label proportions
are not deliberately rebalanced.

On the remaining development data, TRACER:

1. Optionally subsamples for fitting, using the configured `max_fit_labels`.
2. Splits the fit buffer approximately 60/20/20 into training, validation and
   threshold-selection data, with small-class exceptions.
3. Selects a surrogate by validation macro-F1 against teacher labels.
4. Trains an acceptor on validation correctness, using top-1 probability, top-2
   probability, their margin and normalized entropy.
5. Searches acceptance thresholds on development calibration data.
6. Selects the candidate with highest development coverage at the requested
   agreement target, breaking ties by agreement and then fewer stages.

The acceptor is logistic regression. If its correctness targets have only one
class, the classifier's maximum probability is used as a ranking signal.
`accept_score` is not a certified probability that an individual answer is right.
The teacher can also be wrong: matching its label and matching an independently
established ground-truth label are different evaluation targets.

## Candidate families

- **Global:** one surrogate accepts all inputs before the distance guard.
- **L2D:** one surrogate and an acceptance threshold.
- **RSB:** a two-stage cascade. The second surrogate is trained on examples the
  first stage rejects, when enough residual training and selection data exist.

Each stage chooses from logistic regression, SGD logistic, shallow MLPs,
decision trees, random forests, ExtraTrees, gradient boosting on smaller fit
buffers, and optional XGBoost. Logistic `C=10` means **less** regularization than
`C=1`. See `src/tracer/fit/surrogate.py` for the configured candidates.

`frontier.json` records this adaptive selection process. Its bounds and metrics
are development diagnostics; they are not final certification.

## Distance guard and final check

The distance guard compares a query's mean distance to its nearest development
embeddings against fitted global/per-label thresholds. It is a heuristic for
unfamiliar inputs, not a proof of distribution-shift detection. Its default
quantile is 99.5%. The reference embeddings are saved separately in
`ood_reference.npy`; the full dataset remains stored for future updates.

After selecting one policy, TRACER freezes its classifiers, acceptance
thresholds and distance guard. It then evaluates this exact serving path **once**
on the reserved rows. It counts accepted examples and teacher matches and
computes a one-sided Clopper–Pearson lower bound. The policy is published only
if this bound clears `target_teacher_agreement` and measured coverage clears
`min_deploy_coverage`. There is no retry of other candidates on those outcomes.
Zero accepted examples cannot pass the check.

At inference, ordinary rejection and an OOD distance-guard rejection both return
`decision="deferred"`. The public result does not expose a separate OOD reason.
Without a fallback callback, no teacher is called and `label` is `None`. With a
callback, its result fills `label` but the decision remains `"deferred"`. Batch
prediction and the bundled HTTP server never execute that callback for you.

The defaults are `certification_fraction=0.20` and `certification_alpha=0.10`.
For a fixed policy under independent, identically distributed sampling, the
bound has at least 90% one-sided coverage at that alpha. It bounds agreement
with the teacher on accepted traffic, not agreement with ground truth.

This statement does **not** cover correlated sessions, duplicates across data
partitions, arbitrary future distribution shift, per-class quality, or adaptive
repeated attempts after inspecting reserved outcomes. Random row splitting does
not solve session leakage. Evaluate session/time/workflow holdouts separately,
and collect representative traffic rather than only old-policy deferrals.

Small datasets and strict targets may produce no published policy even when
every reserved prediction is correct. At alpha 0.10, at least 22 accepted,
all-correct examples are needed for a 0.90 lower bound. More errors or a higher
target require more evidence.

## Artifacts, updates and reports

`manifest.certification` contains the final sample counts, coverage, agreement,
lower bound and result. The legacy names `coverage_cal` and
`teacher_agreement_cal` now report final certification measurements for new
artifacts. Older artifacts keep their original meaning and have no certificate.
They remain loadable; loading does not retroactively certify them.

`fit()` and `update()` stage complete artifact generations before publishing.
A fitting or publication exception preserves the old generation. A completed
fit that fails certification instead publishes a null-policy manifest; it does
not keep the old policy active on disk. Use a separate candidate directory when
you need explicit promotion. Use one writer per directory and reload readers
after publication. A required distance guard
that is missing or unreadable prevents loading; it is not silently disabled.

Updates can increase or decrease coverage. The qualitative report describes
routing on the supplied traces, including development rows, so it is not an
independent quality estimate. Use the final certificate and external evaluation
for that purpose. A pre-training `scan()` reports cluster diagnostics and does
not certify the learned serving policy.
