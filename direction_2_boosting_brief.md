# Progressive Partitioned Inference — Direction 2: Boosting Reformulation

**FML Node Project**
**May 2026**

---

## Context

Four partition training variants have been implemented and evaluated on CASIA-WebFace / LFW (see `variant_analysis_brief.md`). All four fail to achieve strict P012 > P01 > P0 monotone improvement under cosine-similarity verification on per-partition L2-normalised concatenated embeddings.

The shared failure mode operates at two stacked levels:

1. **Backbone bias** — a shared backbone trained with any partition as primary objective allocates representational capacity toward that partition's needs, leaving subsequent partitions to operate on a substrate already shaped against them.
2. **Combination metric** — cosine similarity on concatenated, per-partition L2-normalised embeddings approximately averages the partition-wise cosine similarities. Adding a partition with weak signal degrades the average, regardless of whether the partition is geometrically orthogonal to P0.

The deployment story does not require P0 to be competitive with the centralised baseline — it requires P0 to be **functional** as a local-only fallback. The N-node configuration is the comparison target for the centralised baseline; P0 is the comparison target for "no system at all" or "mandatory video uplink to the datacentre." This reframe lowers the floor that must be defended.

Direction 2 is the structural redesign branch, developed in parallel with Direction 1 (learned combiner diagnostic, separate brief). The two directions are complementary rather than alternative, but Direction 2 is now specified to stand independently of Direction 1's outcome.

---

## Hypothesis

Replacing geometric orthogonality with content-based complementarity — training each subsequent partition specifically on the verification pairs the previous partitions fail on — produces non-redundancy by construction rather than by proxy. P1 becomes informative on exactly the cases P0 isn't, where "informative" is measured against the actual verification objective rather than a representational geometry.

---

## Why This Addresses the Upstream Problem

The four variants attempted to enforce non-redundancy through mechanisms downstream of the backbone's representational allocation:

- **Variant A:** post-hoc orthogonality loss
- **Variant B:** structural width assignment (slimmable)
- **Variant C:** phase-gated freeze with separate ArcFace slots
- **Variant D:** orthogonality against frozen reference partitions

In every case, the constraint operated on a representation whose discriminative geometry had already been shaped by P0's training requirements. Geometric orthogonality without a discriminative-content signal pushes new partitions toward the *low-discriminability* region of the representation space — exactly the directions P0 didn't need.

Boosting reformulates the objective itself. Each partition's training signal comes from the residual error of the previous partitions on the actual evaluation task. The backbone-bias problem doesn't disappear — but the loss now explicitly rewards the partition for capturing information the previous partitions missed, on the metric that matters. The semantic gap between "geometrically orthogonal" and "discriminatively complementary" closes by construction.

---

## Design Decisions and CLI Switches

All four open design questions are resolved as command-line switches with sweepable hyperparameters. This makes each decision an experimental axis rather than a fixed choice, and supports systematic ablation without code changes between runs.

### 1. Backbone state during P1/P2 phases

`--backbone-state {frozen, fine_tuned, partial}`

- `frozen` — backbone fully frozen during P1/P2 phases (Variant C-style behaviour, included for direct ablation against existing results).
- `fine_tuned` — backbone fully trainable with low learning rate during P1/P2 phases. Requires `--backbone-lr-multiplier FLOAT` to scale the backbone learning rate relative to partition heads (default 0.1).
- `partial` — early stages frozen, late stages trainable.

For `partial`, additional argument `--frozen-stages INT` controls how many early ResNet-50 stages are frozen. ResNet-50 has six freezable groups (stem + 4 residual stages + final pooling/projection). Range 0–5 with sensible default at 3 (stem + first two residual stages frozen, last two stages and projection adaptive).

### 2. Hard pair mining

`--mining-strategy {band, topk}`

- `band` — pairs whose previous-ensemble cosine score falls in an ambiguous band. Sweepable parameters `--mining-band-low FLOAT` and `--mining-band-high FLOAT` (floats in [-1, 1]; defaults capturing the typical genuine/impostor confusion zone, e.g. [0.2, 0.6]).
- `topk` — top-k hardest pairs by previous-ensemble verification loss. Sweepable parameter `--mining-topk INT` (default e.g. 10% of batch).

Mining is dynamic by default. `--mining-refresh-every INT` controls recomputation frequency in training steps (default e.g. 500 steps; setting it very high approximates static mining). Recomputation honours the boosting principle — as the current partition trains and the backbone (if not frozen) shifts, the failure distribution of the previous ensemble's effective outputs evolves, and the training set should track it.

### 3. Loss formulation

`--loss {arcface_reweighted, arcface_margin, triplet, contrastive, sub_center_arcface}`

Each loss carries its own hyperparameter group:

- `arcface_reweighted` — ArcFace with per-pair weights from the previous ensemble's confidence. Acknowledged geometric assumption violation; included for completeness.
- `arcface_margin` — margin-based loss on hard pairs with ArcFace as regulariser on easy pairs. Two-loss objective with `--easy-loss-weight FLOAT` (default 0.3).
- `triplet` — triplet loss with anchors, positives, and negatives mined from hard pairs. Parameters `--triplet-margin FLOAT` (default 0.3), `--triplet-mining {batch_hard, semi_hard}`.
- `contrastive` — contrastive loss with hard pair sampling. Parameter `--contrastive-margin FLOAT` (default 1.0).
- `sub_center_arcface` — sub-center ArcFace variant. Parameter `--sub-center-K INT` (default 3).

Triplet is the instinct default given the verification-task framing — it optimises pairwise relative ordering directly, which matches the eval objective more closely than classification-derived losses. Worth running the sweep to confirm.

### 4. Combination at inference

`--combination {cosine_concat, confidence_weighted, learned_combiner}`

- `cosine_concat` (default) — cosine similarity on concatenated, per-partition L2-normalised partition outputs, padded for absent partitions. Same as the original eval; baseline behaviour.
- `confidence_weighted` — per-partition similarity scores combined with weights derived from per-partition confidence signals. Confidence source itself parameterised:
  - `--confidence-source {embedding_norm, cosine_magnitude, scalar_head}`
  - `embedding_norm` — pre-normalisation embedding magnitude (proxy for representation quality).
  - `cosine_magnitude` — magnitude of the partition's cosine similarity score (high-confidence partitions have decisive scores either way).
  - `scalar_head` — tiny per-partition learned scalar head producing a confidence score from the embedding. Adds parameters but is the most direct.
- `learned_combiner` — inherits Direction 1's combiner architecture if D1 succeeds. Available as a post-hoc evaluation option without retraining the partitions, since the combiner sits over frozen partition outputs.

The default `cosine_concat` makes Direction 2 evaluable on its own terms, independent of D1's outcome. `confidence_weighted` is the natural Direction-2-native variant. `learned_combiner` becomes available as a third option if and when D1's combiner is validated.

---

## Compute Envelope

A full Direction 2 implementation is a Stage 1-class effort: ~48 hours per CASIA training run, multiple runs for hyperparameter selection and ablation across the four CLI axes. Predictive smoke-testing on a smaller dataset before committing to full CASIA runs is necessary given iteration cost.

Phasing:

1. **Smoke test** on alternate testbed (CIFAR-100 hierarchy or CelebA, see below) — establishes whether the boosting mechanism produces strict monotone improvement under any configuration.
2. **CASIA subset run** — partial dataset (e.g. 100k images, 2k identities) to gate full-scale runs and narrow the hyperparameter grid.
3. **Full CASIA training** for the candidate configuration identified by the subset run.
4. **Ablation runs** along each CLI axis at full CASIA scale, scoped to whichever decisions remain consequential after the candidate run.

---

## Testbed Options

Two alternatives for smoke-testing and CASIA gating, both lighter than full CASIA:

- **CIFAR-100 superclass/subclass.** Explicit hierarchical structure gives the boosting mechanism a natural target — P0 predicts superclass, P1 refines to subclass on P0's hard cases, P2 boosts further on P0+P1's residual errors. Cleanest possible test of the mechanism in isolation. Lowest training cost. Furthest from face domain — failure to transfer is informative but limits the narrative if face verification proves intractable.
- **CelebA multi-attribute classification.** Partitions correspond to attribute clusters with explicit labels. Boosting validated on attributes that are partially independent before face verification transfer. Closer to face domain than CIFAR; lower training cost than full CASIA. Requires adaptation since CelebA is classification rather than verification — either reframe as multi-attribute boosting end-to-end, or use it for backbone pretraining and then transfer to a verification head.

**Recommendation:** CIFAR-100 hierarchy as the first smoke test — it isolates the mechanism in the cleanest setting available. If it succeeds there, CelebA for transfer-domain validation before committing to full CASIA. If the mechanism works on CIFAR-100 but fails on CelebA, that constrains the domain in an interpretable way.

The sensor-bug modality is dropped from this brief — the partition-as-specialist mapping doesn't fit the architecture cleanly enough in this context to justify the dataset definition and backbone training overhead.

---

## Relationship to Direction 1

Direction 2 is now specified to stand independently of Direction 1's outcome. The default combination is `cosine_concat`, matching the original eval and providing a clean comparison against existing variant results. If Direction 1's combiner succeeds, it becomes available via `--combination learned_combiner` as a post-hoc evaluation option — no retraining of Direction 2 partitions required, since the combiner sits over frozen partition outputs.

Direction 1's verdict on orthogonality (whether Variant D's orth contribution was salvageable) is informative but not gating for Direction 2: orth is not part of Direction 2's design space at all. The boosting reformulation replaces orthogonality with content-based complementarity from the start.

---

## Implementation Notes for Claude Code Handoff

- All hyperparameters exposed as CLI args. Config file (YAML) supported with CLI args overriding config — supports both reproducible config-driven runs and quick CLI sweeps.
- Wandb logging of all CLI args, training curves, per-phase loss components, hard pair statistics (number mined per refresh, score distribution), and per-subset eval metrics.
- Seeds set deterministically; cudnn.deterministic=True for reproducibility. Note any unavoidable non-determinism in mining or sampling.
- Phase boundaries logged explicitly. Hard-pair refresh events logged with timestamp and resulting pair count.
- Checkpoints saved per phase, separately for backbone and partition heads, to enable mid-training analysis and reuse with Direction 1's combiner.
- Eval grid: all seven non-empty partition subsets × all combination strategies × LFW (+ CFP-FP, AgeDB-30 if cheap). Single eval pass produces the full table.

---

## Practical Constraints

- **GPU cost.** ~48 hours per full CASIA run. Iterative ablation is expensive; smoke tests on CIFAR-100 / CelebA gate expensive runs.
- **Reproducibility.** All results reproduce from config + seed. CLI args fully captured in wandb run config.
- **Subset agnosticism.** Any P0-anchored subset must remain valid at inference. The boosting mechanism preserves this regardless of training order — partitions are never width-indexed at inference.
- **N-agnosticism.** Boosting generalises by extension — each new partition boosts on the previous ensemble's hard pairs. Phase 3 N-scaling experiments (N=5, N=7) require the mechanism to not be hardcoded to N=3. CLI design supports `--num-partitions N` natively.
- **Single backbone forward pass.** Preserved at inference. Boosting modifies training signal, not inference architecture. All partitions remain projections off a shared backbone forward pass.

---

## Decision Points

- **Before smoke test:** Pick a starting CLI configuration. Suggested initial config: `--backbone-state partial --frozen-stages 3 --mining-strategy topk --mining-topk 0.1 --mining-refresh-every 500 --loss triplet --combination cosine_concat`.
- **After smoke test (CIFAR-100):** If the mechanism produces strict monotone improvement under at least one configuration, advance to CelebA. If not, return to CLI sweep on CIFAR-100 before scaling.
- **After CelebA validation:** If CIFAR-100 succeeds and CelebA fails, decision required on whether to commit CASIA compute or pivot the testbed framing in Paper 1. If both succeed, advance to CASIA subset run.
- **After CASIA subset run:** Narrow CLI grid to the most promising configurations for full CASIA.
- **After full CASIA run:** Compare against Variants A–D as ablations and against the centralised ArcFace baseline. P012 > P01 > P0 monotone improvement is the success criterion; P0 ≥ functional floor is the deployment criterion. Both should hold for the architecture to support the broader project narrative.
