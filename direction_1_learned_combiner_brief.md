# Progressive Partitioned Inference — Direction 1: Learned Combiner Diagnostic

**FML Node Project**
**May 2026**

---

## Context

Four partition training variants have been implemented and evaluated on CASIA-WebFace / LFW (see `variant_analysis_brief.md`). All four fail to achieve strict P012 > P01 > P0 monotone improvement under cosine-similarity verification on per-partition L2-normalised concatenated embeddings.

The shared failure mode operates at two stacked levels:

1. **Backbone bias** — a shared backbone trained with any partition as primary objective allocates representational capacity toward that partition's needs, leaving subsequent partitions to operate on a substrate already shaped against them.
2. **Combination metric** — cosine similarity on concatenated, per-partition L2-normalised embeddings approximately averages the partition-wise cosine similarities. Adding a partition with weak signal degrades the average, regardless of whether the partition is geometrically orthogonal to P0.

The deployment story does not require P0 to be competitive with the centralised baseline — it requires P0 to be **functional** as a local-only fallback. The N-node configuration is the comparison target for the centralised baseline; P0 is the comparison target for "no system at all" or "mandatory video uplink to the datacentre." This reframe lowers the floor that must be defended.

Direction 1 is a cheap diagnostic running over existing checkpoints. Direction 2 (boosting reformulation, separate brief) is developed in parallel as a structural redesign branch.

---

## Hypothesis

The trained partitions in Variants C and D may carry combinable identity information that linear cosine averaging destroys at evaluation time. Replacing the combination function with a learned, mask-conditional projection should recover any such signal — or confirm its absence.

The diagnostic isolates the combination problem from the partition-training problem. If the combiner extracts joint signal from existing checkpoints, the upstream training mechanisms work and the eval was the bottleneck. If it doesn't, the partitions are genuinely uninformative and Direction 2 proceeds with high confidence about what it's solving.

---

## Architecture

A small MLP combiner taking concatenated raw partition outputs plus a presence mask, producing a single combined embedding that is L2-normalised at output:

- **Input:** `[P0 | P1 | P2]` (3K-dim, zero-padded for absent partitions) concatenated with 3-bit presence mask `m ∈ {0,1}³`.
- **Hidden:** Two layers, GELU activation, no batch norm (subset-agnostic), light dropout (0.1).
- **Output:** K_out-dim combined embedding, L2-normalised.
- **Parameters:** Order of 100K — tiny relative to the backbone.

Two K_out values run as a small ablation:

- K_out = K — matches P0-alone dimensionality, apples-to-apples comparison.
- K_out = 3K — preserves capacity, tests dimensionality reduction as part of the problem.

The presence mask is included explicitly because zero-padding alone is ambiguous — the combiner cannot distinguish "absent" from "present and near zero" without it. With three partitions there are seven non-empty subsets; the mask makes subset identity a first-class input.

---

## Training Protocol

- **Frozen partition checkpoints.** Variant C and Variant D final checkpoints are loaded and frozen. Only the combiner is trained.
- **Loss:** ArcFace (s=64, m=0.5) on the combined embedding, single weight slot. Aligned with the original training to keep any improvement attributable to the combiner.
- **Subset sampling:** Each training example is assigned a uniformly-sampled non-empty subset, with the corresponding mask applied before combiner forward pass. This forces the combiner to handle all subsets in training rather than only the full triple.
- **Cached embeddings:** Backbone forward passes are precomputed once over CASIA and cached. Each training epoch becomes combiner-only forward and backward — total compute is hours, not days.
- **Convergence:** A few epochs. If unconverged at epoch 5, abandon.

A second combiner is trained on full-triple inputs only and evaluated on subsets, as a robustness control. Comparing uniform-subset training against full-triple-only training tests whether explicit subset exposure is necessary for joint signal extraction.

---

## Evaluation

Verification on LFW (CFP-FP and AgeDB-30 if cheap). For each pair, compute combined embeddings via the combiner at the relevant subset, then cosine similarity. Report accuracy and TAR@FAR=1e-3 for all seven non-empty subsets, both checkpoints, both K_out settings.

Critical side-by-side comparison: combiner output vs cosine-on-concatenated baseline (the existing eval). The single-partition cosine baseline (P0 alone) defines the functional floor — the combiner's success criterion is "P012 via combiner ≥ P0 cosine, ideally exceeds it," not "beats P012 cosine."

Auxiliary logging:

- Per-subset accuracy and TAR@FAR for combiner vs cosine baseline, both checkpoints
- Combiner training curves
- Cosine similarity between combiner outputs on different subsets of the same identity (graceful-degradation check — if `combine({P0,P1,P2})` and `combine({P0})` are highly aligned for the same person, the combiner preserves identity geometry across subsets and supports gallery indexing)

---

## Diagnostic Outcomes

| Outcome | Interpretation | Next move |
|---|---|---|
| Combiner rescues C only | Partitions carry combinable signal; orth in D was harmful or wasted | Drop orth from Direction 2 design |
| Combiner rescues D only | Orth created real geometric separation that averaging couldn't access | Keep orth as candidate mechanism in Direction 2 |
| Combiner rescues both | Combination was the bottleneck; architectural variants matter less than assumed | Combiner becomes default eval; Direction 2 inherits it |
| Combiner rescues neither | Partitions are genuinely uninformative | Direction 2 with high confidence about what it's solving |

---

## Compute Envelope

Hours on a single GPU. Embeddings precomputed once (~1 hour over CASIA). Combiner training: a few epochs at minutes per epoch. Total: less than one day end-to-end. Both checkpoints (C and D) and both K_out settings fit within a single working session.

---

## Testbed Shift Options

The diagnostic is already cheap, but expanding the eval grid sharpens the signal at trivial cost:

- **CFP-FP and AgeDB-30 alongside LFW.** LFW has known ceiling effects. Pose variation (CFP-FP) and age variation (AgeDB-30) make the verification task harder and give complementary signal more to contribute. Eval-only, trivial cost, worth running by default.

The face recognition path is the principal experiment. CFP-FP and AgeDB-30 expand the eval grid without changing the diagnostic structure.

---

## Practical Constraints

- **Reproducibility:** Results reproduce from config + seed. No non-determinism in the combiner pipeline.
- **Subset agnosticism:** Any P0-anchored subset must remain valid at inference. The combiner explicitly trains for this via uniform subset sampling.
- **N-agnosticism:** The combiner generalises trivially to N>3 — input dimension scales, mask scales. No architectural change required for later N-scaling experiments.
- **Single backbone forward pass:** Preserved. The combiner is a post-backbone projection; nothing about backbone inference changes.

---

## Decision Points

- **On completion:** Commit Direction 2's combination mechanism. If the combiner succeeds, it becomes the default eval and Direction 2 inherits it. If it fails, Direction 2 must specify combination independently.
- **On orth verdict:** If only Variant C is rescued, drop orth from Direction 2's design space. If only D, retain it as a candidate mechanism. If both, the variants matter less than assumed and Direction 2's design space narrows accordingly.
- **On testbed choice:** If LFW results are ambiguous, expand to CFP-FP and AgeDB-30 before declaring the diagnostic inconclusive.
