# Variant C — Phase-gated Residual Boosting

## Context

Variants A and B both failed to produce meaningful partition specialization, by different mechanisms:

- **Variant A (orthogonal).** Soft orthogonality loss on partition outputs was trivially satisfied — random vectors in R^K are near-orthogonal by default, so the aux loss collapsed to ~0 within one epoch and partitions converged to near-identical representations.
- **Variant B (nested prefix + switchable BN).** Structurally forced partition 0 to be the "base" via prefix-only dropout. P0 alone reached 0.657 LFW pair accuracy at epoch 8 (a genuine success — partition 0 became functional standalone). But P012 ≈ P01 ≈ P02 ≈ 0.66, with P1 alone at 0.536 (near-random). Partition 0 absorbed essentially all the signal; partitions 1 and 2 were near-vestigial. The collapse mechanism: during w2/w3 training batches, the model minimized loss by making partitions 1 and 2 small-contribution additions on top of partition 0's already-competent embedding.

The common failure mode across A and B: **no mechanism forces later partitions to carry information that earlier partitions don't already have.** Soft constraints get gamed; structural constraints that only specify ordering (not disjointness of information) don't prevent redundancy.

Variant C addresses this with a hard training-time constraint: freeze earlier partitions while training later ones, so later partitions have no choice but to fit the residual of the committee-so-far.

## Why phase-gated and not architectural residual

Two flavors of "residual" were considered:

- **Phase-gated boosting (this design).** Architecture identical to A/B: all three heads consume the same backbone features in parallel. Specialization comes from sequential training with freezing.
- **Architectural residual.** Partition k+1's forward pass takes partition k's output as input. Stronger guarantee of specialization, but *breaks the any-subset inference property* — partition k cannot be computed without first computing partitions 0..k-1, regardless of whether they appear in the assembled embedding. Losing a partition node mid-chain makes downstream outputs uncomputable. This is incompatible with the PPI thesis of graceful degradation under arbitrary partition loss.

Phase-gated keeps the symmetric forward pass, preserving all 7 partition configs as computable at eval (though with quality degradation for non-prefix configs, which is acceptable).

## Core mechanism

Three phases (plus an optional joint fine-tune). Each phase trains exactly one partition head's parameters plus (in phase 1 only) the backbone.

| Phase | Trainable | Frozen | Embedding during training |
|-------|-----------|--------|---------------------------|
| 1 | Backbone, f_0, ArcFace slot-0 rows | f_1, f_2, ArcFace slots 1/2 | (p_0, 0, 0) |
| 2 | f_1, ArcFace slot-1 rows | Backbone, f_0, ArcFace slots 0/2 | (p_0, p_1, 0) *with training-time subset sampling* |
| 3 | f_2, ArcFace slot-2 rows | Backbone, f_0, f_1, ArcFace slots 0/1 | (p_0, p_1, p_2) *with training-time subset sampling* |
| 4 (opt) | All | — | Full, very short, low LR |

ArcFace head rows are frozen alongside the partition whose slot they serve. Rationale: if f_0 is frozen but ArcFace slot-0 rows aren't, phase 2 can minimize loss by rotating slot-0 rows to compensate for f_1's output, which mangles f_0's standalone semantics and breaks P0 at eval.

## Design decisions and rationale

The decisions below encode specific lessons from A and B. Each lesson is tied to the failure it addresses.

### 1. Do not train phase 1 to convergence

**Lesson:** Variant B's collapse root cause. If f_0 saturates the training distribution alone, there is no residual for later phases to fit, and partitions 1 and 2 will end up low-norm.

**Design:** Phase 1 uses a hard epoch budget that is a *fraction* of total training, not a loss-based stopping criterion. Roughly 50–60% of total budget for phase 1, diminishing for subsequent phases (residual fitting is easier work).

**Example schedule for 24 total epochs:**
- Phase 1: 12 epochs
- Phase 2: 7 epochs
- Phase 3: 4 epochs
- Phase 4 (fine-tune, optional): 1 epoch

### 2. Zero-init the tail of f_1 and f_2

**Lesson:** At phase transition, newly-unfrozen heads are randomly initialized. Their random output corrupts the assembled embedding — the ArcFace head has never seen slot-1 or slot-2 with anything but zeros. Training recovers from this, but the first few hundred steps waste progress on disentangling the noise.

**Design:** Initialize the final `nn.Linear` of each `PartitionHead` to zero (weight and bias), and zero the final BatchNorm's bias. At construction, `f_1(x)` and `f_2(x)` output zeros. Partition k's contribution *grows smoothly* from zero as gradient pushes it, matching the assembled-embedding distribution of the previous phase.

Standard trick from ResNet init and diffusion model "zero convs." Costs nothing to implement; eliminates the transition noise.

### 3. Freeze the backbone from phase 2 onward

**Lesson:** If the backbone is trainable in phase 2, training can subtly "unfreeze" f_0 by drifting backbone features in a direction that makes f_0's frozen weights still optimal on the drifted features — partially defeating the freeze and re-enabling collapse.

**Design:** Backbone trains in phase 1 only. Phases 2, 3, 4 treat the backbone as a fixed feature extractor. Phases 2 and 3 are pure boosting on frozen features. Phase 4 (if used) unfreezes everything for a short polish.

Secondary benefit: massive wall-clock speedup in phases 2/3 (no backbone backward pass).

### 4. Port switchable BatchNorm from Variant B

**Lesson:** The assembled embedding has a different zero-fraction per active-width config:
- Phase 1 assembly: 2/3 zeros (only slot 0 non-zero)
- Phase 2 assembly: 1/3 zeros (slots 0, 1 non-zero)
- Phase 3 assembly: 0 zeros (all slots non-zero)

A single BN averaging running stats over all three distributions produces the distribution mismatch that B was built to solve.

**Design:** Reuse `SwitchableBatchNorm1d` from `nested.py`. Three per-width BN modules applied in `post_assembly`, selected by `set_eval_width`.

### 5. Skip BN for non-prefix eval configs

**Lesson:** Same logic as B. BN running stats are gathered from prefix-structured embeddings only (since training always assembles in prefix order). Non-prefix configs like {0, 2} produce embeddings whose distribution BN has never seen. Applying BN there introduces unpredictable error.

**Design:** Copy B's `_eval_is_prefix` check. Non-prefix eval configs skip BN entirely in `post_assembly`.

### 6. Structured partition dropout during phases 2 and 3

**Lesson:** The ArcFace head in B's training saw every subset via random dropout sampling, which kept all subset distributions represented. In C, phase 2 would naively train only on `{0, 1}` full assemblies — meaning ArcFace slot-0 rows never see the `{0}` alone case, and would drift off-distribution from their phase-1 optima.

**Design:** During phases 2 and 3, sample which subset to assemble on each batch — not randomly, but according to a *curated distribution* that includes the subsets eval will see.

**Phase 2 subset mix:**
- {0, 1}: 60% (primary training target)
- {0}: 40% (keeps ArcFace slot-0 fresh)

**Phase 3 subset mix:**
- {0, 1, 2}: 50% (primary)
- {0, 1}: 25%
- {0}: 15%
- Non-prefix {0, 2} or {1, 2}: 10% (to test graceful degradation in eval distribution)

These ratios are starting points — configurable via YAML and worth ablating.

Note: only partitions currently unfrozen (and ArcFace rows serving them) receive gradient, regardless of which subset is assembled. A phase-2 batch assembled as {0} alone still contributes gradient to f_1 via the frozen f_0's slot being part of the loss computation — but f_1 doesn't produce any output for this batch, so its gradient is zero. In practice, these "keep-fresh" batches only exercise the frozen slot-0 rows and don't actively train f_1.

### 7. Per-partition norm logging

**Lesson:** Partition collapse in B was invisible until LFW eval. In retrospect, `||p_1||` trending low during training would have been a clear canary.

**Design:** Every step, log `||p_0||_2, ||p_1||_2, ||p_2||_2` (per-sample mean over batch). Add `train/partition_norm_{k}` scalars to the existing logging infrastructure. If `||p_k||` trends toward zero during phase k's active training, that is an immediate collapse signal.

### 8. P0 standalone canary eval

**Lesson:** A subtle failure mode: freezing isn't actually freezing (e.g., ArcFace head rows drift, or BN running stats of earlier partitions keep updating, or backbone drift despite nominal freeze). The symptom is P0 standalone accuracy degrading from its phase-1 value during phase 2 or 3.

**Design:** After each phase completes, run a lightweight eval (CIFAR-100 val accuracy or a fixed LFW subset) at P0 alone. Compare to phase 1's P0 result. Regression beyond noise → freezing is broken somewhere. Runs in minutes, no training pause if run on CPU or scheduled between epochs.

### 9. Early-stop each phase

**Lesson:** Budget over-allocation to one phase is wasted epochs and stolen residual from the next.

**Design:** Each phase has a `min_epochs` and `max_epochs`. If loss plateau is detected (moving-average slope < threshold over a window) between min and max, phase ends early. Freed budget accrues to subsequent phases.

### 10. No KD, no orthogonality loss

**Lesson:** B's KD caused partition collapse via the "shortcut" where satisfying KD was cheapest by making extra partitions near-zero. A's orthogonality was trivially satisfied in high-D.

**Design:** No aux losses at all in phases 1–3. The hard freeze is the only constraint. Keep `compute_auxiliary_loss` returning zero.

### 11. Optional phase 4: short, low LR

**Lesson:** If phase 4 is run too long or at too high a learning rate, the collapse dynamic is re-enabled and the specialization achieved by phases 1–3 is partially undone.

**Design:** If `fine_tune.enabled: true`, run at most 1–2 epochs with LR scaled to 0.1× the phase 3 LR. All parameters unfrozen. Monitor P0 canary — if it regresses, cut the fine-tune phase short.

### 12. Reuse variant-agnostic infrastructure

- Width-0 guard in trainer: not triggered in C (f_0 is never dropped) but harmless.
- Per-width loss logging (`train/loss_w1/w2/w3`): useful, reuse.
- Checkpoint resume (`--resume`): critical for multi-phase training — each phase transition is a natural checkpoint boundary.
- CPU eval (`--cpu`): useful for P0 canary between phases without disturbing GPU training.
- Cached-raw-partitions optimization in `evaluate_lfw`: directly applicable.

### 13. Freezing mechanics (key implementation detail)

The trainer's `get_trainable_parameters(model, phase)` hook exists specifically for this. The strategy returns only the currently-trainable parameters per phase, and the optimizer is rebuilt at each phase transition with just those.

ArcFace head freezing by slot is subtler than freezing a whole module — the head is `nn.Linear(embedding_dim, num_classes)`. Need to freeze specific column slices of its weight matrix corresponding to the slot k×K : (k+1)×K range. Easiest approach: register a per-parameter gradient hook that zeros the gradient for frozen slots, rather than trying to split the Linear into frozen/trainable sub-parameters. Alternative: rebuild the ArcFace head each phase with trainable slots as a new `nn.Parameter` and frozen slots as a `register_buffer` (cleaner, but changes checkpoint shape).

Recommendation: gradient hooks. Less invasive, doesn't change checkpoint compatibility.

## Config schema

```yaml
variant: residual

residual:
  phases:
    - name: phase1
      epochs: 12
      min_epochs: 8
      trainable: [backbone, f_0, arcface_slot_0]
      subset_mix:
        "[0]": 1.0  # always train on {0} alone in phase 1
    - name: phase2
      epochs: 7
      min_epochs: 4
      trainable: [f_1, arcface_slot_1]
      subset_mix:
        "[0,1]": 0.6
        "[0]": 0.4
    - name: phase3
      epochs: 4
      min_epochs: 2
      trainable: [f_2, arcface_slot_2]
      subset_mix:
        "[0,1,2]": 0.5
        "[0,1]": 0.25
        "[0]": 0.15
        "[0,2]": 0.05
        "[1,2]": 0.05
  fine_tune:
    enabled: false
    epochs: 2
    lr_scale: 0.1

switchable_bn:
  enabled: true

zero_init_tail: true  # zero-init final Linear of f_1, f_2

early_stop:
  plateau_window: 500       # batches
  plateau_threshold: 0.001  # loss improvement per 100 batches

canary_eval:
  enabled: true
  between_phases: true
  benchmark: lfw_subset  # or cifar_val
```

## Evaluation behavior

Unchanged from B except that all 7 partition configs are supported with the understanding that:
- Prefix configs ({0}, {0,1}, {0,1,2}) will be strongest — they are exactly the subsets phases 1/2/3 optimized for
- Non-prefix configs ({0,2}, {1,2}, {1}, {2}) will be degraded but still computable. Expected quality ordering: prefix > non-prefix-containing-0 > non-prefix-without-0 ({1,2}, {1}, {2} — these are near-pure residual terms without their base).

The evaluator already supports all 7 configs via `_all_partition_configs`; no changes needed there beyond constructing the right `ResidualPartitionStrategy` from config.

## Expected training dynamics

**Phase 1** looks structurally like Variant A without the aux loss (but only slot 0 is populated). P0 standalone accuracy grows to some fraction of total capacity. Target: 80–90% of what an unpartitioned ResNet-50 + full-width ArcFace would achieve on this data. If phase 1 reaches near-full capacity, phase 1 was too long — truncate next run.

**Phase 2** should show a clear accuracy jump on `{0, 1}` over `{0}` alone on held-out data. `||p_1||` should grow from ~0 (zero-init) to comparable magnitude with `||p_0||`. If `||p_1||` plateaus at near-zero, collapse has occurred despite the freezing — most likely cause is that phase 1 was too long.

**Phase 3** adds partition 2's residual. Smaller incremental jump on `{0, 1, 2}` vs `{0, 1}`. If phase 3 shows no improvement, phase 2 fit the residual completely — not a failure, just means 2 partitions were sufficient for this data and 3-partition capacity is redundant for this dataset.

**Non-prefix eval** ({0,2}, {1,2}) is the most novel test. With proper training-time representation via the subset mix in phase 3, these should be meaningfully above random but clearly below their prefix analogues.

## Pitfalls to avoid (explicit list)

1. **Not actually freezing BN running stats.** `module.eval()` alone doesn't stop BN from updating running stats in training mode. When freezing a partition head, explicitly set `bn.track_running_stats = False` or put the frozen subtree in `eval()` mode.

2. **Loading optimizer state across phase transitions.** Optimizer state is per-parameter; rebuilding the optimizer at phase start drops momentum for the newly-active parameters. Acceptable (they start fresh anyway). Do NOT try to carry momentum across phases — it would transfer phase-1 momentum to phase-2 parameters that shouldn't have any.

3. **Scheduler across phases.** Each phase has its own LR schedule. Don't use a single long cosine schedule — it'll decay LR to near-zero by phase 3. Either reset the scheduler at phase start or use phase-local schedules.

4. **Checkpointing mid-phase.** The checkpoint needs to record which phase is active so resume picks up correctly. Extend `ExperimentLogger.save_checkpoint` to include a `phase_idx` field; `Trainer._load_resume_state` reads it and sets the phase accordingly.

5. **Zero-init cancellation by BN.** If the final Linear is zero-initialized but the following `BatchNorm1d` has affine=True and its weight=1, bias=0 at init, the zero output passes through unchanged. Good. But if the BN's running mean shifts later, it could un-zero the output. Verify the `BatchNorm1d` after the zeroed Linear doesn't have a learnable bias that could drift to nonzero during phase 1 (when f_1 has no gradient flowing but BN running stats might still update if not explicitly frozen).

6. **Subset mix dice roll.** When sampling subsets per-batch during phase 2/3, make sure the sampler is deterministic w.r.t. the global seed (use `torch.Generator` tied to seed, not `random.random()` alone) so runs are reproducible.

---

## Summary

Variant C takes the structural constraint principle that partially worked in B (force ordering) and replaces the soft prefix-dropout mechanism (which allowed partitions 1 and 2 to coast) with a hard training-time constraint (freeze earlier partitions entirely). The key structural difference is that phase k's gradient signal cannot adjust partitions 0..k-1 at all — they are fixed, and partition k must fit their residual error or reduce loss by zero.

Specific protections against the failure modes we've seen:

- **Against B's "partition 1 coasts" collapse:** Phase 1 budget is capped below convergence, leaving real residual for phase 2 to fit; freezing in phase 2 means f_1 *cannot* coast — the only way to reduce loss is to find new information.
- **Against A's trivial aux loss satisfaction:** No aux loss at all. Hard constraint only.
- **Against B's BN distribution mismatch:** Switchable BN carried forward.
- **Against phase transition noise:** Zero-init newly-unfrozen heads.
- **Against silent freezing failures:** Per-partition norm logging + P0 canary eval between phases.

If C collapses despite all of this, the diagnostic will be clear from logs (which partition, at which phase, with which norm trajectory) and the next move is the architectural residual variant — with the acceptance that it sacrifices the any-subset inference property.
