# Direction 2 — CIFAR-100 Campaign Record

**Period:** 2026-05-08 → 2026-08-04
**Branch:** `Direction2` (bug fixes originally scoped to `D2_Bugsweep`)
**Scope:** Repair of the Direction 2 boosting pipeline, followed by the Phase 1
CIFAR-100 smoke-test campaign through to the decision to change testbed.

This document is the consolidated record: what was broken, what was measured,
what was established, and what remains open. It is written to be self-contained
for a new session and to serve as source material for write-up.

---

## TL;DR

1. The Direction 2 pipeline contained **20 tracked bugs plus 9 more found during
   experimentation**. The most severe — the hard-pair miner returning **zero
   genuine pairs** — invalidated every result produced before it was fixed.
2. Once repaired, the mechanism was measured cleanly. On CIFAR-100 the boosted
   partitions are **redundant generalists, not specialists**: a deployable
   score-level combiner scores **3.8σ worse than P0 alone**.
3. CIFAR-100's **useful embedding dimensionality is ≈ 12** and its **minimum
   trainable K is 8** (K=4 and K=6 collapse). Since any N ≥ 2 forces N·K ≥ 16 >
   12, **no valid partition configuration exists at any N**. The testbed is
   arithmetically unable to host the progressive claim.
4. **Decision: change the Phase 1 gate dataset.** The recommended replacement is
   `casia_subset` sized via `--num-identities`, validated first with the
   dimensional-headroom protocol described in §5.
5. The bug fixes and the diagnostic suite are **dataset-independent** and carry
   forward unchanged.

---

## 1. Bugs fixed

### 1.1 From `D2_bugfix_tracker.md` (BUG-1 … BUG-20, all closed)

| ID | Severity | Defect | Fix |
|----|----------|--------|-----|
| 1 | Critical | Mined hard pairs never drove training; `HardPairDataset` was dead code. The boosting hypothesis was not implemented at all. | `_train_phase_k` builds a `HardPairDataset`/`DataLoader` from mined indices; pairs refresh at epoch boundaries. |
| 2 | Critical | `backbone_state="frozen"` + triplet/contrastive froze `partition_heads[k]`, the only module producing `pk_emb` → zero gradient. | Force-unfreeze current partition head; freeze only prior ones. |
| 3 | Critical | `SubCenterArcFace` centroids never entered the optimizer. | Added `boosting_loss_fn.parameters()` as a param group. |
| 4 | Critical | CIFAR-100 path ran `trainer.train()` (all phases) then re-looped phases 1+, training them twice, first with wrong labels. | Call `_train_phase_0` directly, then loop phases 1+. |
| 5 | Critical | `partition_arcface_heads[0]` used instead of `[k]`; dim mismatch crash for k>1; ArcFaceMargin passed identical hard/easy splits. | Index `[k]`; real per-pair weights; genuine hard/easy split by difficulty. |
| 6 | Critical | Phase 1+ arcface heads built with `num_classes=20`, then fed labels ≥ 20. | `rebuild_head_for_num_classes()` before each phase. |
| 7 | Critical | `ContrastiveLoss` NaN gradient at zero distance (`sqrt(0)`). | `+1e-8` inside the sqrt. |
| 8 | Critical | `requires_grad_(False)` does not freeze BatchNorm running stats; frozen stages drifted during phases 1+. | `_set_frozen_modules_eval()` calls `.eval()` on frozen submodules after every `.train()`. |
| 9 | High | D2 saved `{"backbone": …}`; D1's `EmbeddingCache` reads `ckpt["model_state_dict"]["backbone"]`. | Nested under `model_state_dict`; loader accepts both. |
| 10 | High | `ConfidenceWeighted` confidence signals constant (norm of a unit vector = 1.0); `scalar_head` params unreachable. | Backbone exposes `partitions_raw`; class is now an `nn.Module`. |
| 11 | High | `TripletLoss` semi_hard accepted anchors with no positive (`ap ≈ 0` passed the `ap >= 0` check). | Explicit `has_pos` filter. |
| 12 | High | No class-balanced sampling; CASIA batches would rarely contain positives. | `build_class_balanced_sampler`. |
| 13 | High | `LearnedCombiner` shape inference returned the first factorisation (P=2, K=129 for a P=4, K=64 checkpoint). | Iterate largest-first; prefer checkpoint metadata. |
| 14 | High | `LearnedCombiner.combine()` had a different signature to the other combiners. | Unified with `mask=None` default. |
| 15 | Medium | `build_scheduler` divided by zero when `epochs ≤ warmup_epochs`. | `T_max = max(1, …)`. |
| 16 | Medium | TAR@FAR incomparable across subsets (CosineConcat of P unit vectors has norm √P). | L2-normalise `combined` before scoring, both eval paths. |
| 17 | Medium | `partition_k.pt` stored an untrained arcface head for non-arcface losses. | Also persist `partition_proj` (the trained projection head). |
| 18 | Medium | `_get_train_loader(shuffle=False)` ignored on the CASIA path. | Loader construction honours the flag. |
| 19 | Low | K-fold threshold grid 0.01-step. | `np.linspace(-1, 1, 1000)`. |
| 20 | Low | Epoch summary printed head LR only. | Prints `head_lr` and `backbone_lr` separately. |

### 1.2 Found during experimentation (not in the original tracker)

These were discovered *because* of the campaign and are the more instructive set.

**B-21 — `HardPairMiner` selected 100% impostor pairs.** The single most
damaging bug in the project. `difficulty = where(is_same, 1-score, score+1)`
places impostors on a strictly higher scale than genuines, so top-k never
reaches a genuine pair. `band` was nearly as bad (2.8% genuine). P1/P2 were
therefore trained with **no positive pairs**, whose only gradient is "push
apart" — optimum a uniform spread on the hypersphere, i.e. chance verification.
*Fix:* stratified candidate generation (genuine pairs sampled explicitly by
class) and stratified selection within pair type; `genuine_fraction` now logged
per refresh. Verified 50.0% at both CIFAR (100 classes) and CASIA (10 572)
scale. **Every result predating this fix is void.**

**B-22 — Sweep monotone check was wrong, twice.** First it tested only the
prefix chain P0<P01<P012 and never P02→P012, flagging runs where P02 > P012.
After that was fixed it still had no significance gate and flagged a
**+0.0015 gain against a 0.004 SE** as success. *Fix:* full cover-pair lattice
check plus a 2σ gate on the P0→full gain.

**B-23 — Checkpoint collisions across sweep runs.** Default run name is
`d2_{dataset}_{backbone_state}_{loss}` — contains neither `K` nor
`mining_strategy`. The K-sweep wrote all four runs to one directory; the
original 10-run sweep collided pairwise. Printed accuracies were unaffected
(eval is in-process), but only the last checkpoint of each group survived.
*Fix:* `sweep.py` passes a per-combination `--run-name`; trainer warns loudly
before overwriting.

**B-24 — `--combination` silently ignored on the CIFAR-100 path.** The eval
hardcoded `CosineConcat()`. A `confidence_weighted` run would have returned
identical numbers and read as a clean negative.

**B-25 — All three confidence sources structurally dead.** `embedding_norm` is
unconstrained by any loss (every loss normalises its input, so the pre-norm
magnitude is a free parameter); `cosine_magnitude` is hardcoded to uniform
weights, making it identical to `cosine_concat`; `scalar_head` is randomly
initialised with no training path. *Partial fix:* warning on `scalar_head`;
the others remain unusable by design and are documented as such.

**B-26 — `--eval-only` on the CIFAR path never loaded checkpoints**, scoring a
randomly-initialised backbone. *Fix:* calls `_load_phase_checkpoints`.

**B-27 — Checkpoint loader could not restore a changed label space.** Phase 1+
heads are rebuilt to 100 classes during training (BUG-6's fix) but the loader
constructed 20-class heads. *Fix:* infer `num_classes` from the checkpoint
weight shape and rebuild before loading.

**B-28 — Backbone loader looked only at the last phase**, silently skipping the
load if absent. *Fix:* walk phases downward; raise rather than evaluate an
untrained backbone.

**B-29 — `convert_rec.py` used the wrong RecordIO magic number**
(`0xCEDAEDFE`; MXNet uses `0xCED7230A`), so every record failed the header
check and the converter reported zero records from a valid `.rec` file.

---

## 2. Experimental record

All CIFAR-100. ResNet-50 backbone, 32×32 input, 3×3 stem.
Chance = 0.50 for `pair_acc` (binary same/different).

### 2.1 Sweep 1 — mining × loss, `backbone_state=partial`, K=128 (10 runs)

**Invalidated by B-21 and by trunk drift.** Retained because the drift
measurement is itself a finding.

P0 should be constant across all ten runs (phase 0 is identical; `--loss` and
`--mining-strategy` affect phases 1+ only). It ranged **0.4905 – 0.7441**.

Cause: `partial` with `frozen_stages=3` leaves layer3, layer4 and the pooling
tail trainable for 40 further epochs. P0's projection head is frozen but reads a
trunk that moves underneath it. Directly measured later: **P0 drift under
`partial` = 1.04** on unit-norm embeddings (a completely different vector);
under `frozen` = **0.000e+00**, bit-identical.

`corr(P0, P012 − P0) = −0.848`. The apparent "gains" were recovery from trunk
damage, not complementarity. The best single number in the whole sweep was a
**P0-alone** value (0.7441); in 10/10 runs P012 failed to be the best subset.

### 2.2 Sweep 2 — K sweep, `backbone_state=frozen` (control verified)

| K | P0 | P01 | P02 | P012 | Δ(P012−P0) |
|---|----|----|----|----|----|
| 8 | 0.8538 | 0.8539 | 0.8534 | 0.8524 | −0.0014 |
| 16 | 0.8638 | 0.8645 | 0.8650 | 0.8648 | +0.0010 |
| 32 | 0.4894 | 0.4897 | 0.4889 | 0.4902 | *(diverged run)* |
| 128 | 0.7611 | 0.7497 | 0.7308 | 0.7260 | −0.0351 |

K=128 shows monotone **decrease** — the concat-averaging dilution effect.
K=32 collapsed to chance; a training divergence, never re-seeded.

*Note:* this was read at the time as disconfirming the capacity hypothesis. It
does not — see §3.2. K=8 and K=16 are both at or above the task's useful
dimensionality, so the hypothesis was tested entirely inside the saturated
region.

### 2.3 Residual-information probe (K=128 checkpoint)

`scripts/residual_probe.py`. 100-way subclass linear probes, fit on train,
scored on val.

| representation | dim | subclass acc |
|---|---|---|
| trunk F | 2048 | 0.4285 |
| P0 only | 128 | 0.2842 |
| P1 only | 128 | 0.3506 |
| P2 only | 128 | 0.3516 |
| residual F⊥ | 2048 | 0.3521 |

R² (trunk variance explained by P0) = 0.5397. Residual retains **81.7%** of the
trunk's decodable subclass signal. P1's 128 dims capture as much as the entire
2048-dim residual.

**Caveat:** this checkpoint predates the B-21 fix, so P1/P2 were trained on an
all-impostor set. Interpret with care.

### 2.4 K × combination sweep (`confidence_weighted`)

`confidence_weighted` was **strictly worse at every multi-partition subset, at
every K** (e.g. K=16: P012 0.8648 → 0.7436). Explained by B-25: all confidence
sources are noise. P0 was bit-identical between combiners at each K, confirming
the combination axis touches only multi-partition subsets.

### 2.5 Post-B-21, `batch_hard` — collapse

`genuine=50.0%` as intended, but phase-1 loss pinned at **0.3003** (margin =
0.3) from epoch 5 through 20. A triplet loss at exactly the margin means
`ap == an` for every anchor — a single point. Confirmed: collapsed embedding
returns 0.3000, healthy returns 0.0000.

`P1 = P2 = P12 = 0.4886`, `std cos = 0.0000`. Double-hard mining (miner selects
extremes, then `batch_hard` re-mines within them) on a frozen trunk at lr=0.1.

### 2.6 Post-B-21, `semi_hard` — collapse fixed

`std cos` 0.0000 → **0.2979**. P1 alone 0.4886 → **0.8455**. The loss sitting
at ~0.28 is *not* collapse: semi-hard selects the negative just above `ap` by
construction, bounding the loss in (0, margin).

### 2.7 Definitive measurement — semi_hard @ 100 000 eval pairs

| subset | pair_acc | pair_std | TAR@1e-3 |
|---|---|---|---|
| P0 | **0.8650** | 0.0024 | 0.0482 |
| P1 | 0.8514 | 0.0039 | 0.0452 |
| P2 | 0.8530 | 0.0030 | 0.0403 |
| P01 | 0.8534 | 0.0041 | 0.0471 |
| P02 | 0.8553 | 0.0029 | 0.0410 |
| P12 | 0.8570 | 0.0037 | 0.0440 |
| P012 | 0.8577 | 0.0036 | 0.0445 |
| TRUNK | 0.8486 | 0.0032 | 0.0576 |

- **Oracle (union of per-pair correctness):** 0.9011
- **Realisable combiner** (logistic regression on partition scores, 10-fold
  out-of-sample): **0.8559 ± 0.0038 vs P0 0.8650 → −0.0092**, against 2·SE =
  0.0024. **3.8σ worse than P0.**
- P1 fixes 21.6% of P0's errors, error correlation **+0.704**; P2 19.7%, **+0.730**

**Specialisation profile** (accuracy by P0 decision-margin quartile):

| stratum | P0 | P1 | P2 |
|---|---|---|---|
| Q1 (P0 least sure) | 0.7574 | 0.7252 | 0.7294 |
| Q2 | 0.8784 | 0.8590 | 0.8608 |
| Q3 | 0.8760 | 0.8710 | 0.8718 |
| Q4 (P0 most sure) | 0.9512 | 0.9512 | 0.9512 |

P1/P2 are **worse than P0 in every stratum**. No region where they know better.
The +0.0330 oracle gain is noise-driven union, not exploitable structure.

### 2.8 Ceiling runs (phase 0 trained directly on 100-way subclass)

| config | pair_acc | TAR@1e-3 | TRUNK |
|---|---|---|---|
| K=128, 40 ep | 0.8366 ± 0.0077 | 0.1206 | 0.7665 |
| K=16, 40 ep | **0.8973 ± 0.0019** | **0.3211** | 0.8852 |

Against D2's P0 (0.8650 / 0.0482): headroom **+0.0323 (≈13σ)** on pair_acc and
**6.7×** on TAR. **Saturation refuted** — real headroom exists and D2 fails to
capture it, moving backwards instead.

### 2.9 Dimensional-headroom curve (subclass, single partition, 20 ep, 100k pairs)

| K | pair_acc | TAR@1e-3 | std cos | status |
|---|---|---|---|---|
| 4 | 0.5086 ± 0.0042 | 0.0006 | 0.0000 | **COLLAPSED** |
| 6 | — | — | — | **COLLAPSED** |
| 8 | 0.8654 ± 0.0040 | 0.0992 | 0.0236 | ok |
| 12 | **0.8786 ± 0.0046** | **0.1393** | 0.0194 | peak |
| 16 | 0.8734 ± 0.0042 | 0.1351 | 0.0169 | past peak |

K=8 → 12 gains +0.0132 (≈3σ). Curve peaks at ~12 and declines by 16.

Collapse cause: ArcFace `s=64, m=0.5` cannot place 100 class centres with a 0.5
angular margin in fewer than ~8 dimensions. The **minimum trainable K is 8**.

---

## 3. Established findings

### 3.1 The partitions are redundant generalists, not specialists

Converging evidence, all post-fix and at high precision: P1/P2 lose to P0 in
every decision-margin stratum; error correlation +0.70/+0.73; a deployable
score-level combiner is 3.8σ *worse* than P0 alone. Hard-pair **sampling** does
not produce complementarity.

**Structural reason.** With the trunk frozen, P0 and P1 are both linear readouts
of the same 2048-d vector. Reweighting *which pairs* P1 sees does not change
*which directions* best separate them — those are the most discriminative
directions available, and P0 already occupies them. AdaBoost's reweighting works
because it changes the optimal hypothesis; here it changes only the sampling
distribution and the optimum barely moves.

**Implementation gap consistent with this.** For `triplet`,
`_compute_boosting_loss` passes only `(pk_emb, batch_labels)` and discards
`is_same` / `pidx_a` / `pidx_b`; TripletLoss then re-mines internally. The
boosting signal enters as **batch composition only, never as a reweighted
objective**. This has not been tested and is a live avenue.

### 3.2 CIFAR-100 cannot host a progressive claim at any N

Two independently measured constants close this arithmetically:

- **Minimum trainable K = 8.** K=4 and K=6 both collapse (`std cos = 0.0000`,
  pair_acc at chance); K=8 trains cleanly.
- **Peak useful dimensionality = 12.** Accuracy rises to K=12 and declines by
  K=16.

A progressive configuration needs `K ≥ 8` (each partition trainable) **and**
`N·K ≤ 12` (total under the peak, so there is headroom to progress into). Since
`N ≥ 2`, the smallest possible total is `2 × 8 = 16 > 12`. **No valid (N, K)
exists** — not merely for N=3, but for any N.

This is a property of the *testbed*, not the mechanism, and no D2 fix addresses
it.

This is a property of the *testbed*, not the mechanism, and no D2 fix addresses
it. It also quantifies why CASIA should differ: useful dimensionality there is
in the hundreds, so N=3 × K=128 = 384 sits comfortably under it with every
partition far above collapse.

**Correction to an earlier reading.** The K-sweep in §2.2 was taken as
disconfirming the capacity hypothesis. It does not: K=8 (0.8538) and K=16
(0.8638) differ by 0.010 and are both essentially at the useful dimensionality,
so the hypothesis was never tested in the regime where capacity binds.

### 3.3 The frozen trunk caps the partitions

P1 = 0.8514, P2 = 0.8530 against TRUNK = 0.8486. The boosted partitions land
barely above raw trunk quality, while P0's co-trained projection reaches 0.8650.
Under `frozen` they are linear readouts of a trunk trained only on 20
superclasses and cannot access structure it never encoded.

This is the core tension, now quantified: **frozen** preserves P0 exactly
(0.000e+00 drift) but caps partitions at trunk quality; **partial/fine_tuned**
lets the trunk acquire task structure but destroys P0 (1.04 drift). D2 has no
mechanism to obtain one without the other.

### 3.4 `pair_acc` saturates; TAR@1e-3 does not

Ceiling K=16: pair_acc 0.8973 vs D2's 0.8650 (+0.032), but TAR 0.3211 vs 0.0482
(**6.7×**). `Direction_2_PLAN.md` defines success as monotone `pair_acc` — the
metric with the least headroom. TAR@1e-3 needs ≥50k pairs to be stable (the
FAR=1e-3 operating point is fixed by n_impostor/1000 scores).

*Note:* TAR did **not** rescue the D2 comparison — P012's TAR (0.0445) is at or
below P0's (0.0482). Both metrics agree there is no progressive gain.

### 3.5 A confound not yet ruled out

P1/P2 are trained on far less data than P0: ~3 900 optimizer steps for P0
(50 000 images ÷ 256) versus ~780 for P1 (~5 000 mined pairs ÷ 128). Phase
epochs run 6.2 s against phase 0's 70 s. "P1 lands at trunk quality" may be
partly **undertrained** rather than purely redundant. The control
(`--epochs-per-phase 100`) was specified but not run.

---

## 4. Diagnostic tooling built

All dataset-independent; transfers to CASIA unchanged.

All live in **`src/ppi/evaluation/diagnostics.py`** as one dataset-agnostic
module. It takes pair-aligned `(N, P, K)` partition embeddings plus a
same/different vector, so CIFAR-100, LFW, CASIA and any future testbed share a
single implementation and a single output format. Callers:

- `scripts/train_boosting.py::_eval_cifar100_verification`
- `BoostingEvaluator.run_diagnostics()` (LFW / CASIA), invoked automatically
  after `evaluate_lfw()`

`DiagnosticsReport.render()` prints the tables; `.to_dict()` returns a
JSON-serialisable structure for wandb. A `verdict` property applies the
precedence rules (collapse invalidates everything below it; the realisable
combiner overrides the oracle).

| Diagnostic | Answers | Location |
|---|---|---|
| **Collapse check** — std of pairwise cosine per partition | Did the partition degenerate to a point? (`std cos < 0.01`) | eval table |
| **Complementarity / oracle** — union of per-pair correctness | Ceiling for *any* combiner | `--eval-all-subsets` |
| **Specialisation profile** — accuracy by P0 decision-margin quartile | Is P_k a specialist or a worse generalist? | `--eval-all-subsets` |
| **Realisable combiner** — 10-fold logistic regression on partition scores | Is the oracle gain actually achievable? | `--eval-all-subsets` |
| **TRUNK reference row** | Does the shared representation beat its own projection? | `--eval-all-subsets` |
| **Dimensional-headroom curve** — single-partition K sweep | Can this testbed host the progressive claim at all? | protocol, §5 |
| **Residual-information probe** — regress trunk on P0, probe the residual | Is unused signal present, and did P_k capture it? | `scripts/residual_probe.py` |
| **Non-anchored subsets** (P1, P2, P12) | Per-partition metric quality in isolation | `--eval-all-subsets` |

Why the last two matter: the collapse check uses **variance**, not mean cosine —
a concentrated-but-healthy embedding can sit at high mean cosine and still
separate pairs; a collapsed one gives every pair the same score. And the oracle
is an **upper bound only**: it credits any disagreement including noise, which
is why the realisable-combiner test exists alongside it.

**Regression suite:** `tests/test_boosting.py`, 31 tests, ~70 s on CPU, no
dataset download or wandb required. Covers every bug contract above including
the `frozen ⇒ P0 pinned` experimental control.

Note `pyproject.toml` gained `pythonpath = ["src"]` so `pytest` runs without an
editable install.

---

## 5. Testbed validation protocol

The most transferable output of this campaign. **Before adopting any dataset as
a gate, measure its dimensional-headroom curve.**

```bash
python scripts/sweep.py --config configs/direction_2_base.yaml \
    --dataset <DATASET> --axes K=16,32,64,128 --run-prefix dimhead \
    --fixed --num-partitions 1 --epochs-phase0 20 --eval-pairs 100000
```

Read the P0 row from each run. The gate is **valid** only if:

1. K → N·K buys a clear gain (there is room to progress into), and
2. every K trains without collapsing (`std cos > 0.05`).

CIFAR-100 fails both jointly: peak at K≈12, minimum trainable K=8, so the
smallest total (2×8=16) already exceeds the peak.

---

## 6. CLI and API changes

Added this campaign, all opt-in with defaults unchanged:

| Flag | Purpose |
|---|---|
| `--K` | Per-partition embedding dim (overrides `partitions.K`) |
| `--eval-pairs` | Verification pair count (≥50k for stable TAR) |
| `--eval-all-subsets` | Non-anchored subsets + full diagnostic block |
| `--cifar-phase0-labels {superclass,subclass}` | Phase-0 label space; `subclass` gives a ceiling run |
| `--arcface-s`, `--arcface-m` | ArcFace geometry; needed at very low K |
| `--num-identities` | `casia_subset` size (now actually forwarded by `build_dataloader`) |
| `--run-prefix` (sweep) | Per-combination run names; prevents checkpoint collisions |
| `mining_genuine_fraction` (config) | Target genuine share in mined pairs, default 0.5 |

New API: `CIFAR100BoostingAdaptor.get_val_images()` and
`get_val_pair_indices()` — embed each unique val image once and index for pairs,
so pair count is decoupled from embedding cost.

---

## 7. Decision and next steps

**Decision (2026-08-04): change the Phase 1 gate dataset.** CIFAR-100 is
structurally unable to host a three-way progressive claim (§3.2). Continuing to
tune the mechanism there cannot produce an interpretable positive.

### Recommended path

1. **Adopt `casia_subset`**, sized via `--num-identities` (start at 500 for
   iteration speed; step to 2000 if too thin). Already implemented,
   identity-stratified, deployment domain, no superclass/subclass scaffolding —
   and a gate result transfers to Phase 3 without a domain-transfer argument.
2. **Run the §5 validation protocol first.** Do not run the mechanism until the
   testbed is known to have headroom.
3. **Run D2** at whichever K places N·K near the knee, with
   `--eval-all-subsets --eval-pairs 100000`.

### Open questions carried forward

- **Is the redundancy finding dataset-specific?** §3.1 was established on a
  frozen superclass trunk. If the same signature (P_k below P0 in every stratum,
  realisable combiner worse than P0) appears on CASIA, it is the mechanism.
- **Undertraining confound** (§3.5) — run `--epochs-per-phase 100`.
- **Boosting as a reweighted objective, not just a sampler** (§3.1) — untested
  and arguably the most faithful reading of the brief's hypothesis.
- **Adaptive trunk with P0 preservation** — the tension in §3.3 is now
  measurable, so a distillation penalty against phase-0 embeddings is
  implementable and controllable. Not yet built.
- **Success criterion** — `Direction_2_PLAN.md` gates on monotone `pair_acc`;
  §3.4 shows that metric saturates. Consider TAR@FAR as primary, with ≥50k pairs.
- **K=32 divergence** (§2.2) never re-seeded.

### Reading guide for any future run

In order — a failure at any step invalidates everything below it:

1. `genuine=` in the mining log must be ≈50%.
2. `std cos` per partition must exceed 0.05 (not collapsed).
3. Loss must not sit at exactly the triplet margin (`batch_hard` only).
4. P0 must match its value from the corresponding sweep row.
5. Only then read the subset table, and prefer the **realisable combiner** over
   the oracle.
