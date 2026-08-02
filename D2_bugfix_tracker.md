# Direction 2 Bug Tracker

Produced by full codebase audit on 2026-05-08 (branch `D2_Bugsweep`).
Fix items in order of severity. Check off as done.

---

## CRITICAL — design not implemented / guaranteed crash

### BUG-1 · Mined hard pairs never drive training  
**Files:** `trainer.py:272-290`, `mining.py:134-164`  
`idx_a, idx_b` from `HardPairMiner.mine()` are passed to `_compute_boosting_loss` but **no branch uses them**. `HardPairDataset` / `build_hard_pair_dataset` are dead code. The trainer loops the ordinary `DataLoader`; the boosting hypothesis (train P_k on pairs the previous ensemble fails) is not realised.  
**Fix:** Build a `HardPairDataset` inside `_train_phase_k` from the mined indices, replace the inner `DataLoader` with it (or weight-sample the batch toward hard pairs).  
- [x] Fixed

---

### BUG-2 · `backbone_state="frozen"` + triplet/contrastive = no gradient  
**Files:** `backbone_state.py:42-44`, `trainer.py:227-237`  
`set_backbone_state("frozen")` freezes `backbone.partition_heads[k]` — the very module that produces `pk_emb`. With triplet or contrastive loss, `pk_emb` is the only thing in the compute graph; the only optimizer params are `partition_arcface_heads[k]` (never called). **Zero gradient, no learning.** Same applies when `frozen_stages >= 6` in partial mode.  
**Fix:** Unfreeze `backbone.partition_heads[k]` unconditionally (only the *previous* partition heads need freezing, not the current one). Or reconsider whether "frozen backbone" is meaningful for verification losses.  
- [x] Fixed

---

### BUG-3 · `SubCenterArcFace` centroids never enter the optimizer  
**Files:** `trainer.py:82`, `losses.py:187`, `trainer.py:227-237`  
`build_loss()` creates a `SubCenterArcFace` with learnable `weight` parameter. The trainer only adds `partition_arcface_heads[k].parameters()` to the optimizer. Centers stay at Xavier init forever.  
**Fix:** After `build_loss()`, add `boosting_loss_fn.parameters()` to the optimizer param groups.  
- [x] Fixed

---

### BUG-4 · CIFAR-100 path trains phases 1+ twice, wrong dataset first time  
**Files:** `scripts/train_boosting.py:238-247`  
`trainer.train(train_dataset=superclass_ds, num_classes=20)` runs **all** phases (0 through N-1) — then the loop on line 243 re-runs phases 1+. Phases 1+ run once with superclass labels (wrong) then again with subclass labels.  
**Fix:** Replace `trainer.train(...)` with just `trainer._train_phase_0(...); trainer._save_phase_checkpoint(0)` then let the existing loop handle phases 1+.  
- [x] Fixed

---

### BUG-5 · ArcFaceReweighted / ArcFaceMargin: wrong head index + crash in phase k>1  
**Files:** `trainer.py:352, 362`  
Both branches call `self.partition_arcface_heads[0](pk_emb)` — should be `[k]`.  
For `k > 1`, `ensemble_embs[:B]` has dim `k*K` but `pk_emb` has dim `K` → **verified crash**: `RuntimeError: The size of tensor a (256) must match the size of tensor b (128)`.  
ArcFaceMargin additionally passes `(hard_cosine, labels, hard_cosine, labels)` → hard_loss == easy_loss (double-counted, no actual hard/easy split).  
ArcFaceReweighted's weight computation uses `ensemble_embs[:B]` (first B rows of full training set, unrelated to current batch) — semantically wrong even when dims match.  
**Fix:** Change head index to `[k]`; rewrite pair-weight computation to use per-batch ensemble scores; fix ArcFaceMargin to pass a proper hard/easy split.  
- [x] Fixed

---

### BUG-6 · CIFAR-100 phase 1+ arcface-head has wrong output size (20 not 100)  
**Files:** `scripts/train_boosting.py:232-234`, `trainer.py:71-74`  
`partition_arcface_heads` are built once at `__init__` with `num_classes=20`. After phase 0, `trainer.num_classes` is mutated to 100 but the existing weight matrices keep shape `(K, 20)`. Any arcface-based loss in phase 1+ with a label ≥ 20 → `Target X is out of bounds`.  
**Fix:** Rebuild `partition_arcface_heads[k]` for k≥1 with `num_classes_phase1plus` before starting that phase.  
- [x] Fixed

---

### BUG-7 · `ContrastiveLoss` NaN gradient when `emb_a ≈ emb_b`  
**Files:** `losses.py:165`  
`dist = (emb_a - emb_b).pow(2).sum(dim=1).sqrt()`. `∂/∂x sqrt(0) = +inf` → NaN gradient. More likely under fp16 (rounding to zero).  
**Fix:** `dist = ((emb_a - emb_b).pow(2).sum(dim=1) + 1e-8).sqrt()`  
- [x] Fixed

---

### BUG-8 · BN running stats drift in "frozen" submodules  
**Files:** `trainer.py:260, 391-392`  
`self.backbone.train()` puts **all** submodules in train mode. `requires_grad_(False)` does not freeze `BatchNorm` running stats — they update on every forward pass. Frozen stages and earlier `partition_heads[j]` silently drift their running mean/var during phase k, corrupting phase-0 behaviour at eval.  
**Fix:** After `set_backbone_state` and after freezing prior partition heads, call `.eval()` on those frozen submodules explicitly.  
- [x] Fixed

---

## HIGH — wrong results without crash

### BUG-9 · D2 backbone checkpoint format incompatible with D1 EmbeddingCache  
**Files:** `trainer.py:404-407`, `combiner/cache.py:61`  
D2 saves `{"backbone": state_dict}`. D1 `EmbeddingCache.build()` loads via `ckpt["model_state_dict"]["backbone"]`. Brief says D1 combiner should load D2 checkpoints — contract violated.  
**Fix:** Either nest D2's save under `model_state_dict`, or update `EmbeddingCache` to accept both formats.  
- [x] Fixed

---

### BUG-10 · `ConfidenceWeighted` confidence signals are constant (always 1.0)  
**Files:** `combination.py:101-110`, `resnet.py:106`  
Backbone L2-normalises all partition outputs before returning them. `embedding_norm` then computes norm of a unit vector → 1.0. `cosine_magnitude` proxy is also norm of unit vector → 1.0. All confidences identical → uniform weights → identical result to CosineConcat.  
`scalar_head` is the only option that produces signal, but `ConfidenceWeighted` is not an `nn.Module`, so `scalar_head` parameters aren't reachable via `.parameters()` and can't be trained.  
**Fix:** Return pre-normalisation features from the backbone (or pass norms explicitly); make `ConfidenceWeighted` an `nn.Module` when `confidence_source="scalar_head"`.  
- [x] Fixed

---

### BUG-11 · `TripletLoss` semi_hard: anchors with no positive are not filtered  
**Files:** `losses.py:124-142`  
In the semi_hard branch, `ap = sum(dist * same_no_diag, dim=1) / (count + 1e-12)`. Anchors with no same-class neighbour get `ap ≈ 0`, but the validity check `(ap >= 0)` accepts them. They contribute spurious loss. The batch_hard branch is correct (`ap = -1` when no positive → filtered). Only semi_hard is broken.  
**Fix:** Add explicit filter: `has_pos = same_no_diag.any(dim=1)` and incorporate into `valid`.  
- [x] Fixed

---

### BUG-12 · TripletLoss with face datasets: batches almost never contain positives  
**Files:** `trainer.py:252, 427-448`  
CASIA has 10 572 classes; batch_size=256 → most batches have zero same-class pairs → loss returns `embeddings.sum() * 0.0` → no gradient. The dataloaders use plain random shuffle, not class-balanced sampling. On CIFAR-100 (100 classes, batch 256) it's marginal but not guaranteed.  
**Fix:** Add a class-balanced sampler (`torch.utils.data.WeightedRandomSampler` or a custom `BalancedBatchSampler`) to `_get_train_loader`.  
- [x] Fixed

---

### BUG-13 · `LearnedCombiner` shape inference picks wrong (P, K) when metadata absent  
**Files:** `combination.py:154-158`  
`for P in range(2, 10)` returns the first `P` where `(in_dim - P) % P == 0`. For `P=4, K=64 → in_dim=260`, the loop returns `P=2, K=129` first. Subsequent `load_state_dict` mismatches.  
**Fix:** Iterate largest-to-smallest, or make metadata a hard requirement (raise if absent).  
- [x] Fixed

---

### BUG-14 · `LearnedCombiner.combine()` has a different signature  
**Files:** `combination.py:170-173`, `boosting_eval.py:141-145`  
Takes `(partition_embeddings, mask)` while `CosineConcat` and `ConfidenceWeighted` take only `(partition_embeddings)`. The evaluator special-cases this; any other call site will break.  
**Fix:** Unify signature (`mask=None` default), or give it a protocol/base class.  
- [x] Fixed

---

## MEDIUM — broken in edge cases / misleading

### BUG-15 · `build_scheduler` crashes when `epochs ≤ warmup_epochs`  
**Files:** `training/schedulers.py:25`  
`T_max = total_epochs - warmup_epochs` → 0 or negative → CosineAnnealingLR raises "division by zero".  
**Fix:** Clamp `T_max = max(1, total_epochs - warmup_epochs)`.  
- [x] Fixed

---

### BUG-16 · TAR@FAR comparisons across partition subsets are on different score scales  
**Files:** `train_boosting.py:375-376`, `boosting_eval.py:152-155`  
CosineConcat of P unit vectors has norm √P; raw dot product `(emb_a * emb_b).sum()` is in `[-P, P]`. P0 scores are in [-1,1], P012 in [-3,3]. TAR@FAR is monotonic so within-subset the number is valid, but **cross-subset TAR@FAR are not comparable**. `pair_accuracy` re-normalises internally and is fine.  
**Fix:** L2-normalise `combined` before computing `sims` in both eval paths.  
- [x] Fixed

---

### BUG-17 · `_save_phase_checkpoint` stores untrained arcface head for non-arcface losses  
**Files:** `trainer.py:408-416`  
For triplet/contrastive losses, `partition_arcface_heads[k]` is never called during phase k training — it stays at Xavier init. The file `partition_k.pt` therefore contains a random matrix. (Eval doesn't use these heads, so it doesn't break verification metrics, but the file is misleading and the contract with D1 combiner loading is unclear.)  
**Fix:** Either also save `backbone.partition_heads[k].state_dict()` under a `partition_proj` key, or clarify that `partition_k.pt` is the arcface head only and document separately how to load the projection head (it's inside `backbone.pt`).  
- [x] Fixed

---

### BUG-18 · `_get_train_loader(shuffle=False)` ignored on CASIA path  
**Files:** `trainer.py:447`  
When `_train_dataset is None`, `_get_train_loader(shuffle=False)` falls through to `build_dataloader`, which forces `shuffle=True` for train split. `_compute_ensemble_embeddings` then returns embeddings in random order. Each call is internally consistent (labels paired with their embs), so mining still works, but indexes change between refreshes. Latent trap.  
**Fix:** Pass a `shuffle` kwarg through `build_dataloader`, or always set `_train_dataset`.  
- [x] Fixed

---

## LOW — cosmetic / minor accuracy

### BUG-19 · K-fold threshold grid too coarse (0.01 step)  
**Files:** `evaluation/metrics.py:109`  
`np.linspace(-1, 1, 200)` gives 0.01-step threshold search. Reported accuracy can wobble ~0.5% for high-accuracy models. Change to 1000 steps (0.002).  
- [x] Fixed

### BUG-20 · Phase-k epoch summary prints head LR only  
**Files:** `trainer.py:316, 321`  
`param_groups[-1]['lr']` is always the head group. Backbone LR (backbone groups) is already logged per-step at lines 304-305 but not in the end-of-epoch summary print. Misleading when backbone is partially frozen at a different LR.  
- [x] Fixed

---

## No-tests note

There are **zero boosting-specific tests** in `tests/`. All bugs above survived because there's no test harness for any of the D2 modules. A minimal smoke-test module covering:
- `HardPairMiner` cache/refresh logic  
- `TripletLoss` forward + valid-anchor filter  
- `set_backbone_state` param-group shapes and `requires_grad` state  
- `_compute_boosting_loss` dimensional contract (no crash for k=0,1,2)  
- `_eval_cifar100_verification` end-to-end on 2-epoch toy run  

would catch most of the above immediately.
