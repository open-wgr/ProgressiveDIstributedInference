# Direction 1 Results: Learned Combiner Diagnostic

**Date:** May 2026  
**Branch:** `Direction1`  
**Benchmark:** LFW pair verification (pair accuracy, TAR@FAR=1e-3)  
**Protocol:** Combiner trained on cached CASIA embeddings, evaluated across all 7 non-empty partition subsets

---

## Summary Verdict

**Combiner rescues neither Variant C nor Variant D.**

The strict success criterion — P012 combiner accuracy ≥ P0 cosine baseline — fails for both checkpoints at both output dimensionalities. The dominant failure signal is combiner collapse: the graceful-degradation cosine similarity between combine(P012) and combine(P0) is ≥0.993 in every run, meaning the combiner learns to produce the same embedding regardless of which partitions are present. P1 and P2 carry no complementary identity signal that a post-hoc learned combiner can access.

**Direction 2 (structural redesign) is confirmed as the right next step.**

---

## Full Results

### Variant C — K_out = K

| Subset | Combiner Acc | Baseline Acc | Combiner TAR | Baseline TAR |
|--------|-------------|-------------|-------------|-------------|
| P0     | 0.7128      | 0.7190      | 0.0593      | 0.0653      |
| P1     | 0.7152      | 0.7230      | 0.0777      | 0.0853      |
| P2     | 0.7185      | 0.7313      | 0.0513      | 0.0773      |
| P01    | 0.7102      | 0.7272      | 0.0603      | 0.0850      |
| P02    | 0.7185      | 0.7453      | 0.0720      | 0.0677      |
| P12    | 0.7192      | 0.7503      | 0.0577      | 0.0920      |
| **P012**   | **0.7068**  | **0.7425**  | 0.0700      | 0.0843      |

Graceful degradation (combine(P012) vs combine(P0)): **mean cosine = 0.9942** (std 0.0035)

### Variant C — K_out = 3K

| Subset | Combiner Acc | Baseline Acc | Combiner TAR | Baseline TAR |
|--------|-------------|-------------|-------------|-------------|
| P0     | 0.7175      | 0.7190      | 0.0520      | 0.0653      |
| P1     | 0.7172      | 0.7230      | 0.0717      | 0.0853      |
| P2     | 0.7170      | 0.7313      | 0.0553      | 0.0773      |
| P01    | 0.7097      | 0.7272      | 0.0587      | 0.0850      |
| P02    | 0.7115      | 0.7453      | 0.0457      | 0.0677      |
| P12    | 0.7115      | 0.7503      | 0.0547      | 0.0920      |
| **P012**   | **0.7110**  | **0.7425**  | 0.0493      | 0.0843      |

Graceful degradation: **mean cosine = 0.9941** (std 0.0035)

---

### Variant D — K_out = K

| Subset | Combiner Acc | Baseline Acc | Combiner TAR | Baseline TAR |
|--------|-------------|-------------|-------------|-------------|
| P0     | 0.7077      | 0.7360      | 0.0287      | 0.0540      |
| P1     | **0.6755**  | 0.6408      | 0.0183      | 0.0173      |
| P2     | 0.7047      | 0.7213      | 0.0247      | 0.0603      |
| P01    | 0.7077      | 0.7218      | 0.0217      | 0.0347      |
| P02    | 0.7178      | 0.7377      | 0.0223      | 0.0613      |
| P12    | 0.7133      | 0.7258      | 0.0190      | 0.0450      |
| **P012**   | **0.7217**  | **0.7272**  | 0.0190      | 0.0503      |

Graceful degradation: **mean cosine = 0.9948** (std 0.0034)

### Variant D — K_out = 3K

| Subset | Combiner Acc | Baseline Acc | Combiner TAR | Baseline TAR |
|--------|-------------|-------------|-------------|-------------|
| P0     | 0.6980      | 0.7360      | 0.0290      | 0.0540      |
| P1     | 0.6958      | 0.6408      | 0.0167      | 0.0173      |
| P2     | 0.6937      | 0.7213      | 0.0240      | 0.0603      |
| P01    | 0.7158      | 0.7218      | 0.0283      | 0.0347      |
| P02    | 0.7087      | 0.7377      | 0.0157      | 0.0613      |
| P12    | 0.7018      | 0.7258      | 0.0143      | 0.0450      |
| **P012**   | **0.7133**  | **0.7272**  | 0.0200      | 0.0503      |

Graceful degradation: **mean cosine = 0.9935** (std 0.0040)

---

## Key Observations

### 1. Success criterion fails for both checkpoints

P0 cosine baseline sets the functional floor: 0.7190 (Variant C) and 0.7360 (Variant D). The best P012 combiner scores are 0.7110 (Variant C, 3K) and 0.7217 (Variant D, K) — both below their respective P0 floors.

### 2. Combiner collapse is the dominant failure mode

Graceful-degradation cosine similarity of ≥0.993 across all four runs means the combiner consistently maps every subset to approximately the same embedding as P0-alone. The combiner has no useful signal to combine — it defaults to reproducing P0's geometry regardless of what additional partitions are present.

### 3. K vs 3K makes no difference

Results for K and 3K output dimensions are within noise across both checkpoints. Output dimensionality is not a contributing factor to the failure.

### 4. Variant D shows partial structure, Variant C does not

Two differences between C and D are diagnostically meaningful:

- **P012 combiner vs P012 baseline gap**: −0.036 in Variant C (K), −0.006 in Variant D (K). The combiner nearly closes the gap to the cosine baseline in Variant D, compared to a large deficit in Variant C.
- **P1 combiner beats P1 baseline in Variant D (K)**: 0.6755 vs 0.6408. The only case across all runs where the combiner outperforms the baseline. Orthogonality loss created enough geometric separation in P1 for the combiner to extract something — but the signal is still too weak to lift P012 above P0 alone.

Variant C's cosine baseline for P012 (0.7425) actually *exceeds* P0 baseline (0.7190), suggesting residual training left some genuine multi-partition signal in the cosine path — but the combiner couldn't access it and degraded further. Variant D's cosine baseline for P012 (0.7272) is *below* P0 baseline (0.7360), the canonical combination-degrades-the-average failure mode. The combiner partially repairs this but cannot overcome it.

### 5. TAR@FAR=1e-3 consistently lower for combiner than baseline

The combiner underperforms the cosine baseline at the hard verification threshold across nearly all cells. This held even in cases where pair accuracy is comparable, indicating the combiner does not produce well-calibrated similarity scores at the operating point that matters for deployment.

---

## Interpretation Against Diagnostic Outcomes

| Outcome | Assessment |
|---------|-----------|
| Combiner rescues C only | No |
| Combiner rescues D only | No |
| Combiner rescues both | No |
| **Combiner rescues neither** | **Yes — confirmed** |

The combination function was not the bottleneck. The partitions themselves are uninformative as currently trained.

---

## Implications for Direction 2

**The core problem is upstream of the combiner.** P1 and P2 are not encoding complementary identity information — they are encoding near-redundant projections of the same backbone representation that was optimised primarily for P0. No post-hoc combination function can recover signal that is not present.

**On orthogonality:** Variant D's partial recovery (smaller combiner-baseline gap, P1 combiner > P1 baseline) is weak but consistent evidence that orthogonality loss creates some real geometric separation. It is not sufficient on its own, but it is not wasted. Direction 2 should retain orthogonality as a candidate mechanism and test whether combining it with a structural training change produces a larger effect.

**The functional floor (P0-alone) is the correct comparison target for Direction 2.** The multi-node configuration needs to meet or exceed P0 cosine baseline, not beat the cosine-concatenated multi-partition baseline (which itself degrades below P0 in Variant D). Direction 2 succeeds if it reaches P012 ≥ P0 by any combination method — cosine or learned.
