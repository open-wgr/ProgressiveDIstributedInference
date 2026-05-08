"""Combination strategies for Direction 2 multi-partition inference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CosineConcat:
    """Cosine similarity on concatenated per-partition L2-normalised embeddings.

    Default baseline — exactly replicates the existing eval in Evaluator.evaluate_lfw().
    Absent partitions are zero-padded. Enables Direction 2 to be evaluated
    independently of Direction 1's combiner outcome.
    """

    def combine(
        self,
        partition_embeddings: list[Tensor | None],
        mask: Tensor | None = None,
    ) -> Tensor:
        """Return (B, num_partitions * K) concatenated embedding."""
        del mask  # unused; accepted for signature parity
        parts = []
        for emb in partition_embeddings:
            if emb is None:
                parts.append(None)
            else:
                parts.append(F.normalize(emb.float(), dim=1, eps=1e-12))

        # Infer shape from first non-None embedding
        ref = next(e for e in parts if e is not None)
        B, K = ref.shape

        filled = []
        for emb in parts:
            if emb is None:
                filled.append(torch.zeros(B, K, device=ref.device, dtype=ref.dtype))
            else:
                filled.append(emb)
        return torch.cat(filled, dim=1)


class ConfidenceWeighted(nn.Module):
    """Per-partition similarity scores weighted by per-partition confidence.

    confidence_source: "embedding_norm" | "cosine_magnitude" | "scalar_head"
      - embedding_norm: pre-normalisation embedding magnitude (proxy for
                        representation quality; high norm → confident).
      - cosine_magnitude: |cosine(emb_a, emb_b)| — decisive scores signal
                          confident partitions.
      - scalar_head: tiny per-partition learned scalar head, ConfidenceHead(K, 1),
                     producing a calibrated confidence score from the embedding.
                     Adds ~K parameters per partition; is the most direct.
    """

    def __init__(
        self,
        confidence_source: str,
        num_partitions: int,
        partition_dim: int,
    ) -> None:
        super().__init__()
        if confidence_source not in ("embedding_norm", "cosine_magnitude", "scalar_head"):
            raise ValueError(
                f"Unknown confidence_source '{confidence_source}'. "
                "Choose from: embedding_norm, cosine_magnitude, scalar_head"
            )
        self.confidence_source = confidence_source
        self.num_partitions = num_partitions
        self.partition_dim = partition_dim

        # Always register a ModuleList so .parameters() yields the scalar
        # heads when used; for other confidence sources it is empty.
        if confidence_source == "scalar_head":
            self.scalar_heads = nn.ModuleList([
                nn.Linear(partition_dim, 1) for _ in range(num_partitions)
            ])
        else:
            self.scalar_heads = None

    def combine(
        self,
        partition_embeddings: list[Tensor | None],
        mask: Tensor | None = None,
        raw_embeddings: list[Tensor | None] | None = None,
        emb_norms: list[Tensor | None] | None = None,
    ) -> Tensor:
        """Return weighted combination embedding (B, num_partitions * K).

        ``raw_embeddings`` (pre-L2-normalisation) is required for a meaningful
        ``embedding_norm`` confidence signal — backbone outputs are unit
        vectors, so without raw features the norm is constant 1.0.
        """
        del mask  # unused; accepted for parity with other combiners
        ref = next(e for e in partition_embeddings if e is not None)
        B, K = ref.shape
        device = ref.device

        normed = []
        confidences = []

        for i, emb in enumerate(partition_embeddings):
            if emb is None:
                normed.append(torch.zeros(B, K, device=device, dtype=ref.dtype))
                confidences.append(torch.zeros(B, 1, device=device, dtype=ref.dtype))
                continue

            emb_f = emb.float()
            n = F.normalize(emb_f, dim=1, eps=1e-12)
            normed.append(n)

            if self.confidence_source == "embedding_norm":
                if emb_norms is not None and emb_norms[i] is not None:
                    norm_val = emb_norms[i].float()
                elif raw_embeddings is not None and raw_embeddings[i] is not None:
                    norm_val = raw_embeddings[i].float().norm(dim=1, keepdim=True)
                else:
                    # Fallback: incoming emb may already be unit-normalised, in
                    # which case the norm is uninformative — warn once via
                    # uniform weight (equivalent to CosineConcat).
                    norm_val = torch.ones(B, 1, device=device, dtype=ref.dtype)
                confidences.append(norm_val)
            elif self.confidence_source == "cosine_magnitude":
                # Per-pair magnitude is only knowable at scoring time; here
                # treat as uniform so the combiner is still well-defined.
                confidences.append(torch.ones(B, 1, device=device, dtype=ref.dtype))
            else:  # scalar_head
                assert self.scalar_heads is not None
                conf = torch.sigmoid(self.scalar_heads[i](emb_f))
                confidences.append(conf)

        # Normalise confidence weights to sum to 1 across partitions
        conf_stack = torch.stack(confidences, dim=1)  # (B, P, 1)
        conf_weights = conf_stack / (conf_stack.sum(dim=1, keepdim=True) + 1e-12)  # (B, P, 1)

        normed_stack = torch.stack(normed, dim=1)  # (B, P, K)
        weighted = (normed_stack * conf_weights).view(B, -1)
        return weighted


class LearnedCombiner:
    """Placeholder wrapping a trained Direction 1 PartitionCombiner.

    No retraining of Direction 2 partitions required. The combiner is loaded
    from a Direction 1 checkpoint and applied post-hoc over Direction 2's
    frozen partition outputs.

    Available as --combination learned_combiner only when a Direction 1
    combiner checkpoint is provided via --d1-combiner-path.
    """

    def __init__(self, combiner_checkpoint: str, device: torch.device) -> None:
        from ppi.combiner.mlp import PartitionCombiner
        ckpt = torch.load(combiner_checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("combiner_state_dict", ckpt)
        # Infer architecture from weight shapes
        # First layer: in_dim = num_partitions * partition_dim + num_partitions
        first_w = state.get("net.0.weight")
        if first_w is None:
            raise ValueError("Combiner checkpoint missing 'net.0.weight'.")
        in_dim = first_w.shape[1]
        hidden_dim = first_w.shape[0]
        # Last layer: output_dim
        last_key = sorted(k for k in state if k.startswith("net.") and "weight" in k)[-1]
        output_dim = state[last_key].shape[0]

        # Recover num_partitions and partition_dim from in_dim:
        # in_dim = P * K + P → solve for P and K using the checkpoint's partition dim
        # We attempt common K values
        num_partitions = ckpt.get("num_partitions")
        partition_dim = ckpt.get("partition_dim")
        if num_partitions is None or partition_dim is None:
            # Fallback: iterate largest-to-smallest so e.g. P=4, K=64
            # (in_dim=260) doesn't degenerate into P=2, K=129.
            for P in range(9, 1, -1):
                if (in_dim - P) % P == 0:
                    candidate_K = (in_dim - P) // P
                    if candidate_K > 0:
                        num_partitions = P
                        partition_dim = candidate_K
                        break
            if num_partitions is None:
                raise ValueError(
                    f"Cannot infer (P, K) from in_dim={in_dim}; pass "
                    "num_partitions/partition_dim in the checkpoint."
                )

        self.combiner = PartitionCombiner(
            num_partitions=num_partitions,
            partition_dim=partition_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ).to(device)
        self.combiner.load_state_dict(state)
        self.combiner.eval()
        self.device = device

    def combine(
        self,
        partition_embeddings: list[Tensor | None],
        mask: Tensor | None = None,
    ) -> Tensor:
        ref = next(e for e in partition_embeddings if e is not None)
        B, K = ref.shape
        if mask is None:
            # Build a mask where present partitions are 1, absent are 0.
            mask = torch.tensor(
                [[0.0 if e is None else 1.0 for e in partition_embeddings]] * B,
                device=self.device,
                dtype=ref.dtype,
            )
        filled = []
        for emb in partition_embeddings:
            if emb is None:
                filled.append(torch.zeros(B, K, device=self.device, dtype=ref.dtype))
            else:
                filled.append(F.normalize(emb.float(), dim=1, eps=1e-12))
        concat = torch.cat(filled, dim=1)
        with torch.no_grad():
            return self.combiner(concat, mask)


def get_combiner(
    strategy: str,
    **kwargs,
) -> CosineConcat | ConfidenceWeighted | LearnedCombiner:
    """Factory. strategy: 'cosine_concat' | 'confidence_weighted' | 'learned_combiner'."""
    if strategy == "cosine_concat":
        return CosineConcat()
    elif strategy == "confidence_weighted":
        return ConfidenceWeighted(
            confidence_source=kwargs.get("confidence_source", "embedding_norm"),
            num_partitions=kwargs.get("num_partitions", 3),
            partition_dim=kwargs.get("partition_dim", 128),
        )
    elif strategy == "learned_combiner":
        d1_path = kwargs.get("d1_combiner_path")
        if not d1_path:
            raise ValueError(
                "--d1-combiner-path must be provided when using learned_combiner combination."
            )
        device = kwargs.get("device", torch.device("cpu"))
        return LearnedCombiner(combiner_checkpoint=d1_path, device=device)
    else:
        raise ValueError(
            f"Unknown combination strategy '{strategy}'. "
            "Choose from: cosine_concat, confidence_weighted, learned_combiner"
        )
