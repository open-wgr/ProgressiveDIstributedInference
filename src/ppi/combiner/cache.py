"""Embedding cache: run frozen backbone over CASIA once, persist partition outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ppi.backbones import build_backbone
from ppi.data import build_dataloader
from ppi.utils.logging import ExperimentLogger


class EmbeddingCache:
    """Build and load a cache of raw partition embeddings from a frozen checkpoint.

    Saves two files to cache_dir:
      embeddings.pt  — shape (N, num_partitions, K), float32, as output by backbone
      labels.pt      — shape (N,), int64

    The backbone outputs are stored as-is (already per-partition L2-normalised by
    the partition head).  Per-partition normalisation before concat is deferred to
    the dataset, so both K and 3K combiner variants share the same cache files.
    """

    def __init__(
        self,
        config: dict[str, Any],
        checkpoint_path: str,
        cache_dir: str,
        batch_size: int = 256,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def build(self) -> Path:
        """Run backbone over CASIA training set; save embeddings.pt and labels.pt.

        Skips the build entirely if embeddings.pt already exists.
        Returns the cache directory Path.
        """
        emb_path = self.cache_dir / "embeddings.pt"
        if emb_path.exists():
            print(f"[EmbeddingCache] Cache exists at {self.cache_dir}, skipping build.")
            return self.cache_dir

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[EmbeddingCache] Building cache at {self.cache_dir} ...")

        # Frozen backbone — no gradients computed during the loop
        backbone = build_backbone(self.config).to(self.device)
        ckpt = ExperimentLogger.load_checkpoint(self.checkpoint_path)
        backbone.load_state_dict(ckpt["model_state_dict"]["backbone"])
        backbone.eval()
        backbone.requires_grad_(False)

        # Use the backbone config's data settings, but override batch size
        cache_config = {
            **self.config,
            "training": {**self.config.get("training", {}), "batch_size": self.batch_size},
        }
        train_loader, _ = build_dataloader(cache_config, split="train")

        all_embeddings: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total = len(train_loader)

        with torch.no_grad():
            for i, (images, labels) in enumerate(train_loader):
                if i % 50 == 0:
                    print(f"  [EmbeddingCache] Batch {i}/{total} ...", flush=True)
                images = images.to(self.device)
                out = backbone(images)
                # Stack partition list → (B, num_partitions, K); move to CPU immediately
                parts = torch.stack(out["partitions"], dim=1).cpu()
                all_embeddings.append(parts)
                all_labels.append(labels.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)   # (N, num_partitions, K)
        labels_t = torch.cat(all_labels, dim=0)          # (N,)

        torch.save(embeddings, self.cache_dir / "embeddings.pt")
        torch.save(labels_t, self.cache_dir / "labels.pt")
        print(
            f"[EmbeddingCache] Saved: embeddings {tuple(embeddings.shape)}, "
            f"labels {tuple(labels_t.shape)} → {self.cache_dir}"
        )
        return self.cache_dir

    @staticmethod
    def load(cache_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (embeddings, labels) tensors from a built cache."""
        d = Path(cache_dir)
        embeddings = torch.load(d / "embeddings.pt", weights_only=True)
        labels = torch.load(d / "labels.pt", weights_only=True)
        return embeddings, labels
