"""Evaluator: run all partition configurations against benchmarks."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ppi.backbones import build_backbone
from ppi.heads.arcface import ArcFaceHead
from ppi.training.partition_dropout import assemble_embedding
from ppi.utils.logging import ExperimentLogger


def _all_partition_configs(num_partitions: int = 3) -> list[set[int]]:
    """Return all 7 non-degenerate partition configurations."""
    indices = list(range(num_partitions))
    configs = []
    for r in range(1, num_partitions + 1):
        for combo in combinations(indices, r):
            configs.append(set(combo))
    return configs


class Evaluator:
    """Evaluate a checkpoint at multiple partition configurations."""

    def __init__(self, config: dict[str, Any], checkpoint_path: str) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build model
        self.backbone = build_backbone(config).to(self.device)

        # Load checkpoint
        ckpt = ExperimentLogger.load_checkpoint(checkpoint_path)
        self.backbone.load_state_dict(ckpt["model_state_dict"]["backbone"])
        self.backbone.eval()

        self.num_partitions = config["partitions"]["num_partitions"]
        self.K = config["partitions"]["K"]

    @torch.no_grad()
    def extract_embeddings(
        self,
        dataloader,
        active_partitions: set[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract embeddings with partition masking.

        Returns (embeddings, labels) as numpy arrays.
        """
        all_embs = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)
            out = self.backbone(images)
            partition_outputs = out["partitions"]

            # Mask inactive partitions
            masked = []
            for idx, p in enumerate(partition_outputs):
                if idx in active_partitions:
                    masked.append(p)
                else:
                    masked.append(torch.zeros_like(p))

            embedding = assemble_embedding(masked)
            all_embs.append(embedding.cpu().numpy())
            all_labels.append(labels.numpy())

        return np.concatenate(all_embs), np.concatenate(all_labels)

    def evaluate(
        self,
        partition_configs: list[set[int]] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run evaluation at specified partition configurations.

        Parameters
        ----------
        partition_configs:
            List of partition index sets to evaluate. Defaults to all 7
            non-degenerate configs.

        Returns
        -------
        dict mapping config name to metric dict.
        """
        if partition_configs is None:
            partition_configs = _all_partition_configs(self.num_partitions)

        from ppi.data import build_dataloader
        val_loader, num_classes = build_dataloader(self.config, split="val")

        results = {}
        for config_set in partition_configs:
            config_name = "P" + "".join(str(i) for i in sorted(config_set))
            embeddings, labels = self.extract_embeddings(val_loader, config_set)

            # For CIFAR-100 / classification tasks: top-1 accuracy
            # Build a simple nearest-centroid classifier
            unique_labels = np.unique(labels)
            centroids = np.zeros((len(unique_labels), embeddings.shape[1]))
            for i, lbl in enumerate(unique_labels):
                centroids[i] = embeddings[labels == lbl].mean(axis=0)

            # Normalise
            emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
            cen_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
            sim = emb_norm @ cen_norm.T
            preds = unique_labels[sim.argmax(axis=1)]
            accuracy = float((preds == labels).mean())

            results[config_name] = {"accuracy": accuracy}

        return results
