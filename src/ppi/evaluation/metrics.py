"""Evaluation metrics: TAR@FAR, rank-1, pair accuracy."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def compute_tar_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    far_target: float = 1e-4,
) -> float:
    """Compute TAR (True Accept Rate) at a given FAR (False Accept Rate).

    Parameters
    ----------
    genuine_scores : array of shape (N,)
        Cosine similarities for genuine pairs.
    impostor_scores : array of shape (M,)
        Cosine similarities for impostor pairs.
    far_target : float
        Target false accept rate.

    Returns
    -------
    float
        TAR at the threshold achieving the requested FAR.
    """
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    # Find the TPR at the closest FPR to far_target
    idx = np.searchsorted(fpr, far_target, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def compute_rank1(
    query_embs: np.ndarray,
    gallery_embs: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """Compute rank-1 identification rate via cosine similarity.

    Parameters
    ----------
    query_embs : array of shape (Q, D)
    gallery_embs : array of shape (G, D)
    query_labels : array of shape (Q,)
    gallery_labels : array of shape (G,)

    Returns
    -------
    float
        Fraction of queries whose top-1 gallery match has the correct label.
    """
    # Normalise
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-12)
    gallery_norm = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-12)
    sim = query_norm @ gallery_norm.T  # (Q, G)
    top1_indices = sim.argmax(axis=1)
    predicted = gallery_labels[top1_indices]
    return float((predicted == query_labels).mean())


def compute_pair_accuracy(
    embeddings1: np.ndarray,
    embeddings2: np.ndarray,
    issame: np.ndarray,
    n_folds: int = 10,
) -> tuple[float, float]:
    """K-fold pair verification accuracy (LFW-style protocol).

    Parameters
    ----------
    embeddings1, embeddings2 : arrays of shape (N, D)
        Paired embeddings.
    issame : array of shape (N,)
        Boolean labels.
    n_folds : int
        Number of cross-validation folds.

    Returns
    -------
    tuple[float, float]
        (mean_accuracy, std_accuracy) across folds.
    """
    n = len(issame)
    # Cosine similarity
    e1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-12)
    e2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-12)
    similarities = (e1 * e2).sum(axis=1)

    fold_size = n // n_folds
    indices = np.arange(n)
    accuracies = []

    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        # Find best threshold on train set
        best_acc = 0.0
        best_thresh = 0.0
        for thresh in np.linspace(-1, 1, 200):
            preds = similarities[train_idx] >= thresh
            acc = (preds == issame[train_idx]).mean()
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        # Evaluate on val set
        val_preds = similarities[val_idx] >= best_thresh
        val_acc = (val_preds == issame[val_idx]).mean()
        accuracies.append(val_acc)

    return float(np.mean(accuracies)), float(np.std(accuracies))
