"""Classification metrics computed from saved logits and true labels.

All functions take ``labels`` of shape ``(N,)`` (integer class indices) and, where needed,
``logits`` of shape ``(N, C)`` in class-index order.
"""

from __future__ import annotations

import numpy as np

PRIMARY_METRIC = {"flowers102": "mean_per_class_accuracy", "food101": "top1_accuracy"}
ECE_BINS = 15


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def top_k_accuracy(labels: np.ndarray, logits: np.ndarray, k: int = 1) -> float:
    top_k = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    return float((top_k == labels[:, None]).any(axis=1).mean())


def mean_per_class_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Average of per-class recall over the classes present in ``labels``."""
    classes = np.unique(labels)
    return float(np.mean([(predictions[labels == c] == c).mean() for c in classes]))


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Unweighted mean F1 over classes present in labels or predictions (scikit-learn's default)."""
    scores = []
    for c in np.union1d(labels, predictions):
        true_positive = np.sum((predictions == c) & (labels == c))
        predicted = np.sum(predictions == c)
        actual = np.sum(labels == c)
        denominator = predicted + actual
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = ECE_BINS) -> float:
    """Top-label ECE with equal-width bins (Guo et al., 2017). Bin m covers ((m-1)/M, m/M]."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    bin_index = np.clip(np.ceil(confidence * bins).astype(int) - 1, 0, bins - 1)
    error = 0.0
    for m in range(bins):
        in_bin = bin_index == m
        if in_bin.any():
            error += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


def negative_log_likelihood(labels: np.ndarray, logits: np.ndarray) -> float:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probabilities[np.arange(len(labels)), labels].mean())


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    predictions = logits.argmax(axis=1)
    return {
        "num_examples": int(len(labels)),
        "num_classes_present": int(len(np.unique(labels))),
        "mean_per_class_accuracy": mean_per_class_accuracy(labels, predictions),
        "top1_accuracy": top_k_accuracy(labels, logits, 1),
        "top5_accuracy": top_k_accuracy(labels, logits, 5),
        "macro_f1": macro_f1(labels, predictions),
        "ece": expected_calibration_error(labels, softmax(logits)),
        "nll": negative_log_likelihood(labels, logits),
    }
