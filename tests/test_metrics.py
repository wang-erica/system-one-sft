import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, top_k_accuracy_score

from src import metrics


@pytest.fixture
def random_predictions():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 20, size=500)
    logits = rng.normal(size=(500, 20))
    logits[np.arange(500), labels] += rng.normal(1.5, 1.0, size=500)  # better than chance
    return labels, logits


def test_matches_scikit_learn(random_predictions):
    labels, logits = random_predictions
    predictions = logits.argmax(axis=1)
    result = metrics.compute_metrics(labels, logits)
    assert result["mean_per_class_accuracy"] == pytest.approx(balanced_accuracy_score(labels, predictions))
    assert result["macro_f1"] == pytest.approx(f1_score(labels, predictions, average="macro"))
    assert result["top1_accuracy"] == pytest.approx((predictions == labels).mean())
    assert result["top5_accuracy"] == pytest.approx(
        top_k_accuracy_score(labels, logits, k=5, labels=np.arange(20))
    )
    assert result["nll"] == pytest.approx(log_loss(labels, metrics.softmax(logits), labels=np.arange(20)))


def test_mean_per_class_accuracy_differs_from_top1_when_unbalanced():
    labels = np.array([0, 0, 0, 1])
    predictions = np.array([0, 0, 0, 0])
    assert metrics.mean_per_class_accuracy(labels, predictions) == pytest.approx(0.5)
    assert (predictions == labels).mean() == pytest.approx(0.75)


def test_ece_hand_computed():
    # Two images at confidence 0.9 (one correct), two at 0.6 (both correct).
    probabilities = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1, 0, 1])
    # Bin with 0.9: |0.5 - 0.9| = 0.4, weight 0.5. Bin with 0.6: |1.0 - 0.6| = 0.4, weight 0.5.
    assert metrics.expected_calibration_error(labels, probabilities) == pytest.approx(0.4)


def test_ece_zero_when_calibrated():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert metrics.expected_calibration_error(np.array([0, 1]), probabilities) == pytest.approx(0.0)


def test_ece_bin_edges_are_right_closed():
    # Confidence exactly 0.6 with 15 bins falls in bin 9 = (0.6 - 1/15, 0.6], same as 0.59.
    probabilities = np.array([[0.6, 0.4], [0.59, 0.41]])
    labels = np.array([0, 0])
    expected = abs(1.0 - (0.6 + 0.59) / 2)
    assert metrics.expected_calibration_error(labels, probabilities) == pytest.approx(expected)


def test_confusion_matrix():
    matrix = metrics.confusion_matrix(np.array([0, 1, 1, 2]), np.array([0, 2, 1, 2]), 3)
    assert matrix.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


def test_softmax_is_stable_for_large_logits():
    probabilities = metrics.softmax(np.array([[1000.0, 0.0], [-1000.0, -1000.0]]))
    assert np.allclose(probabilities, [[1.0, 0.0], [0.5, 0.5]])
