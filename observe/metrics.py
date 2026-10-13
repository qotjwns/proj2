"""Confidence scores and exact expected risk under uniformly ordered ties."""

import numpy as np


SCORE_NAMES = {"drainage": "Drainage", "msp": "MSP", "margin": "Logit margin"}
COVERAGE_PERCENTAGES = (90, 95, 99)


def confidence_scores(logits, drainage_idx):
    """Return one fixed class prediction and three higher-is-better scores.

    The ten non-drainage columns are in CIFAR-10 label order. Compute scores
    in float64 without rounding or clipping; retain the original logits in
    the inference cache separately.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 11 or len(logits) == 0:
        raise ValueError("Expected nonempty CIFAR-10 logits with shape [N, 11].")
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain NaN or infinity.")
    if not isinstance(drainage_idx, (int, np.integer)) or not -11 <= drainage_idx < 11:
        raise ValueError("drainage_idx must index one of the 11 output columns.")
    drainage_idx %= 11
    class_logits = np.delete(logits, drainage_idx, axis=1)
    predictions = class_logits.argmax(axis=1)

    def softmax(values):
        exponentials = np.exp(values - values.max(axis=1, keepdims=True))
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    full_probabilities = softmax(logits)
    class_probabilities = softmax(class_logits)
    top_two = np.sort(class_logits, axis=1)[:, -2:]
    scores = {
        "drainage": 1.0 - full_probabilities[:, drainage_idx],
        "msp": class_probabilities.max(axis=1),
        "margin": top_two[:, 1] - top_two[:, 0],
    }
    return predictions, scores


def expected_risk_curve(scores, errors):
    """Return risks for k=1..N and a score-only descending ordering.

    If a tied group of size m contains e errors, and a prefix accepts t of
    that group after a samples containing E errors, its expected risk is
    (E + t * e / m) / (a + t). This averages all within-tie permutations
    analytically, including ties crossing a requested coverage boundary.
    The returned order is for inspection; its arbitrary tie order is never
    used to decide which tied errors enter a risk prefix.
    """
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(errors)
    if scores.ndim != 1 or errors.shape != scores.shape or not len(scores):
        raise ValueError("Scores and errors must be nonempty matching 1D arrays.")
    if not np.isfinite(scores).all() or not np.isin(errors, [0, 1]).all():
        raise ValueError("Scores must be finite and errors must be binary.")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(scores)]
    sizes = ends - starts
    error_prefix = np.r_[0, np.cumsum(errors[order], dtype=np.int64)]
    group_errors = error_prefix[ends] - error_prefix[starts]
    accepted = np.arange(1, len(scores) + 1)
    within_group = accepted - np.repeat(starts, sizes)
    expected_errors = (
        np.repeat(error_prefix[starts], sizes)
        + within_group * np.repeat(group_errors / sizes, sizes)
    )
    return expected_errors / accepted, order


def evaluate_logits(logits, labels, drainage_idx):
    """Evaluate three selectors with identical predictions and accuracy."""
    predictions, scores = confidence_scores(logits, drainage_idx)
    labels = np.asarray(labels)
    if labels.shape != predictions.shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be an integer vector matching the logits.")
    if not ((labels >= 0) & (labels < 10)).all():
        raise ValueError("CIFAR-10 labels must be between 0 and 9.")
    errors = predictions != labels
    n = len(labels)
    accuracy = float(1.0 - errors.mean())
    curves = {"accepted_count": np.arange(1, n + 1), "coverage": np.arange(1, n + 1) / n}
    samples = {"labels": labels, "predictions": predictions, "errors": errors}
    rows = []
    for key, name in SCORE_NAMES.items():
        risk, order = expected_risk_curve(scores[key], errors)
        curves[key] = risk
        samples[key] = scores[key]
        samples[key + "_order"] = order
        row = {"score": name, "accuracy": accuracy}
        for percentage in COVERAGE_PERCENTAGES:
            accepted = max(1, n * percentage // 100)
            row[f"risk_at_{percentage}"] = float(risk[accepted - 1])
        # This is the requested discrete average, NOT trapezoidal integration.
        row["aurc"] = float(risk.mean())
        rows.append(row)
    return rows, curves, samples


def markdown_table(rows):
    lines = [
        "| 거절 점수 | 전체 정확도 ↑ | 오류율 @90% ↓ | 오류율 @95% ↓ | 오류율 @99% ↓ | AURC ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        risks = " | ".join(f"{row[f'risk_at_{p}']:.4%}" for p in COVERAGE_PERCENTAGES)
        lines.append(f"| {row['score']} | {row['accuracy']:.4%} | {risks} | {row['aurc']:.6f} |")
    return "\n".join(lines)
