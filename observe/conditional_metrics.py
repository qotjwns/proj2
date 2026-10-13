"""Compare a second confidence score inside quantile bins of a control score."""

import numpy as np


def quantile_bins(values, requested_bins):
    """Keep equal control scores together; drop duplicate cuts and empty bins."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Control scores must be a finite nonempty vector.")
    if requested_bins < 1:
        raise ValueError("The number of bins must be positive.")
    cuts = np.unique(np.quantile(values, np.arange(1, requested_bins) / requested_bins))
    cuts = cuts[(cuts > values.min()) & (cuts < values.max())]
    raw = np.searchsorted(cuts, values, side="left")
    _, assignments = np.unique(raw, return_inverse=True)
    return assignments


def tail_weights(values, count, *, high=False):
    """Expected membership in a fixed-size tail, with uniform selection in ties."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not 1 <= count <= len(values):
        raise ValueError("Invalid scores or tail size.")
    cutoff = np.partition(values, len(values) - count if high else count - 1)[
        len(values) - count if high else count - 1
    ]
    strict = values > cutoff if high else values < cutoff
    ties = values == cutoff
    weights = strict.astype(np.float64)
    weights[ties] = (count - int(strict.sum())) / int(ties.sum())
    return weights, float(cutoff)


def bootstrap_tail_risks(values, errors, count, repetitions, rng):
    """Resample whole samples within a fixed control bin; reselect both tails.

    Both tails use the same bootstrap draw. Tied memberships are averaged
    analytically instead of selecting tied samples by their correctness.
    """
    values, errors = np.asarray(values), np.asarray(errors, dtype=float)
    rates = np.empty((repetitions, 2), dtype=float)
    n = len(values)
    for start in range(0, repetitions, 128):
        stop = min(repetitions, start + 128)
        indices = rng.integers(0, n, size=(stop - start, n))
        sampled, sampled_errors = values[indices], errors[indices]
        partitioned = np.partition(sampled, [count - 1, n - count], axis=1)
        for column, high in enumerate((False, True)):
            cut = partitioned[:, n - count if high else count - 1, None]
            strict = sampled > cut if high else sampled < cut
            tied = sampled == cut
            expected_errors = (sampled_errors * strict).sum(axis=1)
            remaining = count - strict.sum(axis=1)
            expected_errors += remaining * (sampled_errors * tied).sum(axis=1) / tied.sum(axis=1)
            rates[start:stop, column] = expected_errors / count
    return rates


def intervals(rates):
    """Percentile intervals for each risk and their paired low-minus-high gap."""
    return {
        "low_ci": np.quantile(rates[:, 0], [0.025, 0.975]).tolist(),
        "high_ci": np.quantile(rates[:, 1], [0.025, 0.975]).tolist(),
        "gap_ci": np.quantile(rates[:, 0] - rates[:, 1], [0.025, 0.975]).tolist(),
    }


def conditional_comparison(control, target, errors, *, bins=10, tail_fraction=0.25,
                           repetitions=2000, seed=42):
    control = np.asarray(control, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    errors = np.asarray(errors)
    if control.shape != target.shape or control.shape != errors.shape:
        raise ValueError("Control scores, target scores and errors must have matching shapes.")
    if not np.isfinite(target).all() or not np.isin(errors, [0, 1]).all():
        raise ValueError("Target scores must be finite and errors binary.")
    if not 0 < tail_fraction <= 0.5 or repetitions < 1:
        raise ValueError("tail_fraction must be in (0, 0.5] and repetitions positive.")
    assignment = quantile_bins(control, bins)
    errors = errors.astype(float)
    rng = np.random.default_rng(seed)
    low_membership = np.zeros(len(control))
    high_membership = np.zeros(len(control))
    rows, bootstrap_rates = [], []
    for bin_index in np.unique(assignment):
        index = np.flatnonzero(assignment == bin_index)
        x, y, e = control[index], target[index], errors[index]
        count = int(np.floor(len(index) * tail_fraction))
        if count < 1:
            raise ValueError("A control bin is too small for the requested tails; reduce --bins.")
        low, low_cut = tail_weights(y, count)
        high, high_cut = tail_weights(y, count, high=True)
        low_membership[index], high_membership[index] = low, high
        low_errors, high_errors = float(low @ e), float(high @ e)
        rates = bootstrap_tail_risks(y, e, count, repetitions, rng)
        bootstrap_rates.append(rates)
        rows.append({
            "bin": int(bin_index + 1), "n": len(index), "bin_errors": int(e.sum()),
            "control_min": float(x.min()), "control_max": float(x.max()),
            "tail_n": count, "tail_fraction_actual": count / len(index),
            "target_low_cutoff": low_cut, "target_high_cutoff": high_cut,
            "target_has_variation": bool(y.min() < y.max()),
            "low_expected_errors": low_errors, "high_expected_errors": high_errors,
            "low_risk": low_errors / count, "high_risk": high_errors / count,
            "gap": (low_errors - high_errors) / count,
            "control_mean_low": float(low @ x / count),
            "control_mean_high": float(high @ x / count),
            "target_mean_low": float(low @ y / count),
            "target_mean_high": float(high @ y / count),
            **intervals(rates),
        })
    # Use exactly the same original bin weights for both tails (standardization).
    bin_weights = np.array([row["n"] for row in rows]) / len(control)
    low_risk = float(bin_weights @ np.array([row["low_risk"] for row in rows]))
    high_risk = float(bin_weights @ np.array([row["high_risk"] for row in rows]))
    aggregate_rates = np.einsum("b,brc->rc", bin_weights, np.array(bootstrap_rates))
    summary = {
        "actual_bins": len(rows), "n": len(control),
        "tail_n_total": sum(row["tail_n"] for row in rows),
        "low_risk": low_risk, "high_risk": high_risk, "gap": low_risk - high_risk,
        **intervals(aggregate_rates),
    }
    membership = {"bin": assignment + 1, "low_weight": low_membership, "high_weight": high_membership}
    return {"summary": summary, "bins": rows}, membership
