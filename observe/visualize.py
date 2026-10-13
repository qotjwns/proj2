#!/usr/bin/env python3
"""Visualize all saved confidence scores and selective-classification results."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observe.metrics import COVERAGE_PERCENTAGES, SCORE_NAMES

COLORS = {"drainage": "#0072B2", "msp": "#D55E00", "margin": "#009E73"}
STYLES = {"drainage": "-", "msp": "--", "margin": "-."}
CORRECT_COLOR, ERROR_COLOR = "#0072B2", "#D55E00"
DEFINITIONS = {
    "drainage": (
        r"$s_D = 1 - p_d$",
        r"$p_d = \mathrm{softmax}(z_1,\ldots,z_{10},z_d)_d$",
        "Softmax over all 11 outputs;\nsubtract Drainage probability from 1.",
    ),
    "msp": (
        r"$s_{MSP} = \max_{c=1,\ldots,10} p_c$",
        r"$p = \mathrm{softmax}(z_1,\ldots,z_{10})$",
        "Softmax over the 10 class outputs;\ntake the largest probability.",
    ),
    "margin": (
        r"$s_M = z_{(1)} - z_{(2)}$",
        r"$z_{(1)} \geq z_{(2)} \geq \cdots \geq z_{(10)}$",
        "Largest minus second-largest class logit;\nexclude Drainage.",
    ),
}


def title(summary):
    accuracy = summary["results"][0]["accuracy"]
    return f"Clean CIFAR-10 · N={summary['num_samples']:,} · Fixed accuracy={accuracy:.2%}"


def format_axis(axis, *, x_percent=False, y_percent=False):
    from matplotlib.ticker import PercentFormatter

    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    if x_percent:
        axis.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    if y_percent:
        axis.yaxis.set_major_formatter(PercentFormatter(1))


def definition_card(axis, key, *, compact=False):
    formula, detail, explanation = DEFINITIONS[key]
    axis.set_axis_off()
    axis.set_title(SCORE_NAMES[key], color=COLORS[key], fontweight="bold", fontsize=14)
    axis.text(0.5, 0.76, formula, transform=axis.transAxes, ha="center", va="center", fontsize=19)
    axis.text(0.5, 0.39, detail, transform=axis.transAxes, ha="center", va="center", fontsize=12)
    if not compact:
        axis.text(0.5, 0.02, explanation, transform=axis.transAxes,
                  ha="center", va="center", fontsize=11, linespacing=1.6)


def summary_table(axis, summary):
    axis.set_axis_off()
    rows = summary["results"]
    columns = ["score", "accuracy", "risk_at_90", "risk_at_95", "risk_at_99", "aurc"]
    labels = ["Score", "Accuracy ↑", "Risk @90% ↓", "Risk @95% ↓", "Risk @99% ↓", "AURC ↓"]
    cells = [[row["score"], f"{row['accuracy']:.2%}",
              *(f"{row[f'risk_at_{p}']:.4%}" for p in COVERAGE_PERCENTAGES),
              f"{row['aurc']:.6f}"] for row in rows]
    table = axis.table(cellText=cells, colLabels=labels, cellLoc="center",
                       colWidths=[0.20, 0.16, 0.16, 0.16, 0.16, 0.16], bbox=[0, 0.1, 1, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DDE3EA")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#EAF0F6")
            cell.get_text().set_fontweight("bold")
        elif column == 0:
            cell.get_text().set_color(list(COLORS.values())[row - 1])
            cell.get_text().set_fontweight("bold")
        elif column == 1:
            cell.set_facecolor("#F4F6F8")
        elif rows[row - 1][columns[column]] == min(r[columns[column]] for r in rows):
            cell.set_facecolor("#DDF2E8")
            cell.get_text().set_fontweight("bold")
    axis.set_title("One fixed prediction per image · Green cells: lowest risk / AURC", fontsize=11)


def score_distribution(axis, key, samples, *, compact=False):
    values = samples[key]
    errors = samples["errors"].astype(bool)
    upper = 1.0 if key != "margin" else max(1e-9, float(values.max()) * 1.001)
    edges = np.linspace(0, upper, 41)
    for mask, name, color, style in (
        (~errors, "Correct", CORRECT_COLOR, "-"), (errors, "Incorrect", ERROR_COLOR, "--"),
    ):
        count = int(mask.sum())
        heights = np.histogram(values[mask], bins=edges)[0] / max(1, count)
        axis.stairs(heights, edges, fill=True, alpha=0.14, color=color, linewidth=0)
        axis.stairs(heights, edges, color=color, linestyle=style, linewidth=1.6,
                    label=f"{name} (n={count:,})")
    axis.set_title(SCORE_NAMES[key], color=COLORS[key], fontsize=13, fontweight="bold")
    axis.set_xlabel("Confidence score (higher → accept)")
    axis.set_ylabel("Within-group share per bin")
    axis.set_xlim(0, upper)
    axis.set_ylim(bottom=0)
    format_axis(axis, y_percent=True)
    axis.legend(fontsize=8 if compact else 9, loc="upper right" if key == "margin" else "upper left")


def risk_curve(axis, summary, curves, *, high_coverage=False, show_legend=True):
    metrics = {row["score"]: row for row in summary["results"]}
    baseline = 1.0 - summary["results"][0]["accuracy"]
    for key, name in SCORE_NAMES.items():
        axis.plot(curves["coverage"], curves[key], color=COLORS[key], linestyle=STYLES[key],
                  linewidth=1.8, label=f"{name} (AURC {metrics[name]['aurc']:.5f})")
    axis.axhline(baseline, color="#777777", linestyle=":", linewidth=1.2,
                 label="Random rejection (expected)")
    axis.set_xlabel("Coverage (accepted fraction)")
    axis.set_ylabel("Risk (error rate among accepted)")
    format_axis(axis, x_percent=True, y_percent=True)
    if high_coverage:
        axis.set_title("High coverage: 90–100%")
        axis.set_xlim(0.9, 1.0)
        axis.set_xticks([0.9, 0.95, 0.99, 1.0])
        high = curves["coverage"] >= 0.90
        high_max = max(float(curves[key][high].max()) for key in SCORE_NAMES)
        axis.set_ylim(0, min(1.05, max(0.01, high_max * 1.12)))
        for count in summary["coverage_counts"].values():
            index = count - 1
            axis.axvline(curves["coverage"][index], color="#AAAAAA", linewidth=0.7, alpha=0.5)
            for key in SCORE_NAMES:
                axis.scatter(curves["coverage"][index], curves[key][index], color=COLORS[key], s=24, zorder=4)
    else:
        axis.set_title("Full risk–coverage curve")
        axis.set_xlim(0, 1)
        axis.set_ylim(bottom=0)
    if show_legend:
        axis.legend(loc="best", fontsize=9)


def save_figure(figure, stem, output_dir, dpi, plt):
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {path.resolve()}", flush=True)
    plt.close(figure)


def plot_all(summary, curves, samples, output_dir, dpi, plt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for axis, key in zip(axes, SCORE_NAMES):
        definition_card(axis, key)
    fig.suptitle("Three confidence scores · Higher values are accepted first\n"
                 r"Fixed class prediction: $\hat{y}=\arg\max_{c=1,\ldots,10}z_c$", fontsize=15)
    save_figure(fig, "score_definitions", output_dir, dpi, plt)

    fig = plt.figure(figsize=(14, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.8], width_ratios=[2, 1])
    summary_table(fig.add_subplot(grid[0, :]), summary)
    risks_axis, aurc_axis = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    positions = np.arange(len(COVERAGE_PERCENTAGES))
    metrics = {row["score"]: row for row in summary["results"]}
    for offset, (key, name) in enumerate(SCORE_NAMES.items()):
        risks = [metrics[name][f"risk_at_{p}"] for p in COVERAGE_PERCENTAGES]
        bars = risks_axis.bar(positions + (offset - 1) * 0.25, risks, width=0.23,
                             color=COLORS[key], label=name)
        risks_axis.bar_label(bars, labels=[f"{value * 100:.2f}" for value in risks], padding=4, fontsize=9)
    risks_axis.set_xticks(positions, [f"{p}%" for p in COVERAGE_PERCENTAGES])
    risks_axis.set_xlabel("Coverage (accepted fraction)")
    risks_axis.set_ylabel("Accepted error rate (%)")
    risks_axis.set_title("Risk at the requested coverages ↓")
    max_risk = max(row[f"risk_at_{p}"] for row in metrics.values() for p in COVERAGE_PERCENTAGES)
    risks_axis.set_ylim(0, max(0.01, max_risk * 1.3))
    risks_axis.legend(fontsize=9, loc="upper left", ncols=3)
    format_axis(risks_axis, y_percent=True)
    aurcs = [metrics[name]["aurc"] for name in SCORE_NAMES.values()]
    bars = aurc_axis.bar(list(SCORE_NAMES.values()), aurcs, color=list(COLORS.values()), width=0.65)
    aurc_axis.bar_label(bars, labels=[f"{value:.6f}" for value in aurcs], padding=5, fontsize=10)
    aurc_axis.set_ylim(0, max(0.001, max(aurcs) * 1.3))
    aurc_axis.set_title("AURC ↓")
    aurc_axis.set_ylabel("Discrete mean of risk (fraction)")
    format_axis(aurc_axis)
    fig.suptitle(title(summary) + "\nIdentical accuracy for all three selectors · Tie-aware expected risk", fontsize=14)
    save_figure(fig, "metrics_comparison", output_dir, dpi, plt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for axis, key in zip(axes, SCORE_NAMES):
        score_distribution(axis, key, samples)
    fig.suptitle("Correct vs. incorrect prediction: confidence-score distributions\n"
                 "Each group is normalized separately to 100% · 40 equal-width bins · All test samples", fontsize=14)
    save_figure(fig, "score_distributions", output_dir, dpi, plt)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), layout="constrained")
    errors = samples["errors"].astype(bool)
    ranks = np.arange(1, len(errors) + 1) / len(errors)
    for axis, key in zip(axes, SCORE_NAMES):
        order = samples[key + "_order"]
        ordered_errors = errors[order]
        for mask, name, color, alpha in (
            (~ordered_errors, "Correct", CORRECT_COLOR, 0.30),
            (ordered_errors, "Incorrect", ERROR_COLOR, 0.60),
        ):
            axis.scatter(ranks[mask], samples[key][order][mask], color=color, alpha=alpha,
                         s=5, linewidths=0, label=name, rasterized=True)
        for p in COVERAGE_PERCENTAGES:
            count = summary["coverage_counts"][str(p)]
            axis.axvline(count / len(errors), color="#777777", linestyle=":", linewidth=0.9)
        axis.set_title(SCORE_NAMES[key], color=COLORS[key], fontweight="bold")
        axis.set_xlabel("Rank / N after descending confidence sort (high confidence ← → low confidence)")
        axis.set_ylabel("Confidence score")
        axis.set_xlim(0, 1)
        axis.set_ylim(bottom=0)
        format_axis(axis, x_percent=True)
        axis.legend(loc="upper right", markerscale=3, fontsize=9)
    fig.suptitle("Every test sample, sorted separately by each confidence score\n"
                 "Orange points are errors · Vertical lines: 90%, 95%, 99% · Within-tie display order is arbitrary", fontsize=14)
    save_figure(fig, "sorted_scores", output_dir, dpi, plt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    risk_curve(axes[0], summary, curves)
    risk_curve(axes[1], summary, curves, high_coverage=True, show_legend=False)
    fig.suptitle(title(summary) + "\nLower risk is better · Ties use expected risk · AURC = mean of risks at k=1,...,N",
                 fontsize=12)
    save_figure(fig, "risk_coverage", output_dir, dpi, plt)

    # One summary image plus larger individual figures for close inspection.
    fig = plt.figure(figsize=(16, 14), layout="constrained")
    grid = fig.add_gridspec(4, 6, height_ratios=[1.0, 1.1, 2.1, 2.5])
    for i, key in enumerate(SCORE_NAMES):
        definition_card(fig.add_subplot(grid[0, i * 2:(i + 1) * 2]), key, compact=True)
    summary_table(fig.add_subplot(grid[1, :]), summary)
    for i, key in enumerate(SCORE_NAMES):
        score_distribution(fig.add_subplot(grid[2, i * 2:(i + 1) * 2]), key, samples, compact=True)
    risk_curve(fig.add_subplot(grid[3, :3]), summary, curves)
    risk_curve(fig.add_subplot(grid[3, 3:]), summary, curves, high_coverage=True, show_legend=False)
    fig.suptitle(title(summary) + "\nThree confidence scores · Distributions, selective risk and AURC", fontsize=16)
    save_figure(fig, "overview", output_dir, dpi, plt)


def write_gallery(summary, output_dir):
    sections = [
        ("overview", "한눈에 보는 전체 결과", "점수 정의, 결과 표, 정답·오답 점수 분포, Risk–coverage 곡선을 모았습니다."),
        ("score_definitions", "세 점수의 계산 방식", "Drainage는 11개 출력, MSP와 Logit margin은 클래스 10개 출력으로 계산합니다. 모두 클수록 먼저 수용합니다."),
        ("metrics_comparison", "평가 결과 표와 지표 비교", "전체 정확도는 동일합니다. 초록색 표 칸은 가장 낮은 오류율·AURC입니다. 막대 위 오류율 값은 %이고 AURC는 0~1 비율입니다."),
        ("score_distributions", "정답·오답의 점수 분포", "정답 그룹과 오답 그룹을 각각 100%로 정규화했습니다. 각 패널은 40개 동일 너비 구간을 사용하며, y축은 해당 그룹에서 그 구간에 속하는 비율입니다. 점수별 x축 단위가 다르므로 분포 모양과 겹침을 살펴보세요."),
        ("sorted_scores", "전체 이미지의 점수순 정렬", "점수별로 높은 순서대로 모든 평가 이미지를 표시했습니다. 파란 점은 정답, 주황 점은 오답입니다. 오답이 오른쪽에 몰릴수록 거절에 유리합니다. 점선은 90·95·99% 수용 위치이며, 동점 내 점 표시 순서는 임의이고 평가지표는 기대 오류율을 사용합니다."),
        ("risk_coverage", "Risk–coverage 곡선", "왼쪽은 전체 수용률, 오른쪽은 90~100% 구간입니다. 점선 수평선은 무작위 거절의 기대 오류율입니다. AURC는 k=1,…,N에서의 기대 오류율 이산 평균입니다."),
    ]
    lines = ["# Drainage · MSP · Logit margin 전체 시각화", "",
             f"CIFAR-10 평가 이미지 {summary['num_samples']:,}개 · 전체 정확도 {summary['results'][0]['accuracy']:.2%}", "",
             "저장된 평가 결과를 사용했습니다. 서비스 임계값은 별도 validation에서 정해야 합니다.", ""]
    for stem, heading, explanation in sections:
        lines.extend([f"## {heading}", "", explanation, "", f"![{heading}]({stem}.png)", "",
                      f"[PNG 원본]({stem}.png) · [PDF]({stem}.pdf)", ""])
    path = output_dir / "visualizations.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "observe/results/noise_0.45")
    parser.add_argument("--output-dir", type=Path, help="Default: same as --results-dir.")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive.")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "observe/.mplconfig"))
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit("Install plotting dependencies: python -m pip install -r observe/requirements.txt") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((args.results_dir / "metrics.json").read_text(encoding="utf-8"))
    with np.load(args.results_dir / "curves.npz", allow_pickle=False) as saved:
        curves = {key: saved[key] for key in saved.files}
    with np.load(args.results_dir / "scores.npz", allow_pickle=False) as saved:
        samples = {key: saved[key] for key in saved.files}
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        plot_all(summary, curves, samples, output_dir, args.dpi, plt)
    write_gallery(summary, output_dir)


if __name__ == "__main__":
    main()
