#!/usr/bin/env python3
"""Compare Drainage and MSP conditionally using the cached per-image scores."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observe.conditional_metrics import conditional_comparison
from observe.metrics import SCORE_NAMES

DIRECTIONS = [("msp", "drainage"), ("drainage", "msp")]
LOW_COLOR, HIGH_COLOR = "#D55E00", "#0072B2"


def flatten(row):
    result = {}
    for name, value in row.items():
        if name.endswith("_ci"):
            result[name + "_lower"], result[name + "_upper"] = value
        else:
            result[name] = value
    return result


def write_csv(path, rows):
    rows = [flatten(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_range(row):
    lower, upper = row["control_min"], row["control_max"]
    if lower >= 0.99:
        return f"1−s: {1 - upper:.3g} ~ {1 - lower:.3g}"
    return f"{lower:.6f} ~ {upper:.6f}"


def summary_table(results):
    lines = [
        "| 묶는 기준 | 비교하는 점수 | 하위 그룹 오류율 | 상위 그룹 오류율 | 차이: 하위−상위 | 차이의 95% 구간 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        row = result["summary"]
        interval = row["gap_ci"]
        lines.append(
            f"| {SCORE_NAMES[result['control']]} | {SCORE_NAMES[result['target']]} | "
            f"{row['low_risk']:.2%} | {row['high_risk']:.2%} | {100 * row['gap']:+.2f}%p | "
            f"[{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}]%p |"
        )
    return "\n".join(lines)


def write_report(report, output_dir):
    setup, results = report["setup"], report["directions"]
    tail = setup["tail_fraction"]
    lines = ["# MSP와 Drainage의 조건부 오류율 비교", "",
             f"평가 이미지 {setup['n']:,}개. 기준 점수를 낮은 순서로 {setup['bins']}개 분위 구간에 나누고, "
             f"각 구간에서 다른 점수의 하위·상위 {tail:.0%}를 비교합니다.", "",
             summary_table(results), "",
             "양수 차이는 비교 점수가 높은 그룹에서 오답이 더 적다는 뜻입니다. "
             "전체 요약은 기준 구간의 표본 수/N을 양쪽 그룹에 동일하게 적용한 가중 평균입니다.", "",
             "![구간별 오류율과 차이](conditional_comparison.png)", "",
             "[PNG 원본](conditional_comparison.png) · [PDF](conditional_comparison.pdf)", "",
             "## 해석 범위", "",
             "같은 분위 구간이어도 기준 점수가 정확히 같은 것은 아닙니다. 두 그룹의 기준 점수 평균이 "
             "다르면 기준 점수 자체의 영향이 남을 수 있으므로, 이 결과만으로 독립적인 추가 예측 정보를 확정하지 않습니다.", "",
             "아래 그림의 회색 선은 각 구간의 기준 점수 최솟값~최댓값이고, 두 점은 비교 점수 하위·상위 그룹의 "
             "기준 점수 평균입니다. 두 점이 떨어져 있을수록 기준 점수의 차이가 남아 있습니다.", "",
             "![기준 점수 균형](control_balance.png)", "",
             "[PNG 원본](control_balance.png) · [PDF](control_balance.pdf)", ""]
    for result in results:
        lines.extend([
            f"## {SCORE_NAMES[result['control']]}가 비슷할 때 {SCORE_NAMES[result['target']]} 비교", "",
            "B1은 가장 낮은 기준 점수 구간입니다. 점수가 1에 매우 가까운 구간은 `1−s` 범위로 표시합니다.", "",
            "| 구간 | 기준 점수 범위 | 전체 n / 오답 | 그룹별 n | 하위 오류율 | 상위 오류율 | 차이 (%p) | 95% 구간 (%p) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in result["bins"]:
            low, high = row["gap_ci"]
            lines.append(
                f"| B{row['bin']} | {score_range(row)} | {row['n']} / {row['bin_errors']} | {row['tail_n']} | "
                f"{row['low_risk']:.2%} | {row['high_risk']:.2%} | {row['gap'] * 100:+.2f} | "
                f"[{low * 100:+.2f}, {high * 100:+.2f}] |"
            )
        largest = max(result["bins"], key=lambda row: abs(row["control_mean_high"] - row["control_mean_low"]))
        lines.extend(["", f"기준 점수 평균 차이가 가장 큰 B{largest['bin']}: "
                      f"하위 그룹 {largest['control_mean_low']:.6f}, 상위 그룹 {largest['control_mean_high']:.6f}.", ""])
    lines.extend([
        "## 계산과 재현", "",
        "- 구간 및 하위·상위 그룹은 점수만으로 정합니다. 정답 여부로 그룹을 나누지 않습니다.",
        "- 기준 점수가 동점이면 같은 구간에 둡니다. 중복 경계나 빈 구간은 합칩니다.",
        "- 그룹별 크기는 floor(구간 n × 비율)입니다. 중간 점수 그룹은 비교에서 제외합니다.",
        "- 비교 점수의 경계에 동점이 있으면 균등 무작위 선택의 기대 포함 가중치를 사용합니다. "
        "그룹의 오답 개수는 이 가중치로 계산하므로 소수가 될 수 있습니다.",
        f"- 95% 구간은 {setup['bootstrap']:,}회 stratified bootstrap의 2.5·97.5 분위수입니다(방향별 seed={setup['seed']}, {setup['seed'] + 1}). "
        "기준 구간 경계는 고정하고 구간 안의 이미지를 복원 추출합니다. 각 반복에서 비교 점수의 하위·상위 그룹을 다시 정합니다.",
        "- 하위·상위 오류율 차이는 같은 bootstrap 표본에서 계산합니다. 구간별 95% 구간에는 다중 비교 보정을 적용하지 않았습니다.",
        "- 관측 오답이 없는 그룹에서 bootstrap 구간이 0으로 나와도 실제 오류율 0을 보장하지 않습니다.",
        "- 이 결과는 하나의 체크포인트와 테스트셋에서의 탐색적 비교입니다.", "",
        "파일: `conditional_bins.csv`(구간별 결과와 기준 점수 평균), `conditional_summary.csv`(전체 요약), "
        "`conditional_groups.npz`(이미지별 구간·하위/상위 기대 포함 가중치), `conditional_results.json`(설정과 전체 결과).", "",
    ])
    (output_dir / "conditional_results.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(report, output_dir, dpi):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "observe/.mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    results, setup = report["directions"], report["setup"]
    tail_label = f"{setup['tail_fraction']:.0%}"
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey="row", layout="constrained")
        for column, result in enumerate(results):
            rows = result["bins"]
            x = np.arange(len(rows))
            top, bottom = axes[0, column], axes[1, column]
            for group, offset, color, name in (("low", -0.19, LOW_COLOR, "Low"), ("high", 0.19, HIGH_COLOR, "High")):
                risks = np.array([row[group + "_risk"] for row in rows])
                bounds = np.array([row[group + "_ci"] for row in rows])
                top.bar(x + offset, risks, width=0.36, color=color,
                        label=f"{name} {SCORE_NAMES[result['target']]} ({'bottom' if group == 'low' else 'top'} {tail_label})")
                top.vlines(x + offset, bounds[:, 0], bounds[:, 1], color="#222222", linewidth=1)
                top.hlines(bounds[:, 0], x + offset - 0.04, x + offset + 0.04, color="#222222", linewidth=1)
                top.hlines(bounds[:, 1], x + offset - 0.04, x + offset + 0.04, color="#222222", linewidth=1)
            top.set_title(f"Within {SCORE_NAMES[result['control']]} bins: compare {SCORE_NAMES[result['target']]}", fontweight="bold")
            top.set_ylabel("Error rate in selected group")
            top.yaxis.set_major_formatter(PercentFormatter(1))
            top.legend(fontsize=9, loc="upper right")
            top.set_xticks(x, [f"B{row['bin']}\nn={row['tail_n']}" for row in rows], fontsize=8)
            top.set_xlabel(f"{SCORE_NAMES[result['control']]} bins, low → high (n per selected group)")
            gap = np.array([row["gap"] * 100 for row in rows])
            bounds = np.array([row["gap_ci"] for row in rows]) * 100
            bottom.axhline(0, color="#777777", linestyle="--", linewidth=1)
            bottom.vlines(x, bounds[:, 0], bounds[:, 1], color=HIGH_COLOR, linewidth=1.5)
            bottom.hlines(bounds[:, 0], x - 0.08, x + 0.08, color=HIGH_COLOR)
            bottom.hlines(bounds[:, 1], x - 0.08, x + 0.08, color=HIGH_COLOR)
            bottom.scatter(x, gap, c=np.where(gap >= 0, HIGH_COLOR, LOW_COLOR), s=30, zorder=3)
            for position, value in zip(x, gap):
                bottom.annotate(f"{value:+.1f}", (position, value), xytext=(5, 5),
                                textcoords="offset points", fontsize=9)
            row = result["summary"]
            lower, upper = np.array(row["gap_ci"]) * 100
            bottom.set_title(f"Weighted gap: {row['gap'] * 100:+.2f} pp · 95% CI [{lower:+.2f}, {upper:+.2f}]")
            bottom.set_ylabel("Low-score risk − high-score risk (percentage points)")
            bottom.set_xticks(x, [f"B{row['bin']}" for row in rows])
            bottom.set_xlabel("Positive gap: higher score identifies fewer errors")
            for axis in (top, bottom):
                axis.grid(axis="y", alpha=0.2)
                axis.set_axisbelow(True)
        top_max = max(row[group + "_ci"][1] for result in results for row in result["bins"] for group in ("low", "high"))
        axes[0, 0].set_ylim(0, min(1.12, max(0.1, top_max * 1.18)))
        fig.suptitle(f"Conditional confidence comparison · {setup['bins']} quantile bins · Low/high {tail_label}\n"
                     f"N={setup['n']:,} · {setup['bootstrap']:,} within-bin bootstrap draws · Control bins fixed; tails reselected", fontsize=14)
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"conditional_comparison.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, layout="constrained")
        for axis, result in zip(axes, results):
            rows = result["bins"]
            x = np.arange(len(rows))
            axis.vlines(x, [row["control_min"] for row in rows], [row["control_max"] for row in rows],
                        color="#CCCCCC", linewidth=6, label="Full control-score range in bin")
            for group, offset, color in (("low", -0.07, LOW_COLOR), ("high", 0.07, HIGH_COLOR)):
                axis.scatter(x + offset, [row["control_mean_" + group] for row in rows], color=color,
                             s=35, label=f"Control mean: {group} {SCORE_NAMES[result['target']]}", zorder=3)
            axis.set_title(f"Control: {SCORE_NAMES[result['control']]} · Split: {SCORE_NAMES[result['target']]}")
            axis.set_xticks(x, [f"B{row['bin']}" for row in rows])
            axis.set_xlabel("Control-score bin (low → high)")
            axis.set_ylabel("Control score")
            axis.set_ylim(bottom=0)
            axis.legend(fontsize=9, loc="lower right")
            axis.grid(alpha=0.2)
        fig.suptitle("Check remaining differences in the control score\n"
                     "A shared quantile bin does not guarantee equal control scores in the two groups", fontsize=14)
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"control_balance.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "observe/results/noise_0.45")
    parser.add_argument("--output-dir", type=Path, help="Default: RESULTS_DIR/conditional")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.bins < 1 or not 0 < args.tail_fraction <= 0.5 or args.bootstrap < 1 or args.dpi < 1:
        parser.error("Require bins/bootstrap/dpi > 0 and 0 < tail-fraction <= 0.5.")
    with np.load(args.results_dir / "scores.npz", allow_pickle=False) as saved:
        samples = {name: saved[name] for name in saved.files}
    if not np.array_equal(samples["errors"], samples["predictions"] != samples["labels"]):
        raise ValueError("Cached error indicators disagree with the fixed predictions.")
    source = json.loads((args.results_dir / "metrics.json").read_text(encoding="utf-8"))
    report = {
        "setup": {
            "n": len(samples["errors"]), "bins": args.bins, "tail_fraction": args.tail_fraction,
            "bootstrap": args.bootstrap, "seed": args.seed,
            "source_scores": str((args.results_dir / "scores.npz").resolve()),
            "checkpoint": source["inference"]["checkpoint"],
            "checkpoint_sha256": source["inference"]["checkpoint_sha256"],
            "gap_definition": "risk(low target confidence) - risk(high target confidence)",
            "aggregate_weights": "original control-bin sample counts divided by N, identical for both groups",
            "bootstrap_method": "Within fixed control bins, resample samples with replacement and reselect tails; percentile 95% CI.",
            "ties": "Equal control scores stay in one bin; target boundary ties use expected membership.",
        },
        "directions": [],
    }
    groups = {"sample_indices": samples["sample_indices"], "errors": samples["errors"]}
    for direction_index, (control, target) in enumerate(DIRECTIONS):
        print(f"Condition on {control}; compare {target} low/high groups...", flush=True)
        result, membership = conditional_comparison(
            samples[control], samples[target], samples["errors"], bins=args.bins,
            tail_fraction=args.tail_fraction, repetitions=args.bootstrap, seed=args.seed + direction_index,
        )
        result.update(control=control, target=target, bootstrap_seed=args.seed + direction_index)
        report["directions"].append(result)
        for name, values in membership.items():
            groups[f"{target}_given_{control}_{name}"] = values
    output_dir = args.output_dir or args.results_dir / "conditional"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "conditional_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "conditional_summary.csv", [
        {"control": result["control"], "target": result["target"], **result["summary"]}
        for result in report["directions"]
    ])
    write_csv(output_dir / "conditional_bins.csv", [
        {"control": result["control"], "target": result["target"], **row}
        for result in report["directions"] for row in result["bins"]
    ])
    np.savez_compressed(output_dir / "conditional_groups.npz", **groups)
    plot_results(report, output_dir, args.dpi)
    write_report(report, output_dir)
    print("\n" + summary_table(report["directions"]))
    print(f"\nResults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
