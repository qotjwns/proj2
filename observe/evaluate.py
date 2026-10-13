#!/usr/bin/env python3
"""Infer clean CIFAR-10 once, or recompute selective metrics from cached logits."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from observe.metrics import COVERAGE_PERCENTAGES, evaluate_logits, markdown_table

DEFAULT_CHECKPOINT = ROOT / "experiment/cifar10/asym/alpha_dl/n0.45/seed1_epoch120_KOITBALW.pt"
DEFAULT_OUTPUT = ROOT / "observe/results/noise_0.45"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help=f"Default: {DEFAULT_CHECKPOINT}")
    source.add_argument("--cache", type=Path, help="Re-evaluate logits.pt without loading a model or dataset.")
    parser.add_argument("--config", type=Path, default=ROOT / "config/cifar10_alpha_dl.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4, help="PyTorch CPU threads (default: 4).")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.num_threads < 1:
        parser.error("batch-size/num-threads must be positive; num-workers must be nonnegative.")
    return args


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_device(requested, torch):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return torch.device(requested)


def infer(args):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from dataset import CIFAR10_MEAN, CIFAR10_STD
    from utils import get_model

    cache_path = args.output_dir / "logits.pt"
    if cache_path.exists():
        raise FileExistsError(
            f"Inference cache already exists: {cache_path}\n"
            f"Use --cache {cache_path} to reuse it, or choose a different --output-dir."
        )
    checkpoint = (args.checkpoint or DEFAULT_CHECKPOINT).resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["dataset"]["name"] != "cifar10" or config["dataset"]["num_classes"] != 10:
        raise ValueError("This evaluator requires a CIFAR-10 configuration with 10 classes.")
    if config["loss"]["name"] != "alpha_dl":
        raise ValueError("The configuration must define a Drainage model (alpha_dl).")
    drainage_idx = config["loss"]["drainage_idx"]
    # The repository's training labels directly index logits 0..9.
    if drainage_idx not in (10, -1):
        raise ValueError("This training code requires the Drainage node at index 10 (or -1).")
    drainage_idx = 10
    torch.set_num_threads(args.num_threads)
    device = choose_device(args.device, torch)
    model = get_model(config["model"]["name"], 10, "alpha_dl", "cifar10")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in weights:
        weights = weights["state_dict"]
    elif "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    if all(name.startswith("module.") for name in weights):
        weights = {name.removeprefix("module."): value for name, value in weights.items()}
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    # Build only the clean test split, never the noisy training dataset.
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    dataset = datasets.CIFAR10(
        root=str(args.data_root), train=False, transform=transform, download=args.download,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    print(f"Checkpoint: {checkpoint}\nDevice: {device}; clean test samples: {len(dataset)}", flush=True)
    all_logits, all_labels = [], []
    processed = 0
    with torch.inference_mode():
        for batch, (images, labels) in enumerate(loader, 1):
            logits = model(images.to(device, non_blocking=device.type == "cuda"))
            if logits.shape != (len(labels), 11) or not torch.isfinite(logits).all().item():
                raise ValueError("The model must return finite raw logits with shape [batch, 11].")
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            processed += len(labels)
            if batch == 1 or batch % 10 == 0 or batch == len(loader):
                print(f"Inference: {processed}/{len(dataset)}", flush=True)

    payload = {
        "logits": torch.cat(all_logits),
        "labels": torch.cat(all_labels),
        "sample_indices": torch.arange(len(dataset)),
        "metadata": {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256(checkpoint),
            "config_path": str(args.config.resolve()),
            "config": config,
            "dataset": "CIFAR-10",
            "split": "test",
            "num_samples": len(dataset),
            "data_root": str(args.data_root.resolve()),
            "drainage_idx": drainage_idx,
            "class_indices": list(range(10)),
            "normalization": {"mean": CIFAR10_MEAN, "std": CIFAR10_STD},
            "augmentation": False,
            "device": str(device),
            "torch_version": str(torch.__version__),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"Saved original logits and labels: {cache_path}", flush=True)
    return payload, cache_path.resolve()


def load_cache(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"logits", "labels", "sample_indices", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Expected a logits.pt cache created by observe/evaluate.py.")
    metadata = payload["metadata"]
    if metadata.get("schema_version") != 1 or metadata.get("dataset") != "CIFAR-10" or metadata.get("split") != "test":
        raise ValueError("Expected a version-1 clean CIFAR-10 test cache.")
    if metadata.get("augmentation") is not False or metadata.get("drainage_idx") != 10:
        raise ValueError("Unexpected augmentation or Drainage output mapping in the cache.")
    n = len(payload["labels"])
    if metadata.get("num_samples") != n or not torch.equal(payload["sample_indices"], torch.arange(n)):
        raise ValueError("Cache sample indices/count do not match the ordered test split.")
    return payload


def save_evaluation(payload, cache_path, output_dir):
    logits = payload["logits"].numpy()
    labels = payload["labels"].numpy()
    rows, curves, samples = evaluate_logits(logits, labels, payload["metadata"]["drainage_idx"])
    samples["sample_indices"] = payload["sample_indices"].numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "curves.npz", **curves)
    np.savez_compressed(output_dir / "scores.npz", **samples)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    n = len(labels)
    summary = {
        "source_cache": str(cache_path),
        "inference": payload["metadata"],
        "num_samples": n,
        "num_errors": int(samples["errors"].sum()),
        "coverage_counts": {str(p): max(1, n * p // 100) for p in COVERAGE_PERCENTAGES},
        "units": "Accuracy, risks and AURC are fractions in [0, 1].",
        "score_dtype": "float64",
        "ties": "Exact score equality; expected risk over uniform within-tie permutations.",
        "aurc_definition": "mean(R_k) for k=1,...,N; no trapezoidal integration",
        "scope": "Test-score ranking comparison; deployment thresholds require separate validation.",
        "results": rows,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    table = markdown_table(rows)
    report = (
        "# CIFAR-10 선택적 분류 평가\n\n"
        f"평가 이미지: {n:,}개 · 전체 오답: {summary['num_errors']:,}개\n\n"
        f"체크포인트: `{payload['metadata']['checkpoint']}`\n\n"
        + table
        + "\n\n정확도·오류율은 %, AURC는 0~1 비율 단위입니다.\n"
        "AURC는 k=1,…,N에서 계산한 기대 오류율의 이산 평균입니다.\n"
        "동점 내 순서는 정답 여부로 결정하지 않고, 무작위 선택의 기대 오류율을 사용합니다.\n"
        "이번 결과는 테스트 점수의 순위 비교이며, 서비스 임계값은 별도 validation에서 정해야 합니다.\n"
    )
    (output_dir / "results.md").write_text(report, encoding="utf-8")
    print("\n" + table + f"\n\nResults: {output_dir.resolve()}")


def main():
    args = parse_args()
    if args.cache:
        payload, cache_path = load_cache(args.cache), args.cache.resolve()
        print(f"Using cached logits; no inference: {cache_path}", flush=True)
    else:
        payload, cache_path = infer(args)
    save_evaluation(payload, cache_path, args.output_dir)


if __name__ == "__main__":
    main()
