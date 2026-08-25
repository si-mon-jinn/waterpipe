"""Compute aggregate statistics from experiment results."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .metrics import get_metric_id


def load_jsonl(path: Path) -> list[dict]:
    """Load records from JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_metric_stats(metric_path: Path) -> dict:
    """Compute statistics for a metric, handling both simple and distributional outputs."""
    records = load_jsonl(metric_path)
    if not records:
        return {}
    
    first = records[0]
    keys = [k for k in first.keys() if k != "sample_id"]
    
    # Check if this is a distributional metric (BERTScore-style with intra_* keys)
    if any(k.startswith("intra_") for k in keys):
        return _compute_bertscore_stats(records)
    
    # Simple metric with two float values (e.g., watermarked/non_watermarked or attacked/watermarked)
    label_a, label_b = [k for k in keys if isinstance(first[k], (int, float))][:2]
    values_a = [r[label_a] for r in records]
    values_b = [r[label_b] for r in records]
    
    _, pvalue = stats.ttest_rel(values_a, values_b)
    
    return {
        f"{label_a}_mean": float(np.mean(values_a)),
        f"{label_a}_std": float(np.std(values_a)),
        f"{label_b}_mean": float(np.mean(values_b)),
        f"{label_b}_std": float(np.std(values_b)),
        "paired_ttest_pvalue": float(pvalue),
    }


def _compute_bertscore_stats(records: list[dict]) -> dict:
    """Compute statistics for BERTScore distributional output."""
    first = records[0]
    
    # Find the two labels from intra_* keys
    intra_keys = [k for k in first.keys() if k.startswith("intra_")]
    labels = [k.replace("intra_", "") for k in intra_keys]
    
    if len(labels) < 2:
        return {}
    
    label_a, label_b = labels[0], labels[1]
    
    # Flatten all scores across samples
    intra_a = [s for r in records for s in r.get(f"intra_{label_a}", [])]
    intra_b = [s for r in records for s in r.get(f"intra_{label_b}", [])]
    ref_a = [s for r in records for s in r.get(f"ref_{label_a}", [])]
    ref_b = [s for r in records for s in r.get(f"ref_{label_b}", [])]
    
    result = {}
    
    # Intra-group diversity analysis
    if intra_a and intra_b:
        _, intra_pvalue = stats.mannwhitneyu(intra_a, intra_b, alternative="two-sided")
        result["intra_diversity"] = {
            f"{label_a}_mean": float(np.mean(intra_a)),
            f"{label_a}_std": float(np.std(intra_a)),
            f"{label_b}_mean": float(np.mean(intra_b)),
            f"{label_b}_std": float(np.std(intra_b)),
            "mannwhitney_pvalue": float(intra_pvalue),
        }
    
    # Reference fidelity analysis
    if ref_a and ref_b:
        _, ref_pvalue = stats.mannwhitneyu(ref_a, ref_b, alternative="two-sided")
        result["reference_fidelity"] = {
            f"{label_a}_mean": float(np.mean(ref_a)),
            f"{label_a}_std": float(np.std(ref_a)),
            f"{label_b}_mean": float(np.mean(ref_b)),
            f"{label_b}_std": float(np.std(ref_b)),
            "mannwhitney_pvalue": float(ref_pvalue),
        }
    
    return result


def compute_detection_stats(detection_path: Path) -> dict:
    """Compute detection statistics."""
    records = load_jsonl(detection_path)
    
    wm_detected = [r["watermarked"]["detected"] for r in records]
    no_wm_detected = [r["non_watermarked"]["detected"] for r in records]
    wm_z = [r["watermarked"]["z_score"] for r in records]
    no_wm_z = [r["non_watermarked"]["z_score"] for r in records]
    
    return {
        "watermarked_tpr": float(np.mean(wm_detected)),
        "non_watermarked_fpr": float(np.mean(no_wm_detected)),
        "mean_z_score_watermarked": float(np.mean(wm_z)),
        "mean_z_score_non_watermarked": float(np.mean(no_wm_z)),
    }


def compute_attack_stats(attack_path: Path) -> dict:
    """Compute statistics for an attack."""
    records = load_jsonl(attack_path)
    
    detected = [r["detection"]["detected"] for r in records]
    z_scores = [r["detection"]["z_score"] for r in records]
    
    return {
        "tpr": float(np.mean(detected)),
        "mean_z_score": float(np.mean(z_scores)),
    }


def compute_stats(experiment_path: Path, config: dict) -> None:
    """Compute all statistics for experiment, incrementally."""
    experiment_path = Path(experiment_path)
    stats_path = experiment_path / "stats.json"
    
    if stats_path.exists():
        with open(stats_path) as f:
            all_stats = json.load(f)
    else:
        all_stats = {"metrics": {}, "detection": {}, "attacks": {}, "attack_metrics": {}}
    
    # Ensure attack_metrics key exists for older stats files
    if "attack_metrics" not in all_stats:
        all_stats["attack_metrics"] = {}
    
    # Metrics
    metrics_dir = experiment_path / "metrics"
    for metric_config in config.get("metrics", []):
        metric_id = get_metric_id(metric_config)
        if metric_id in all_stats["metrics"]:
            continue
        metric_path = metrics_dir / f"{metric_id}.jsonl"
        if metric_path.exists():
            all_stats["metrics"][metric_id] = compute_metric_stats(metric_path)
    
    # Detection
    detection_path = experiment_path / "detection.jsonl"
    if not all_stats["detection"] and detection_path.exists():
        all_stats["detection"] = compute_detection_stats(detection_path)
    
    # Attacks
    attacks_dir = experiment_path / "attacks"
    for attack_config in config.get("attacks", []):
        attack_id = attack_config if isinstance(attack_config, str) else attack_config.get("id", attack_config["name"])
        if attack_id in all_stats["attacks"]:
            continue
        attack_path = attacks_dir / f"{attack_id}.jsonl"
        if attack_path.exists():
            all_stats["attacks"][attack_id] = compute_attack_stats(attack_path)
    
    # Attack metrics
    attack_metrics_dir = experiment_path / "attack_metrics"
    attack_metric_configs = config.get("attack_metrics", config.get("metrics", []))
    for attack_config in config.get("attacks", []):
        attack_id = attack_config if isinstance(attack_config, str) else attack_config.get("id", attack_config["name"])
        if attack_id not in all_stats["attack_metrics"]:
            all_stats["attack_metrics"][attack_id] = {}
        
        for metric_config in attack_metric_configs:
            metric_id = get_metric_id(metric_config)
            if metric_id in all_stats["attack_metrics"][attack_id]:
                continue
            metric_path = attack_metrics_dir / attack_id / f"{metric_id}.jsonl"
            if metric_path.exists():
                all_stats["attack_metrics"][attack_id][metric_id] = compute_metric_stats(metric_path)
    
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"Stats written to {stats_path}")
