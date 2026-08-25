"""Run metrics on generated texts."""
import json
from pathlib import Path

from tqdm import tqdm

from . import get_metric
from ..detection import load_generations


def _count_lines(path: Path) -> int:
    """Count lines in a file."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def run_metrics(experiment_path: Path, config: dict, client=None) -> None:
    """Run all configured metrics on generated texts."""
    experiment_path = Path(experiment_path)
    metrics_dir = experiment_path / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    generations = load_generations(experiment_path)
    
    for metric_config in config.get("metrics", []):
        metric_name, metric_id = _parse_metric_config(metric_config)
        output_path = metrics_dir / f"{metric_id}.jsonl"
        
        existing = _count_lines(output_path)
        if existing >= len(generations):
            print(f"Metric '{metric_id}' complete: {output_path} exists")
            continue
        
        metric = get_metric(metric_config, client=client)
        batch_size = metric.get_batch_size()
        
        # Resume from where we left off
        remaining = generations[existing:]
        mode = "a" if existing > 0 else "w"
        
        with open(output_path, mode) as f:
            for i in tqdm(range(0, len(remaining), batch_size), desc=f"Metric: {metric_id}", initial=existing, total=len(generations)):
                chunk = remaining[i:i+batch_size]
                results = metric.compute_batch(chunk, config)
                
                for gen, result in zip(chunk, results):
                    record = {"sample_id": gen["sample_id"], **result}
                    f.write(json.dumps(record) + "\n")
                f.flush()


def _parse_metric_config(metric_config) -> tuple[str, str]:
    """Return (metric_name, metric_id) from config."""
    if isinstance(metric_config, str):
        return metric_config, metric_config
    return metric_config["name"], metric_config.get("id", metric_config["name"])
