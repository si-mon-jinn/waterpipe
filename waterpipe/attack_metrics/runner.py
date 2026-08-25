"""Run metrics on attacked texts."""
import json
from pathlib import Path

from tqdm import tqdm

from ..metrics import get_metric, get_metric_id
from ..attacks import get_attack_id
from ..detection import load_generations


def load_jsonl(path: Path) -> list[dict]:
    """Load records from JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_attack_metrics(experiment_path: Path, config: dict, client=None) -> None:
    """Run metrics on attacked texts vs original watermarked."""
    experiment_path = Path(experiment_path)
    attack_metrics_dir = experiment_path / "attack_metrics"
    attack_metrics_dir.mkdir(exist_ok=True)
    
    generations = load_generations(experiment_path)
    gen_by_id = {g["sample_id"]: g for g in generations}
    
    metric_configs = config.get("attack_metrics", config.get("metrics", []))
    
    for attack_config in config.get("attacks", []):
        attack_id = get_attack_id(attack_config)
        attack_path = experiment_path / "attacks" / f"{attack_id}.jsonl"
        
        if not attack_path.exists():
            print(f"Attack '{attack_id}' not found, skipping metrics")
            continue
        
        attack_records = load_jsonl(attack_path)
        attack_by_id = {r["sample_id"]: r for r in attack_records}
        
        attack_dir = attack_metrics_dir / attack_id
        attack_dir.mkdir(exist_ok=True)
        
        for metric_config in metric_configs:
            metric_id = get_metric_id(metric_config)
            output_path = attack_dir / f"{metric_id}.jsonl"
            
            if output_path.exists():
                print(f"Attack metric '{attack_id}/{metric_id}' complete")
                continue
            
            metric = get_metric(metric_config, client=client)
            batch_size = metric.get_batch_size()
            
            # Prepare data for batching
            paired_data = []
            for gen in generations:
                sample_id = gen["sample_id"]
                attack_record = attack_by_id.get(sample_id, {})
                attacked_samples = attack_record.get("attacked_samples", [attack_record.get("attacked_text", "")])
                wm_samples = gen.get("watermarked_samples", [gen["watermarked"]])
                paired_data.append((gen, attacked_samples, wm_samples))
            
            with open(output_path, "w") as f:
                for i in tqdm(range(0, len(paired_data), batch_size), desc=f"{attack_id}/{metric_id}"):
                    chunk = paired_data[i:i+batch_size]
                    gens = [p[0] for p in chunk]
                    samples_a = [p[1] for p in chunk]
                    samples_b = [p[2] for p in chunk]
                    
                    results = metric.compute_batch(
                        gens, config,
                        samples_a=samples_a,
                        samples_b=samples_b,
                        label_a="attacked", label_b="watermarked"
                    )
                    
                    for gen, result in zip(gens, results):
                        record = {"sample_id": gen["sample_id"], **result}
                        f.write(json.dumps(record) + "\n")
