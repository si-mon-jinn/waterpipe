"""Run attacks on generated texts."""
import json
from pathlib import Path

from tqdm import tqdm

from . import get_attack, get_attack_id
from ..detection import load_generations, create_detector
from ..metrics import get_metric


def aggregate_attack_requirements(config: dict) -> dict:
    """Compute max watermarked samples needed from attack_metrics."""
    reqs = {"watermarked": 1}
    attack_metrics = config.get("attack_metrics", config.get("metrics", []))
    for metric_config in attack_metrics:
        metric = get_metric(metric_config)
        m_reqs = metric.get_requirements()
        reqs["watermarked"] = max(reqs["watermarked"], m_reqs.get("watermarked", 1))
    return reqs


def load_existing_attacks(output_path: Path) -> dict[int, dict]:
    """Load existing attack records by sample_id."""
    if not output_path.exists():
        return {}
    records = {}
    with open(output_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["sample_id"]] = r
    return records


def _safe_detect(detector, text: str) -> dict:
    """Run detection, handling empty or too-short text gracefully."""
    if not text or len(text.split()) < 2:
        return {"detected": False, "z_score": 0.0, "p_value": 1.0}
    try:
        result = detector.detect(text=text)
        return {
            "detected": result.get("prediction", False),
            "z_score": result.get("z_score", 0.0),
            "p_value": result.get("p_value", 1.0),
        }
    except Exception:
        return {"detected": False, "z_score": 0.0, "p_value": 1.0}


def run_attacks(experiment_path: Path, config: dict, detector=None, batch_size: int = 8) -> None:
    """Run all configured attacks on watermarked texts.
    
    Attacks multiple watermarked_samples if required by attack_metrics.
    Supports incremental computation - only attacks samples not already processed.
    Uses batch processing for attacks that implement attack_batch().
    """
    experiment_path = Path(experiment_path)
    attacks_dir = experiment_path / "attacks"
    attacks_dir.mkdir(exist_ok=True)
    
    generations = load_generations(experiment_path)
    gen_by_id = {g["sample_id"]: g for g in generations}
    reqs = aggregate_attack_requirements(config)
    num_samples = reqs["watermarked"]
    
    if detector is None:
        detector = create_detector(config)
    
    for attack_config in config.get("attacks", []):
        attack_id = get_attack_id(attack_config)
        output_path = attacks_dir / f"{attack_id}.jsonl"
        
        existing = load_existing_attacks(output_path)
        
        # Check if complete
        if existing:
            first = next(iter(existing.values()))
            if len(existing) >= len(generations) and len(first.get("attacked_samples", [])) >= num_samples:
                print(f"Attack '{attack_id}' complete: {output_path} exists")
                continue
        
        attack = get_attack(attack_config)
        has_batch = hasattr(attack, "attack_batch")
        
        if has_batch:
            _run_attack_batched(attack, attack_id, generations, num_samples,
                                existing, detector, output_path, batch_size)
        else:
            _run_attack_sequential(attack, attack_id, generations, num_samples,
                                   existing, detector, output_path)


def _run_attack_sequential(attack, attack_id, generations, num_samples,
                           existing, detector, output_path):
    """Run attack one sample at a time."""
    with open(output_path, "w") as f:
        for gen in tqdm(generations, desc=f"Attack: {attack_id}"):
            sample_id = gen["sample_id"]
            existing_record = existing.get(sample_id, {})
            existing_attacked = existing_record.get("attacked_samples", [])
            
            wm_samples = gen.get("watermarked_samples", [gen["watermarked"]])[:num_samples]
            
            # Attack only new samples
            attacked_samples = list(existing_attacked)
            for i in range(len(attacked_samples), len(wm_samples)):
                attacked_samples.append(attack.attack(wm_samples[i]))
            
            attacked_text = attacked_samples[0] if attacked_samples else ""
            record = {
                "sample_id": sample_id,
                "attacked_text": attacked_text,
                "attacked_samples": attacked_samples,
                "detection": _safe_detect(detector, attacked_text),
            }
            f.write(json.dumps(record) + "\n")


def _run_attack_batched(attack, attack_id, generations, num_samples,
                        existing, detector, output_path, batch_size):
    """Run attack in batches for efficiency."""
    # Collect all texts that need attacking
    all_texts = []
    gen_indices = []
    for i, gen in enumerate(generations):
        wm_samples = gen.get("watermarked_samples", [gen["watermarked"]])[:num_samples]
        all_texts.append(wm_samples[0] if wm_samples else "")
        gen_indices.append(i)
    
    # Process in batches
    all_attacked = [None] * len(all_texts)
    for start in tqdm(range(0, len(all_texts), batch_size),
                      desc=f"Attack (batch): {attack_id}"):
        chunk = all_texts[start:start + batch_size]
        attacked_batch = attack.attack_batch(chunk)
        for k, attacked_text in enumerate(attacked_batch):
            all_attacked[start + k] = attacked_text
    
    # Write results with detection
    with open(output_path, "w") as f:
        for i, gen in enumerate(generations):
            attacked_text = all_attacked[i]
            record = {
                "sample_id": gen["sample_id"],
                "attacked_text": attacked_text,
                "attacked_samples": [attacked_text],
                "detection": _safe_detect(detector, attacked_text),
            }
            f.write(json.dumps(record) + "\n")
