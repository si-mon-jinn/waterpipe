"""Detection stage: run watermark detector on generated texts."""
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from sbw import WatermarkDetector


def create_detector(config: dict) -> WatermarkDetector:
    """Create WatermarkDetector from config."""
    det_config = config["detection"]
    wm_params = config["generation"]["watermark_params"]
    
    tokenizer = AutoTokenizer.from_pretrained(det_config["tokenizer"])
    
    return WatermarkDetector(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tokenizer=tokenizer,
        vocab=[0] * len(tokenizer),
        gamma=wm_params["gamma"],
        seeding_scheme=det_config["seeding_scheme"],
        hash_key=det_config["hash_key"],
        z_threshold=det_config["threshold_z"],
    )


def load_generations(experiment_path: Path) -> list[dict]:
    """Load generations from JSONL file."""
    gen_path = Path(experiment_path) / "generations.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(f"Generations not found: {gen_path}")
    
    records = []
    with open(gen_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_detection(experiment_path: Path, config: dict, detector: WatermarkDetector = None) -> None:
    """Run detection on all generated samples."""
    output_path = Path(experiment_path) / "detection.jsonl"
    
    if output_path.exists():
        print(f"Detection complete: {output_path} exists")
        return
    
    generations = load_generations(experiment_path)
    
    if detector is None:
        detector = create_detector(config)
    
    with open(output_path, "w") as f:
        for gen in tqdm(generations, desc="Detecting"):
            wm_text = gen["watermarked"]
            no_wm_text = gen["non_watermarked"]
            
            # Skip detection for empty/too-short texts
            if wm_text.strip() and len(wm_text.split()) >= 3:
                try:
                    wm_result = detector.detect(text=wm_text)
                except (ValueError, IndexError, RuntimeError):
                    wm_result = {"prediction": False, "z_score": 0.0, "p_value": 1.0, "green_fraction": 0.0}
            else:
                wm_result = {"prediction": False, "z_score": 0.0, "p_value": 1.0, "green_fraction": 0.0}
            
            if no_wm_text.strip() and len(no_wm_text.split()) >= 3:
                try:
                    no_wm_result = detector.detect(text=no_wm_text)
                except (ValueError, IndexError, RuntimeError):
                    no_wm_result = {"prediction": False, "z_score": 0.0, "p_value": 1.0, "green_fraction": 0.0}
            else:
                no_wm_result = {"prediction": False, "z_score": 0.0, "p_value": 1.0, "green_fraction": 0.0}
            
            record = {
                "sample_id": gen["sample_id"],
                "watermarked": {
                    "detected": wm_result.get("prediction", False),
                    "z_score": wm_result.get("z_score", 0.0),
                    "p_value": wm_result.get("p_value", 1.0),
                    "green_fraction": wm_result.get("green_fraction", 0.0),
                },
                "non_watermarked": {
                    "detected": no_wm_result.get("prediction", False),
                    "z_score": no_wm_result.get("z_score", 0.0),
                    "p_value": no_wm_result.get("p_value", 1.0),
                    "green_fraction": no_wm_result.get("green_fraction", 0.0),
                },
            }
            f.write(json.dumps(record) + "\n")
