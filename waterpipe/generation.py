"""Generation stage: generate watermarked and non-watermarked text."""
import json
from pathlib import Path

from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from .metrics import get_metric, METRICS
from .attacks import get_attack, ATTACKS


def load_prompts(config: dict) -> list[dict]:
    """Load prompts and golden continuations from dataset."""
    ds_config = config["dataset"]
    gen_config = config["generation"]
    tokenizer = AutoTokenizer.from_pretrained(gen_config["model"])
    
    ds_name = ds_config["name"]
    if ds_name == "c4":
        ds_name = "allenai/c4"
    if ds_name == "alpaca":
        ds_name = "tatsu-lab/alpaca"
    ds_subset = ds_config.get("subset", None)
    if ds_name == "allenai/c4" and ds_subset is None:
        ds_subset = "en"
    if ds_subset:
        dataset = load_dataset(ds_name, ds_subset, split=ds_config["split"], streaming=True)
    else:
        dataset = load_dataset(ds_name, split=ds_config["split"], streaming=True)
    
    # For alpaca: use instruction format, prompt up to "### Response:\n"
    is_alpaca = "alpaca" in ds_name
    
    results = []
    for item in dataset:
        if len(results) >= ds_config["num_samples"]:
            break
        text = item.get("text", "")
        
        if is_alpaca:
            marker = "### Response:\n"
            marker_pos = text.find(marker)
            if marker_pos == -1:
                continue
            prompt = text[:marker_pos + len(marker)]
            golden = text[marker_pos + len(marker):]
            if prompt.strip():
                results.append({"prompt": prompt, "golden": golden})
        else:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            prompt_tokens = tokens[:ds_config["prompt_tokens"]]
            golden_tokens = tokens[ds_config["prompt_tokens"]:ds_config["prompt_tokens"] + gen_config["max_tokens"]]
            
            prompt = tokenizer.decode(prompt_tokens)
            golden = tokenizer.decode(golden_tokens) if golden_tokens else ""
            
            if prompt.strip() and len(prompt_tokens) >= ds_config["prompt_tokens"]:
                results.append({"prompt": prompt, "golden": golden})
    
    return results[:ds_config["num_samples"]]


def aggregate_requirements(config: dict) -> dict:
    """Compute max generation requirements from all metrics, attacks, and attack_metrics."""
    reqs = {"golden": 0, "watermarked": 1, "non_watermarked": 1}
    
    for metric_config in config.get("metrics", []):
        metric = get_metric(metric_config)
        m_reqs = metric.get_requirements()
        for k in reqs:
            reqs[k] = max(reqs[k], m_reqs.get(k, 0))
    
    for attack_config in config.get("attacks", []):
        attack = get_attack(attack_config)
        a_reqs = attack.get_requirements()
        for k in reqs:
            reqs[k] = max(reqs[k], a_reqs.get(k, 0))
    
    # attack_metrics also need watermarked samples (for attacking)
    attack_metrics = config.get("attack_metrics", config.get("metrics", []))
    for metric_config in attack_metrics:
        metric = get_metric(metric_config)
        m_reqs = metric.get_requirements()
        reqs["watermarked"] = max(reqs["watermarked"], m_reqs.get("watermarked", 1))
    
    return reqs


def load_existing_generations(output_path: Path) -> list[dict]:
    """Load existing generations from JSONL file."""
    if not output_path.exists():
        return []
    records = []
    with open(output_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_generation(experiment_path: Path, config: dict, client: OpenAI = None, batched: bool = False, batch_size: int = None) -> None:
    """Generate watermarked and non-watermarked completions."""
    output_path = Path(experiment_path) / "generations.jsonl"
    reqs = aggregate_requirements(config)
    
    existing = load_existing_generations(output_path)
    if len(existing) >= config["dataset"]["num_samples"]:
        if existing and len(existing[0].get("watermarked_samples", [])) >= reqs["watermarked"]:
            print(f"Generation complete: {len(existing)} samples with sufficient completions")
            return
    
    prompts_data = load_prompts(config)
    
    if client is None:
        client = OpenAI(base_url=config["generation"]["endpoint"], api_key="dummy")
    
    gen_config = config["generation"]
    wm_params = gen_config["watermark_params"]
    
    if batched:
        _run_generation_batched(output_path, prompts_data, config, client, gen_config, wm_params, reqs, batch_size)
    else:
        _run_generation_sequential(output_path, prompts_data, config, client, gen_config, wm_params, reqs)


def _run_generation_batched(output_path, prompts_data, config, client, gen_config, wm_params, reqs, batch_size=None):
    """Generate all completions in batched requests."""
    prompts = [p["prompt"] for p in prompts_data]
    n = len(prompts)
    batch_size = batch_size or n
    
    wm_texts = []
    no_wm_texts = []
    no_wm_params = {**wm_params, "delta": 0.0}
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_prompts = prompts[start:end]
        print(f"Generating watermarked {start+1}-{end}/{n}...")
        wm_response = client.completions.create(
            model=gen_config["model"],
            prompt=batch_prompts,
            max_tokens=gen_config["max_tokens"],
            temperature=gen_config["temperature"],
            seed=config["seed"],
            extra_body={"vllm_xargs": wm_params}
        )
        wm_texts.extend([c.text for c in wm_response.choices])
        
        print(f"Generating non-watermarked {start+1}-{end}/{n}...")
        no_wm_response = client.completions.create(
            model=gen_config["model"],
            prompt=batch_prompts,
            max_tokens=gen_config["max_tokens"],
            temperature=gen_config["temperature"],
            seed=config["seed"],
            extra_body={"vllm_xargs": no_wm_params}
        )
        no_wm_texts.extend([c.text for c in no_wm_response.choices])
    
    # Write results
    with open(output_path, "w") as f:
        for i, prompt_data in enumerate(prompts_data):
            record = {
                "sample_id": i,
                "prompt": prompt_data["prompt"],
                "watermarked": wm_texts[i],
                "non_watermarked": no_wm_texts[i],
                "seed": config["seed"] + i * 1000,
                "watermarked_samples": [wm_texts[i]],
                "non_watermarked_samples": [no_wm_texts[i]],
            }
            if reqs["golden"] > 0:
                record["golden"] = prompt_data["golden"]
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {n} samples to {output_path}")


def _run_generation_sequential(output_path, prompts_data, config, client, gen_config, wm_params, reqs):
    """Generate completions sequentially (original behavior)."""
    existing_by_id = {}
    
    with open(output_path, "w") as f:
        for i, prompt_data in enumerate(tqdm(prompts_data, desc="Generating")):
            prompt = prompt_data["prompt"]
            golden = prompt_data["golden"] if reqs["golden"] > 0 else ""
            
            existing_record = existing_by_id.get(i, {})
            existing_wm = existing_record.get("watermarked_samples", [])
            existing_no_wm = existing_record.get("non_watermarked_samples", [])
            
            wm_samples = list(existing_wm)
            for j in range(len(wm_samples), reqs["watermarked"]):
                sample_seed = config["seed"] + i * 1000 + j
                response = client.completions.create(
                    model=gen_config["model"],
                    prompt=prompt,
                    max_tokens=gen_config["max_tokens"],
                    temperature=gen_config["temperature"],
                    seed=sample_seed,
                    extra_body={"vllm_xargs": wm_params}
                )
                wm_samples.append(response.choices[0].text)
            
            no_wm_samples = list(existing_no_wm)
            no_wm_params = {**wm_params, "delta": 0.0}
            for j in range(len(no_wm_samples), reqs["non_watermarked"]):
                sample_seed = config["seed"] + i * 1000 + j
                response = client.completions.create(
                    model=gen_config["model"],
                    prompt=prompt,
                    max_tokens=gen_config["max_tokens"],
                    temperature=gen_config["temperature"],
                    seed=sample_seed,
                    extra_body={"vllm_xargs": no_wm_params}
                )
                no_wm_samples.append(response.choices[0].text)
            
            record = {
                "sample_id": i,
                "prompt": prompt,
                "watermarked": wm_samples[0] if wm_samples else "",
                "non_watermarked": no_wm_samples[0] if no_wm_samples else "",
                "seed": config["seed"] + i * 1000,
                "watermarked_samples": wm_samples,
                "non_watermarked_samples": no_wm_samples,
            }
            if reqs["golden"] > 0:
                record["golden"] = golden
            
            f.write(json.dumps(record) + "\n")
            f.flush()
