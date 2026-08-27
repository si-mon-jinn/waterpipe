# waterpipe

[![PyPI](https://img.shields.io/pypi/v/llm-waterpipe)](https://pypi.org/project/llm-waterpipe/)
[![License](https://img.shields.io/pypi/l/llm-waterpipe)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/pypi/pyversions/llm-waterpipe)](https://pypi.org/project/llm-waterpipe/)

Evaluation pipeline for LLM watermark detection experiments. This tool orchestrates the complete experimental workflow for watermark research: text generation, watermark detection, quality metrics, and robustness testing against attacks.

## Features

- **Generation**: Produce watermarked and non-watermarked text completions via OpenAI-compatible API (e.g., vLLM server with `vllm-sbw`)
- **Detection**: Run watermark detection on generated samples
- **Metrics**: Compute text quality metrics (perplexity, BERTScore, diversity)
- **Attacks**: Test watermark robustness against various text modifications
- **Statistics**: Aggregate and analyze experimental results

## Installation

```bash
pip install llm-waterpipe
```

For all features (metrics and attacks):

```bash
pip install llm-waterpipe[all]
```

## Quick Start

### 1. Create an experiment directory

```bash
mkdir my_experiment
```

### 2. Create a configuration file

Create `my_experiment/config.json`:

```json
{
  "experiment_id": "example_selfhash_d2_g25",
  "seed": 42,
  "generation": {
    "endpoint": "http://localhost:8000/v1",
    "model": "Qwen/Qwen3-4B",
    "max_tokens": 200,
    "temperature": 1,
    "watermark_params": {
      "gamma": 0.25,
      "delta": 2,
      "seeding_scheme": "selfhash",
      "hash_key": 15485863
    }
  },
  "dataset": {
    "name": "c4",
    "split": "validation",
    "prompt_tokens": 30,
    "num_samples": 100
  },
  "detection": {
    "tokenizer": "Qwen/Qwen3-4B",
    "seeding_scheme": "selfhash",
    "hash_key": 15485863,
    "threshold_z": 4
  },
  "metrics": ["perplexity"],
  "attacks": ["random_char", "word_reorder"],
  "reference_model": {
    "endpoint": "http://localhost:8001/v1",
    "model": "Qwen/Qwen3-8B"
  }
}
```

### 3. Validate the configuration

```bash
waterpipe validate my_experiment
```

### 4. Run the pipeline

```bash
waterpipe run my_experiment --verbose
```

Or run specific stages:

```bash
waterpipe run my_experiment --stage generation
waterpipe run my_experiment --stage detection
waterpipe run my_experiment --stage metrics
waterpipe run my_experiment --stage attacks
waterpipe run my_experiment --stage stats
```

### 5. View results

Results are written to the experiment directory:

```
my_experiment/
├── config.json          # Experiment configuration
├── generations.jsonl    # Generated text samples
├── detection.jsonl      # Detection results
├── metrics/             # Quality metric outputs
├── attacks/             # Attack results
├── attack_metrics/      # Metrics on attacked text
└── stats.json           # Aggregated statistics
```

## Pipeline Stages

### Generation

Generates paired watermarked and non-watermarked completions using an OpenAI-compatible API. Requires a vLLM server running with `vllm-sbw` for watermark injection.

### Detection

Runs watermark detection on all generated samples using the `sbw` library.

### Metrics

Computes text quality metrics:
- **perplexity**: Language model perplexity (requires `reference_model` in config)
- **bertscore**: Semantic similarity to reference text
- **diversity**: Lexical diversity measures

### Attacks

Tests watermark robustness against modifications:
- `truncation`: Remove tokens from the end
- `word_delete`: Randomly delete words
- `word_reorder`: Shuffle word order locally
- `word_substitute`: Replace words with synonyms
- `char_delete`: Remove random characters
- `char_insert`: Insert random characters
- `random_char`: Replace characters randomly
- `paraphrase`: Paraphrase using an LLM
- `mlm_substitute`: Replace tokens using masked language model

### Statistics

Aggregates results into summary statistics including:
- True positive rate (TPR) for watermark detection
- False positive rate (FPR) on non-watermarked text
- Z-score distributions
- Quality metric comparisons
- Attack robustness analysis

## Batched Generation

For faster generation, use batched mode:

```bash
waterpipe run my_experiment --batched --batch-size 32
```

## Related Projects

- [sbw](https://github.com/si-mon-jinn/sbw) — Stateless Bernoulli Watermarking library
- [flip-dont-shuffle](https://github.com/si-mon-jinn/flip-dont-shuffle) — Paper repository

## License

MIT License
