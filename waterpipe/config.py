"""Config loading and validation for experiments."""
import json
from pathlib import Path

from .metrics import METRICS

REQUIRED_FIELDS = ["experiment_id", "seed", "generation", "dataset", "detection", "metrics", "attacks"]
METRIC_REQUIRED_CONFIG = {
    "perplexity": ["reference_model"],
}


def load_config(experiment_path: Path) -> dict:
    """Load config.json from experiment directory."""
    config_path = Path(experiment_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def _validate_metric_config(metric_config, config: dict, field_name: str) -> None:
    """Validate a single metric config entry."""
    metric_name = metric_config if isinstance(metric_config, str) else metric_config["name"]
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric in {field_name}: {metric_name}")
    for required in METRIC_REQUIRED_CONFIG.get(metric_name, []):
        if required not in config:
            raise ValueError(f"Metric '{metric_name}' requires config section: {required}")


def validate_config(config: dict) -> None:
    """Validate experiment config. Raises ValueError if invalid."""
    for field in REQUIRED_FIELDS:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    for metric_config in config.get("metrics", []):
        _validate_metric_config(metric_config, config, "metrics")

    # Validate attack_metrics if present
    for metric_config in config.get("attack_metrics", []):
        _validate_metric_config(metric_config, config, "attack_metrics")
