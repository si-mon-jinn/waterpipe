"""Metric registry."""
from .base import BaseMetric
from .perplexity import PerplexityMetric
from .diversity import DiversityMetric


def _get_bertscore():
    from .bertscore import BERTScoreMetric
    return BERTScoreMetric


METRICS = {"perplexity": PerplexityMetric, "bertscore": _get_bertscore, "diversity": DiversityMetric}


def get_metric(metric_config: dict | str, **kwargs) -> BaseMetric:
    """Get metric instance from config dict or name string."""
    if isinstance(metric_config, str):
        metric_config = {"name": metric_config}
    
    name = metric_config["name"]
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}")
    
    metric_cls = METRICS[name]
    if callable(metric_cls) and not isinstance(metric_cls, type):
        metric_cls = metric_cls()  # Call lazy loader
    
    params = {k: v for k, v in metric_config.items() if k not in ("name", "id")}
    return metric_cls(**{**params, **kwargs})


def get_metric_id(metric_config: dict | str) -> str:
    """Get metric output ID from config."""
    if isinstance(metric_config, str):
        return metric_config
    return metric_config.get("id", metric_config["name"])
