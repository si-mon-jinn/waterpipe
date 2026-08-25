"""Base class for metrics."""
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Abstract base class for quality metrics."""
    
    name: str = "base"
    required_config: list[str] = []
    
    def get_requirements(self) -> dict:
        """Return generation requirements: {golden, watermarked, non_watermarked}."""
        return {"golden": 0, "watermarked": 1, "non_watermarked": 1}
    
    def get_batch_size(self) -> int:
        """Return optimal number of samples to batch. 1 = no batching."""
        return 1
    
    @abstractmethod
    def compute(self, generation: dict, config: dict,
                samples_a: list[str] = None, samples_b: list[str] = None,
                label_a: str = "watermarked", label_b: str = "non_watermarked",
                **kwargs) -> dict:
        """Compute metric for a single generation record."""
        pass
    
    def compute_batch(self, generations: list[dict], config: dict,
                      samples_a: list[list[str]] = None, samples_b: list[list[str]] = None,
                      **kwargs) -> list[dict]:
        """Compute metric for a batch of generations. Default: sequential."""
        results = []
        for i, gen in enumerate(generations):
            sa = samples_a[i] if samples_a else None
            sb = samples_b[i] if samples_b else None
            results.append(self.compute(gen, config, samples_a=sa, samples_b=sb, **kwargs))
        return results
