"""Diversity metric based on unique n-gram fractions."""
import math
from collections import Counter

from .base import BaseMetric


def get_ngrams(tokens: list[str], n: int) -> list[tuple]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def unique_ngram_fraction(tokens: list[str], n: int) -> float:
    """Compute fraction of unique n-grams."""
    ngrams = get_ngrams(tokens, n)
    if not ngrams:
        return 1.0  # No n-grams means no repetition
    return len(set(ngrams)) / len(ngrams)


def compute_diversity(text: str, max_n: int = 4) -> float:
    """Compute diversity score: -log(1 - prod(u_n)) for n=1..N.
    
    Args:
        text: Input text
        max_n: Maximum n-gram order (default 4)
    
    Returns:
        Diversity score. Higher = more diverse.
        Returns inf if all n-gram fractions are 1.0 (perfectly diverse).
    """
    tokens = text.lower().split()
    
    if len(tokens) < 2:
        return float("inf")  # Too short to measure
    
    product = 1.0
    for n in range(1, max_n + 1):
        if len(tokens) < n:
            break
        u_n = unique_ngram_fraction(tokens, n)
        product *= u_n
    
    # -log(1 - product)
    # If product = 1 (all unique), 1 - product = 0, log undefined -> inf
    if product >= 1.0:
        return float("inf")
    
    return -math.log(1.0 - product)


class DiversityMetric(BaseMetric):
    """Compute text diversity based on unique n-gram fractions.
    
    Diversity = -log(1 - prod_{n=1}^{N} u_n)
    
    where u_n is the fraction of unique n-grams at order n.
    Higher values indicate more diverse (less repetitive) text.
    """
    
    name = "diversity"
    required_config = []
    
    def __init__(self, max_n: int = 4, **kwargs):
        """Initialize diversity metric.
        
        Args:
            max_n: Maximum n-gram order to consider (default 4)
        """
        self.max_n = max_n
    
    def compute(self, generation: dict, config: dict,
                samples_a: list[str] = None, samples_b: list[str] = None,
                label_a: str = "watermarked", label_b: str = "non_watermarked",
                **kwargs) -> dict:
        """Compute diversity for two texts."""
        text_a = samples_a[0] if samples_a else generation["watermarked"]
        text_b = samples_b[0] if samples_b else generation["non_watermarked"]
        
        return {
            label_a: compute_diversity(text_a, self.max_n),
            label_b: compute_diversity(text_b, self.max_n),
        }
