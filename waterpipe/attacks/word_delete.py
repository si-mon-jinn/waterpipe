"""Word deletion attack — randomly removes a fraction of words."""
import random

from .base import BaseAttack


class WordDeleteAttack(BaseAttack):
    """Attack that randomly deletes a fraction of words."""

    name = "word_delete"

    def __init__(self, fraction: float = 0.2, seed: int = None):
        self.fraction = fraction
        self.seed = seed

    def attack(self, text: str) -> str:
        rng = random.Random(self.seed)
        words = text.split()
        n_delete = max(1, int(len(words) * self.fraction))
        indices = set(rng.sample(range(len(words)), min(n_delete, len(words))))
        return " ".join(w for i, w in enumerate(words) if i not in indices)
