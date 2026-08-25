"""Word reorder attack — shuffles words within sentences."""
import random

from .base import BaseAttack


class WordReorderAttack(BaseAttack):
    """Attack that randomly reorders words within each sentence."""

    name = "word_reorder"

    def __init__(self, fraction: float = 1.0, seed: int = None):
        self.fraction = fraction
        self.seed = seed

    def attack(self, text: str) -> str:
        rng = random.Random(self.seed)
        sentences = text.split(". ")
        result = []
        for sent in sentences:
            words = sent.split()
            if len(words) > 1:
                n_swap = max(1, int(len(words) * self.fraction))
                indices = list(range(len(words)))
                for _ in range(n_swap):
                    i, j = rng.sample(indices, 2)
                    words[i], words[j] = words[j], words[i]
            result.append(" ".join(words))
        return ". ".join(result)
