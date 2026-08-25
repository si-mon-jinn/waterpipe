"""Word substitution attack — replaces words with random same-length words."""
import random
import string

from .base import BaseAttack


class WordSubstituteAttack(BaseAttack):
    """Attack that substitutes a fraction of words with random alphabetic strings."""

    name = "word_substitute"

    def __init__(self, fraction: float = 0.2, seed: int = None):
        self.fraction = fraction
        self.seed = seed

    def attack(self, text: str) -> str:
        rng = random.Random(self.seed)
        words = text.split()
        n_sub = max(1, int(len(words) * self.fraction))
        indices = rng.sample(range(len(words)), min(n_sub, len(words)))
        for i in indices:
            length = len(words[i])
            replacement = "".join(rng.choices(string.ascii_lowercase, k=length))
            words[i] = replacement
        return " ".join(words)
