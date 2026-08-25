"""Random character deletion attack."""
import random

from .base import BaseAttack


class CharDeleteAttack(BaseAttack):
    """Attack that randomly deletes a fraction of characters."""

    name = "char_delete"

    def __init__(self, fraction: float = 0.05, seed: int = None):
        self.fraction = fraction
        self.seed = seed

    def attack(self, text: str) -> str:
        rng = random.Random(self.seed)
        chars = list(text)
        n_deletes = int(len(chars) * self.fraction)
        indices = set(rng.sample(range(len(chars)), min(n_deletes, len(chars))))
        return "".join(c for i, c in enumerate(chars) if i not in indices)
