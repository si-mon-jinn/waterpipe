"""Random character insertion attack."""
import random
import string

from .base import BaseAttack


class CharInsertAttack(BaseAttack):
    """Attack that randomly inserts characters at a fraction of positions."""

    name = "char_insert"

    def __init__(self, fraction: float = 0.05, seed: int = None):
        self.fraction = fraction
        self.seed = seed

    def attack(self, text: str) -> str:
        rng = random.Random(self.seed)
        chars = list(text)
        n_inserts = int(len(chars) * self.fraction)
        positions = sorted(rng.sample(range(len(chars)), min(n_inserts, len(chars))), reverse=True)
        for i in positions:
            ch = rng.choice(string.ascii_lowercase)
            chars.insert(i, ch)
        return "".join(chars)
