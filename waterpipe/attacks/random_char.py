"""Random character substitution attack."""
import random
import string

from .base import BaseAttack


class RandomCharAttack(BaseAttack):
    """Attack that randomly substitutes a fraction of characters."""
    
    name = "random_char"
    
    def __init__(self, fraction: float = 0.05, seed: int = None):
        self.fraction = fraction
        self.seed = seed
    
    def attack(self, text: str) -> str:
        rng = random.Random(self.seed)
        chars = list(text)
        n_changes = int(len(chars) * self.fraction)
        indices = rng.sample(range(len(chars)), min(n_changes, len(chars)))
        
        for i in indices:
            if chars[i].isalpha():
                pool = string.ascii_lowercase if chars[i].islower() else string.ascii_uppercase
                chars[i] = rng.choice(pool)
        
        return "".join(chars)
