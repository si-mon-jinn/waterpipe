"""Truncation attack — keeps only a fraction of the text."""

from .base import BaseAttack


class TruncationAttack(BaseAttack):
    """Attack that keeps only the first fraction of words."""

    name = "truncation"

    def __init__(self, fraction: float = 0.5, seed: int = None):
        self.fraction = fraction
        self.seed = seed  # unused, but kept for interface consistency

    def attack(self, text: str) -> str:
        words = text.split()
        keep = max(1, int(len(words) * self.fraction))
        return " ".join(words[:keep])
