"""Do-nothing attack (baseline)."""
from .base import BaseAttack


class DoNothingAttack(BaseAttack):
    """Baseline attack that returns text unchanged."""
    
    name = "do_nothing"
    
    def attack(self, text: str) -> str:
        return text
