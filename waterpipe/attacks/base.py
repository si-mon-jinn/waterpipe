"""Base class for attacks."""
from abc import ABC, abstractmethod


class BaseAttack(ABC):
    """Abstract base class for watermark attacks."""
    
    name: str = "base"
    
    def get_requirements(self) -> dict:
        """Return generation requirements: {golden, watermarked, non_watermarked}."""
        return {"golden": 0, "watermarked": 1, "non_watermarked": 1}
    
    @abstractmethod
    def attack(self, text: str) -> str:
        """Transform text to attempt to remove watermark."""
        pass
