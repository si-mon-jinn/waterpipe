"""Attack registry."""
from .base import BaseAttack
from .do_nothing import DoNothingAttack
from .random_char import RandomCharAttack
from .char_delete import CharDeleteAttack
from .char_insert import CharInsertAttack
from .word_delete import WordDeleteAttack
from .word_reorder import WordReorderAttack
from .word_substitute import WordSubstituteAttack
from .truncation import TruncationAttack
from .paraphrase import ParaphraseAttack
from .mlm_substitute import MLMSubstituteAttack

ATTACKS = {
    "do_nothing": DoNothingAttack,
    "random_char": RandomCharAttack,
    "char_delete": CharDeleteAttack,
    "char_insert": CharInsertAttack,
    "word_delete": WordDeleteAttack,
    "word_reorder": WordReorderAttack,
    "word_substitute": WordSubstituteAttack,
    "truncation": TruncationAttack,
    "paraphrase": ParaphraseAttack,
    "mlm_substitute": MLMSubstituteAttack,
}


def get_attack(attack_config: dict | str) -> BaseAttack:
    """Get attack instance from config dict or name string."""
    if isinstance(attack_config, str):
        attack_config = {"name": attack_config}
    
    name = attack_config["name"]
    if name not in ATTACKS:
        raise ValueError(f"Unknown attack: {name}")
    
    params = {k: v for k, v in attack_config.items() if k not in ("name", "id")}
    return ATTACKS[name](**params)


def get_attack_id(attack_config: dict | str) -> str:
    """Get attack output ID from config."""
    if isinstance(attack_config, str):
        return attack_config
    return attack_config.get("id", attack_config["name"])
