"""T5 span-filling substitution attack — iterative single-word masking with T5."""
import random
from collections import Counter

from .base import BaseAttack


def _count_word_diff(orig_text, new_text):
    """Count word difference using multiset (bag-of-words) comparison."""
    orig_counts = Counter(orig_text.lower().split())
    new_counts = Counter(new_text.lower().split())
    removed = sum((orig_counts - new_counts).values())
    added = sum((new_counts - orig_counts).values())
    return max(removed, added)


class MLMSubstituteAttack(BaseAttack):
    """Attack that iteratively masks and replaces words using T5 span-filling."""

    name = "mlm_substitute"

    def __init__(self, model: str = "t5-large", fraction: float = 0.15,
                 seed: int = None, temperature: float = 0.7, **kwargs):
        self.model_name = model
        self.fraction = fraction
        self.seed = seed
        self.temperature = temperature
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._tokenizer is None:
            import torch
            from transformers import T5ForConditionalGeneration, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            )
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()

    @staticmethod
    def _extract_replacement(raw):
        """Extract the replacement word from T5 output."""
        tag = "<extra_id_0>"
        end_tag = "<extra_id_1>"
        if tag in raw:
            s = raw.index(tag) + len(tag)
            e = raw.index(end_tag) if end_tag in raw else (
                raw.index("</s>") if "</s>" in raw else len(raw))
            replacement = raw[s:e].strip()
            return replacement if replacement else None
        return None

    def _fill_batch(self, batch_inputs):
        """Fill masked positions for a batch of texts using T5."""
        import torch

        inputs = self._tokenizer(batch_inputs, return_tensors="pt",
                                 padding=True, truncation=True)
        if next(self._model.parameters()).is_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=5,
                do_sample=True, temperature=self.temperature,
            )

        results = []
        for out in outputs:
            raw = self._tokenizer.decode(out, skip_special_tokens=False)
            results.append(self._extract_replacement(raw))
        return results

    def attack(self, text: str) -> str:
        import torch

        self._load_model()
        rng = random.Random(self.seed)
        words = text.split()
        target = max(1, int(len(words) * self.fraction))

        candidates = list(range(len(words)))
        rng.shuffle(candidates)

        current_words = list(words)
        original_text = text

        for pos in candidates:
            if _count_word_diff(original_text, " ".join(current_words)) >= target:
                break
            masked = list(current_words)
            masked[pos] = "<extra_id_0>"
            input_text = " ".join(masked)
            replacements = self._fill_batch([input_text])
            if replacements[0]:
                current_words[pos] = replacements[0]

        return " ".join(current_words)

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Batch attack — processes all texts in parallel iterations."""
        self._load_model()

        # Initialize state for each text
        states = []
        for text in texts:
            rng = random.Random(self.seed)
            words = text.split()
            target = max(1, int(len(words) * self.fraction))
            candidates = list(range(len(words)))
            rng.shuffle(candidates)
            states.append({
                "original": text,
                "current_words": list(words),
                "candidates": candidates,
                "target": target,
                "candidate_idx": 0,
                "done": False,
            })

        # Iterate until all texts are done
        max_iters = max(len(s["candidates"]) for s in states)
        for _ in range(max_iters):
            # Collect texts that still need processing
            batch_inputs = []
            batch_indices = []
            batch_positions = []

            for i, s in enumerate(states):
                if s["done"] or s["candidate_idx"] >= len(s["candidates"]):
                    s["done"] = True
                    continue
                if _count_word_diff(s["original"], " ".join(s["current_words"])) >= s["target"]:
                    s["done"] = True
                    continue

                pos = s["candidates"][s["candidate_idx"]]
                s["candidate_idx"] += 1
                masked = list(s["current_words"])
                masked[pos] = "<extra_id_0>"
                batch_inputs.append(" ".join(masked))
                batch_indices.append(i)
                batch_positions.append(pos)

            if not batch_inputs:
                break

            # Run T5 on the batch
            replacements = self._fill_batch(batch_inputs)

            # Apply replacements
            for idx, pos, repl in zip(batch_indices, batch_positions, replacements):
                if repl:
                    states[idx]["current_words"][pos] = repl

        return [" ".join(s["current_words"]) for s in states]
