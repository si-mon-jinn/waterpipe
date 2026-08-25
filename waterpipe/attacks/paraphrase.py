"""LLM paraphrase attack — rewrites text using an API or local model."""
import os

from .base import BaseAttack

_PROMPTS = {
    "light": (
        "Rephrase the following text with minimal changes — replace a few words "
        "with synonyms but keep the sentence structure intact. "
        "Keep approximately the same length. "
        "Output only the rewritten text, nothing else.\n\nText: {text}"
    ),
    "moderate": (
        "Rewrite the following text to convey the same meaning using different "
        "words and sentence structure. Do not add or remove information. "
        "Keep approximately the same length. "
        "Output only the rewritten text, nothing else.\n\nText: {text}"
    ),
    "heavy": (
        "Completely rewrite the following text in your own words. Preserve the "
        "meaning but feel free to restructure sentences, change vocabulary, and "
        "alter phrasing significantly. Keep approximately the same length. "
        "Output only the rewritten text, nothing else."
        "\n\nText: {text}"
    ),
}


class ParaphraseAttack(BaseAttack):
    """Attack that paraphrases text using an LLM (API or local)."""

    name = "paraphrase"

    def __init__(self, endpoint: str = None, model: str = None,
                 level: str = "moderate", temperature: float = 0.7, **kwargs):
        self.endpoint = endpoint
        self.model = model or os.environ.get("PARAPHRASE_MODEL", "Qwen/Qwen3-0.6B")
        self.level = level
        self.temperature = temperature
        self._local_pipeline = None
        self._client = None

        if self.endpoint:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.endpoint, api_key="none")

    def _get_local_pipeline(self):
        if self._local_pipeline is None:
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model, padding_side="left")
            model = AutoModelForCausalLM.from_pretrained(
                self.model, device_map="auto", torch_dtype="auto",
            )
            model.resize_token_embeddings(len(tokenizer))
            self._local_pipeline = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
            )
        return self._local_pipeline

    def attack(self, text: str) -> str:
        prompt = _PROMPTS[self.level].format(text=text)

        if self._client:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=int(len(text.split()) * 1.5),
            )
            return resp.choices[0].message.content.strip()

        pipe = self._get_local_pipeline()
        messages = [{"role": "user", "content": prompt}]
        out = pipe(messages, max_new_tokens=int(len(text.split()) * 1.5),
                   temperature=self.temperature, do_sample=True)
        return out[0]["generated_text"][-1]["content"].strip()

    def attack_batch(self, texts: list[str]) -> list[str]:
        """Batch attack for efficiency with local models."""
        if self._client:
            return [self.attack(t) for t in texts]

        pipe = self._get_local_pipeline()
        batch = [[{"role": "user", "content": _PROMPTS[self.level].format(text=t)}] for t in texts]
        max_tokens = int(max(len(t.split()) * 1.5 for t in texts))
        outputs = pipe(batch, max_new_tokens=max_tokens,
                       temperature=self.temperature, do_sample=True,
                       batch_size=len(batch))
        return [o[0]["generated_text"][-1]["content"].strip() for o in outputs]
