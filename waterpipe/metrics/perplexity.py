"""Perplexity metric using vLLM logprobs."""
import math
import os
import time

from openai import OpenAI, RateLimitError

from .base import BaseMetric


class PerplexityMetric(BaseMetric):
    """Compute perplexity using reference model via vLLM."""
    
    name = "perplexity"
    required_config = ["reference_model"]
    
    def __init__(self, client: OpenAI = None):
        self.client = client
    
    def compute(self, generation: dict, config: dict,
                samples_a: list[str] = None, samples_b: list[str] = None,
                label_a: str = "watermarked", label_b: str = "non_watermarked",
                **kwargs) -> dict:
        """Compute perplexity for two texts.
        
        For attack_metrics, call with samples_a=[attacked_text], samples_b=[watermarked],
        label_a="attacked", label_b="watermarked".
        """
        if self.client is None:
            ref_config = config["reference_model"]
            self.client = OpenAI(base_url=ref_config["endpoint"], api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
        
        text_a = samples_a[0] if samples_a else generation["watermarked"]
        text_b = samples_b[0] if samples_b else generation["non_watermarked"]
        
        return {
            label_a: self._compute_single(text_a, generation["prompt"], config),
            label_b: self._compute_single(text_b, generation["prompt"], config),
        }
    
    def _compute_single(self, text: str, prompt: str, config: dict) -> float:
        """Compute perplexity of text given prompt."""
        ref_config = config["reference_model"]
        full_text = prompt + text
        
        for attempt in range(20):
            try:
                response = self.client.completions.create(
                    model=ref_config["model"],
                    prompt=full_text,
                    max_tokens=1,
                    echo=True,
                    logprobs=1,
                )
                time.sleep(0.5)  # Delay between requests to avoid WAF
                break
            except RateLimitError:
                time.sleep(5)
        else:
            raise Exception("Rate limit exceeded after 20 retries")
        
        token_logprobs = response.choices[0].logprobs.token_logprobs
        valid_logprobs = [lp for lp in token_logprobs if lp is not None]
        
        if not valid_logprobs:
            return float("inf")
        
        avg_neg_logprob = -sum(valid_logprobs) / len(valid_logprobs)
        return math.exp(avg_neg_logprob)
