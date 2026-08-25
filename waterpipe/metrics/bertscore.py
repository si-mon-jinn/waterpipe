"""BERTScore metric for diversity and reference fidelity analysis."""
import logging
from itertools import combinations
from math import comb

import torch
from bert_score import BERTScorer

from .base import BaseMetric

logger = logging.getLogger(__name__)


class BERTScoreMetric(BaseMetric):
    """Compute BERTScore for intra-group diversity and reference fidelity."""
    
    name = "bertscore"
    required_config = []
    
    def __init__(self, num_samples_a: int = 5, num_samples_b: int = 5, 
                 model_type: str = "microsoft/deberta-xlarge-mnli", batch_size: int = None, client=None):
        self.num_samples_a = num_samples_a
        self.num_samples_b = num_samples_b
        self.model_type = model_type
        self._batch_size = batch_size
        self._scorer = None
    
    @property
    def batch_size(self) -> int:
        if self._batch_size is None:
            self._batch_size = self._estimate_batch_size()
        return self._batch_size
    
    @property
    def scorer(self) -> BERTScorer:
        if self._scorer is None:
            logger.info(f"BERTScore: loading model={self.model_type}, batch_size={self.batch_size}, samples_per_batch={self.get_batch_size()}")
            self._scorer = BERTScorer(model_type=self.model_type, batch_size=self.batch_size)
        return self._scorer
    
    def _estimate_batch_size(self) -> int:
        """Estimate batch size based on available GPU memory."""
        if not torch.cuda.is_available():
            return 32
        
        free_mem_gb = (torch.cuda.get_device_properties(0).total_memory 
                       - torch.cuda.memory_allocated(0)) / (1024**3)
        estimated = int((free_mem_gb * 0.8 * 1024) / 10)
        return max(16, min(estimated, 4096))
    
    def _pairs_per_sample(self) -> int:
        """Calculate number of bert_score pairs per sample."""
        return (comb(self.num_samples_a, 2) + comb(self.num_samples_b, 2)
                + self.num_samples_a + self.num_samples_b)
    
    def get_requirements(self) -> dict:
        return {
            "golden": 1,
            "watermarked": self.num_samples_a,
            "non_watermarked": self.num_samples_b,
        }
    
    def get_batch_size(self) -> int:
        """Return optimal number of samples to batch together."""
        return max(1, self.batch_size // self._pairs_per_sample())
    
    def compute(self, generation: dict, config: dict,
                samples_a: list[str] = None, samples_b: list[str] = None,
                label_a: str = "watermarked", label_b: str = "non_watermarked",
                **kwargs) -> dict:
        """Compute BERTScore for a single generation."""
        return self.compute_batch([generation], config, 
                                   samples_a=[samples_a] if samples_a else None,
                                   samples_b=[samples_b] if samples_b else None,
                                   label_a=label_a, label_b=label_b)[0]
    
    def compute_batch(self, generations: list[dict], config: dict,
                      samples_a: list[list[str]] = None, samples_b: list[list[str]] = None,
                      label_a: str = "watermarked", label_b: str = "non_watermarked",
                      **kwargs) -> list[dict]:
        """Compute BERTScore for a batch of generations in single scorer.score() call."""
        # Build combined lists for single call
        all_cands = []
        all_refs = []
        sample_slices = []
        
        for i, gen in enumerate(generations):
            sa = samples_a[i] if samples_a and samples_a[i] else gen["watermarked_samples"][:self.num_samples_a]
            sb = samples_b[i] if samples_b and samples_b[i] else gen["non_watermarked_samples"][:self.num_samples_b]
            golden = gen.get("golden", "")
            
            slices = {}
            
            # intra_a pairs
            pairs_a = list(combinations(range(len(sa)), 2))
            start = len(all_cands)
            all_cands.extend([sa[j] for j, k in pairs_a])
            all_refs.extend([sa[k] for j, k in pairs_a])
            slices[f"intra_{label_a}"] = (start, len(all_cands))
            
            # intra_b pairs
            pairs_b = list(combinations(range(len(sb)), 2))
            start = len(all_cands)
            all_cands.extend([sb[j] for j, k in pairs_b])
            all_refs.extend([sb[k] for j, k in pairs_b])
            slices[f"intra_{label_b}"] = (start, len(all_cands))
            
            # ref_a and ref_b
            if golden:
                start = len(all_cands)
                all_cands.extend(sa)
                all_refs.extend([golden] * len(sa))
                slices[f"ref_{label_a}"] = (start, len(all_cands))
                
                start = len(all_cands)
                all_cands.extend(sb)
                all_refs.extend([golden] * len(sb))
                slices[f"ref_{label_b}"] = (start, len(all_cands))
            
            sample_slices.append((slices, bool(golden)))
        
        # Single scorer.score() call - model stays loaded
        f1 = None
        if all_cands:
            _, _, f1 = self.scorer.score(all_cands, all_refs)
        
        # Split results back per sample
        results = []
        for slices, has_golden in sample_slices:
            result = {}
            for key, (start, end) in slices.items():
                result[key] = f1[start:end].tolist() if f1 is not None else []
            if not has_golden:
                result[f"ref_{label_a}"] = []
                result[f"ref_{label_b}"] = []
            results.append(result)
        
        return results
