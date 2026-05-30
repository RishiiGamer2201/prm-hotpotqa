"""
Process Reward Model (PRM) for multi-hop QA.

Scores (query, paragraph) pairs using a cross-encoder and prunes paragraphs
below a relevance threshold. Uses cross-encoder/ms-marco-MiniLM-L-6-v2
zero-shot - no training required.
"""

import random
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple, Optional

# Global seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def _sigmoid(x: float) -> float:
    """Apply sigmoid to convert unbounded logit to (0, 1) probability."""
    return 1.0 / (1.0 + np.exp(-x))


class ProcessRewardModel:
    """
    Process Reward Model using a cross-encoder for passage relevance scoring.

    Scores each (query, paragraph) pair independently and prunes paragraphs
    that fall below a configurable threshold. Raw cross-encoder logits are
    normalized via sigmoid so that threshold values (e.g., 0.4 and 0.6) have
    consistent meaning across queries.

    Attributes:
        model: CrossEncoder instance (ms-marco-MiniLM-L-6-v2).
        device: torch device (cuda or cpu).
        default_threshold: Default pruning threshold.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        default_threshold: float = 0.5,
    ):
        """
        Initialize the PRM with a pretrained cross-encoder.

        Args:
            model_name: HuggingFace model identifier for the cross-encoder.
            device: 'cuda' or 'cpu'. Auto-detected if None.
            default_threshold: Default threshold for prune_steps.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.default_threshold = default_threshold

        # Load cross-encoder - used zero-shot, no training
        self.model = CrossEncoder(model_name, device=device)
        print(f"[PRM] Loaded {model_name} on {device}")

    def score_step(self, query: str, paragraph: str) -> float:
        """
        Score a single (query, paragraph) pair.

        Args:
            query: The question or sub-query string.
            paragraph: The paragraph text to score.

        Returns:
            Sigmoid-normalized relevance score in (0, 1).
        """
        raw_score = self.model.predict([(query, paragraph)])[0]
        return float(_sigmoid(raw_score))

    def score_batch(
        self, query: str, paragraphs: List[str], batch_size: int = 64
    ) -> List[float]:
        """
        Score a list of paragraphs against the same query.

        Args:
            query: The question or sub-query string.
            paragraphs: List of paragraph texts.
            batch_size: Batch size for cross-encoder inference.

        Returns:
            List of sigmoid-normalized scores, same order as input.
        """
        pairs = [(query, p) for p in paragraphs]
        raw_scores = self.model.predict(pairs, batch_size=batch_size)
        return [float(_sigmoid(s)) for s in raw_scores]

    def prune_steps(
        self,
        query: str,
        paragraphs: List[Dict],
        threshold: Optional[float] = None,
        text_key: str = "text",
        min_keep: int = 1,
    ) -> Tuple[List[Dict], List[float]]:
        """
        Score every paragraph and filter out those below the threshold.

        Each kept paragraph dict gets additional keys:
            - prm_score: float, the sigmoid-normalized score
            - prm_threshold: float, the threshold used
            - prm_pass: bool, whether the paragraph passed

        If all paragraphs fall below the threshold, the top `min_keep`
        paragraphs by score are retained (safety floor to avoid empty context).

        Args:
            query: The question or sub-query string.
            paragraphs: List of paragraph dicts (must have `text_key`).
            threshold: Pruning threshold. Uses default_threshold if None.
            text_key: Key in paragraph dict containing the text.
            min_keep: Minimum number of paragraphs to keep.

        Returns:
            Tuple of (kept_paragraphs, all_scores).
        """
        if threshold is None:
            threshold = self.default_threshold

        # Score all paragraphs
        texts = [p[text_key] for p in paragraphs]
        scores = self.score_batch(query, texts)

        # Annotate all paragraph dicts with scores
        for p, s in zip(paragraphs, scores):
            p["prm_score"] = s
            p["prm_threshold"] = threshold
            p["prm_pass"] = s >= threshold

        # Filter by threshold
        kept = [p for p in paragraphs if p["prm_pass"]]

        # Safety floor: if nothing passes, keep top-k by score
        if len(kept) < min_keep:
            sorted_paras = sorted(paragraphs, key=lambda x: x["prm_score"], reverse=True)
            kept = sorted_paras[:min_keep]
            for p in kept:
                p["prm_pass"] = True  # mark as kept despite being below threshold

        return kept, scores

    def rank_steps(
        self,
        query: str,
        paragraphs: List[Dict],
        text_key: str = "text",
    ) -> List[Dict]:
        """
        Score all paragraphs and return them sorted by score descending.

        No threshold filtering - useful for debugging and analysis.

        Args:
            query: The question or sub-query string.
            paragraphs: List of paragraph dicts.
            text_key: Key in paragraph dict containing the text.

        Returns:
            All paragraphs sorted by PRM score descending.
        """
        texts = [p[text_key] for p in paragraphs]
        scores = self.score_batch(query, texts)

        for p, s in zip(paragraphs, scores):
            p["prm_score"] = s

        return sorted(paragraphs, key=lambda x: x["prm_score"], reverse=True)
