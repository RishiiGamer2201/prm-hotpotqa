"""
End-to-end multi-hop QA pipeline with Process Reward Model.

Orchestrates: data loading -> hop 1 retrieval -> PRM pruning ->
hop 2 retrieval -> PRM pruning -> answer synthesis.

Run:
    python src/pipeline.py --threshold 0.4 --output results/t0.4_raw.jsonl
    python src/pipeline.py --threshold 0.6 --output results/t0.6_raw.jsonl
"""

import os
import sys
import json
import random
import argparse
import copy
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prm import ProcessRewardModel
from retriever import HotpotRetriever, build_paragraph_dicts

# Global seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_hotpotqa_sample(n: int = 500, seed: int = 42) -> List[Dict]:
    """
    Load a random sample of n questions from HotpotQA validation (distractor).

    Args:
        n: Number of questions to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of HotpotQA example dicts.
    """
    random.seed(seed)
    print(f"[Data] Loading HotpotQA distractor validation split...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    indices = random.sample(range(len(ds)), n)
    indices = sorted(indices)
    samples = [ds[i] for i in indices]
    print(f"[Data] Loaded {len(samples)} questions (seed={seed})")
    return samples


class MultiHopPipeline:
    """
    End-to-end multi-hop QA pipeline with PRM-gated retrieval.

    Pipeline steps for each question:
    1. Hop 1 retrieval: embed question, retrieve from 10-paragraph pool
    2. Hop 1 PRM: score and prune hop 1 results
    3. Hop 2 retrieval: extract bridge entity, re-retrieve
    4. Hop 2 PRM: score and prune hop 2 results
    5. Answer synthesis: generate answer from pruned context via flan-t5

    Attributes:
        prm: ProcessRewardModel instance.
        retriever: HotpotRetriever instance.
        tokenizer: flan-t5-large tokenizer.
        generator: flan-t5-large model.
        device: torch device.
    """

    def __init__(
        self,
        prm: ProcessRewardModel,
        retriever: HotpotRetriever,
        generator_model: str = "google/flan-t5-large",
        device: Optional[str] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            prm: Initialized ProcessRewardModel.
            retriever: Initialized HotpotRetriever.
            generator_model: HuggingFace model for answer generation.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.prm = prm
        self.retriever = retriever

        # Load answer generator
        print(f"[Pipeline] Loading {generator_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
        self.generator.to(device)
        self.generator.eval()
        print(f"[Pipeline] Generator loaded on {device}")

    def synthesize_answer(
        self, question: str, contexts: List[Dict]
    ) -> str:
        """
        Generate an answer from the kept contexts using flan-t5-large.

        If combined context exceeds flan-t5's 512-token limit, truncates
        to the top 3 paragraphs by PRM score.

        Args:
            question: The original question string.
            contexts: List of paragraph dicts that passed PRM.

        Returns:
            Generated answer string.
        """
        if not contexts:
            return "No relevant context found."

        # Sort by PRM score and take top 3 if context is too long
        sorted_ctx = sorted(
            contexts,
            key=lambda x: x.get("prm_score", 0),
            reverse=True,
        )

        # Build context string
        context_parts = []
        for p in sorted_ctx:
            context_parts.append(f"[{p['title']}] {p['text']}")

        context_str = "\n".join(context_parts)

        # Check token length and truncate if needed
        prompt = (
            f"Answer the following question using only the provided context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\nAnswer:"
        )

        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        if tokens["input_ids"].shape[1] > 512:
            # Truncate to top 3 paragraphs
            context_parts = context_parts[:3]
            context_str = "\n".join(context_parts)
            prompt = (
                f"Answer the following question using only the provided context.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\nAnswer:"
            )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return answer

    def run_single(
        self,
        question: str,
        paragraphs: List[Dict],
        gold_answer: str,
        threshold: float,
        question_id: str = "",
    ) -> Dict:
        """
        Run the full multi-hop pipeline for a single question.

        Args:
            question: The question string.
            paragraphs: The 10-paragraph pool (list of dicts).
            gold_answer: Gold-standard answer (for recording, not inference).
            threshold: PRM pruning threshold.
            question_id: Unique identifier for the question.

        Returns:
            Dict with all intermediate and final results.
        """
        # Step 1: Hop 1 retrieval
        hop1_retrieved = self.retriever.retrieve_hop1(
            question, paragraphs, top_k=10
        )

        # Step 2: Hop 1 PRM pruning
        hop1_kept, hop1_scores = self.prm.prune_steps(
            question, hop1_retrieved, threshold=threshold
        )

        # Step 3: Hop 2 retrieval
        hop2_retrieved, hop2_query = self.retriever.retrieve_hop2(
            question, hop1_kept, paragraphs
        )

        # Step 4: Hop 2 PRM pruning
        hop2_kept, hop2_scores = self.prm.prune_steps(
            hop2_query, hop2_retrieved, threshold=threshold
        )

        # Step 5: Combine and deduplicate contexts
        seen_titles = set()
        final_contexts = []
        for p in hop1_kept + hop2_kept:
            if p["title"] not in seen_titles:
                seen_titles.add(p["title"])
                final_contexts.append(p)

        # Step 6: Answer synthesis
        predicted_answer = self.synthesize_answer(question, final_contexts)

        return {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "hop1_retrieved": self._serialize_paras(hop1_retrieved),
            "hop1_kept": self._serialize_paras(hop1_kept),
            "hop1_scores": hop1_scores,
            "hop2_query": hop2_query,
            "hop2_retrieved": self._serialize_paras(hop2_retrieved),
            "hop2_kept": self._serialize_paras(hop2_kept),
            "hop2_scores": hop2_scores,
            "final_contexts": self._serialize_paras(final_contexts),
            "threshold": threshold,
        }

    @staticmethod
    def _serialize_paras(paras: List[Dict]) -> List[Dict]:
        """
        Serialize paragraph dicts for JSON output.

        Converts any non-serializable values and creates a clean copy.
        """
        serialized = []
        for p in paras:
            sp = {}
            for k, v in p.items():
                if isinstance(v, (np.floating, np.integer)):
                    sp[k] = float(v)
                elif isinstance(v, np.ndarray):
                    sp[k] = v.tolist()
                else:
                    sp[k] = v
            serialized.append(sp)
        return serialized

    def run_all(
        self,
        dataset: List[Dict],
        threshold: float,
        output_path: str,
    ) -> List[Dict]:
        """
        Run the pipeline for all questions in the dataset.

        Saves checkpoints to output_path every 50 questions.

        Args:
            dataset: List of HotpotQA example dicts.
            threshold: PRM pruning threshold.
            output_path: Path for JSONL output file.

        Returns:
            List of result dicts.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        results = []
        checkpoint_interval = 50

        # Check for existing checkpoint
        existing = 0
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
                        existing += 1
            print(f"[Pipeline] Resuming from checkpoint: {existing} questions done")

        for i, example in enumerate(tqdm(
            dataset[existing:],
            initial=existing,
            total=len(dataset),
            desc=f"Pipeline t={threshold}",
        )):
            idx = existing + i

            # Build paragraph dicts
            paragraphs = build_paragraph_dicts(example)

            # Run single question
            result = self.run_single(
                question=example["question"],
                paragraphs=paragraphs,
                gold_answer=example["answer"],
                threshold=threshold,
                question_id=example.get("id", str(idx)),
            )

            results.append(result)

            # Checkpoint save every 50 questions
            if (idx + 1) % checkpoint_interval == 0:
                self._save_jsonl(results, output_path)
                print(f"\n[Checkpoint] Saved {len(results)} results to {output_path}")

        # Final save
        self._save_jsonl(results, output_path)
        print(f"\n[Pipeline] Complete. Saved {len(results)} results to {output_path}")

        return results

    @staticmethod
    def _save_jsonl(results: List[Dict], path: str):
        """Save results as JSONL."""
        with open(path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


def main():
    """Main entry point for running the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run multi-hop QA pipeline with PRM"
    )
    parser.add_argument(
        "--threshold", type=float, required=True,
        help="PRM pruning threshold (e.g., 0.4 or 0.6)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output JSONL file path (e.g., results/t0.4_raw.jsonl)"
    )
    parser.add_argument(
        "--n-questions", type=int, default=500,
        help="Number of questions to process (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    # Load data
    dataset = load_hotpotqa_sample(n=args.n_questions, seed=args.seed)

    # Initialize components
    prm = ProcessRewardModel(default_threshold=args.threshold)
    retriever = HotpotRetriever()
    pipeline = MultiHopPipeline(prm=prm, retriever=retriever)

    # Run pipeline
    pipeline.run_all(
        dataset=dataset,
        threshold=args.threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
