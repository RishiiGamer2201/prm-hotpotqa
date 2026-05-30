"""
FAISS-based retriever for HotpotQA distractor setting.

Handles two-hop iterative retrieval with bridge entity extraction.
Builds per-question FAISS indexes from the 10-paragraph pool and
uses cosine similarity (via L2-normalized inner product) for ranking.
"""

import random
import numpy as np
import torch
import faiss
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import copy

# Global seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class HotpotRetriever:
    """
    Two-hop retriever for HotpotQA distractor setting.

    Embeds paragraphs and queries using sentence-transformers, builds
    a FAISS IndexFlatIP index per question, and supports iterative
    hop 1 -> bridge entity -> hop 2 retrieval.

    Attributes:
        embed_model: SentenceTransformer for generating embeddings.
        device: torch device for embedding computation.
        nlp: spaCy NLP pipeline for named entity recognition.
    """

    def __init__(
        self,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the retriever.

        Args:
            embed_model: HuggingFace model for sentence embeddings (384-dim).
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.embed_model = SentenceTransformer(embed_model, device=device)
        print(f"[Retriever] Loaded {embed_model} on {device}")

        # Load spaCy for bridge entity extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("[Retriever] Loaded spaCy en_core_web_sm")
        except OSError:
            print("[Retriever] WARNING: spaCy model not found. "
                  "Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts and L2-normalize for cosine similarity.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), 384), L2-normalized.
        """
        embeddings = self.embed_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        # L2-normalize so inner product == cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        return (embeddings / norms).astype(np.float32)

    def build_index(self, paragraphs: List[Dict]) -> faiss.Index:
        """
        Build a FAISS IndexFlatIP from paragraph texts.

        Embeddings are L2-normalized before indexing, so inner product
        gives cosine similarity.

        Args:
            paragraphs: List of paragraph dicts with 'text' key.

        Returns:
            FAISS IndexFlatIP index.
        """
        texts = [p["text"] for p in paragraphs]
        embeddings = self._embed(texts)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return index

    def retrieve_hop1(
        self,
        question: str,
        paragraphs: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Hop 1 retrieval: embed the question, search against all 10 paragraphs.

        Returns all paragraphs ranked by cosine similarity to the question.
        The full set is returned so the PRM can score everything.

        Args:
            question: The original question string.
            paragraphs: The 10-paragraph pool for this question.
            top_k: Number of paragraphs to retrieve (default: all 10).

        Returns:
            List of paragraph dicts ranked by cosine similarity.
        """
        # Deep copy to avoid mutating original dicts
        paras = [copy.deepcopy(p) for p in paragraphs]

        index = self.build_index(paras)
        q_emb = self._embed([question])

        k = min(top_k, len(paras))
        scores, indices = index.search(q_emb, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            para = paras[idx]
            para["retrieval_score"] = float(score)
            para["retrieval_rank"] = rank
            para["hop"] = 1
            results.append(para)

        return results

    def extract_bridge_entity(
        self,
        question: str,
        hop1_context: List[Dict],
    ) -> str:
        """
        Extract the bridge entity connecting hop 1 to hop 2.

        Strategy:
        1. Use spaCy NER to extract named entities from hop 1 context.
        2. Find entities that appear in the context but are NOT the
           question's subject (heuristic: not in the first 5 words).
        3. Return the most common such entity as the bridge.

        Fallback: If NER fails or finds nothing, return the first sentence
        of the highest-scored hop 1 paragraph as the sub-query.

        Args:
            question: The original question string.
            hop1_context: PRM-pruned paragraphs from hop 1.

        Returns:
            Bridge entity string or fallback sub-query.
        """
        if not hop1_context:
            return question

        # Sort by PRM score if available, else by retrieval score
        sorted_ctx = sorted(
            hop1_context,
            key=lambda x: x.get("prm_score", x.get("retrieval_score", 0)),
            reverse=True,
        )

        # Try spaCy NER
        if self.nlp is not None:
            # Combine all hop 1 context text
            combined_text = " ".join(p["text"] for p in sorted_ctx)
            doc = self.nlp(combined_text)

            # Get question tokens for filtering
            q_tokens = set(question.lower().split())

            # Extract entities not in the question subject
            entities = []
            for ent in doc.ents:
                ent_lower = ent.text.lower()
                # Skip entities that are fully contained in the question
                ent_words = set(ent_lower.split())
                if not ent_words.issubset(q_tokens):
                    entities.append(ent.text)

            if entities:
                # Return the first (most prominent) bridge entity
                return entities[0]

        # Fallback: first sentence of the highest-scored paragraph
        best_para = sorted_ctx[0]
        sentences = best_para.get("sentences", [])
        if sentences:
            return sentences[0]
        # Last resort: return the first 100 chars of the best paragraph
        return best_para["text"][:100]

    def generate_hop2_query(
        self,
        question: str,
        bridge_entity: str,
        hop1_context: List[Dict],
    ) -> str:
        """
        Combine bridge entity with original question for hop 2 query.

        Args:
            question: The original question.
            bridge_entity: Entity extracted from hop 1 context.
            hop1_context: PRM-pruned paragraphs from hop 1 (unused, for extension).

        Returns:
            Hop 2 query string.
        """
        return f"{bridge_entity} {question}"

    def retrieve_hop2(
        self,
        question: str,
        hop1_pruned: List[Dict],
        all_paragraphs: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Hop 2 retrieval using the bridge entity as a query modifier.

        Extracts bridge entity from hop 1 pruned results, forms a hop 2
        query, and retrieves from the full 10-paragraph pool. Results
        include both hop 1 pruned paragraphs and newly retrieved ones
        so the PRM can re-score the full context.

        Args:
            question: The original question.
            hop1_pruned: PRM-pruned paragraphs from hop 1.
            all_paragraphs: The full 10-paragraph pool.
            top_k: Number of paragraphs to retrieve.

        Returns:
            List of paragraph dicts ranked by cosine similarity to hop 2 query.
        """
        # Extract bridge entity and form hop 2 query
        bridge_entity = self.extract_bridge_entity(question, hop1_pruned)
        hop2_query = self.generate_hop2_query(question, bridge_entity, hop1_pruned)

        # Deep copy all paragraphs for hop 2
        paras = [copy.deepcopy(p) for p in all_paragraphs]

        # Build index from all paragraphs
        index = self.build_index(paras)
        q_emb = self._embed([hop2_query])

        k = min(top_k, len(paras))
        scores, indices = index.search(q_emb, k)

        # Track which titles we already have from hop 1
        hop1_titles = {p["title"] for p in hop1_pruned}

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            para = paras[idx]
            para["retrieval_score"] = float(score)
            para["retrieval_rank"] = rank
            # Mark as hop 2 if not already in hop 1 pruned
            if para["title"] in hop1_titles:
                para["hop"] = 1  # already seen in hop 1
            else:
                para["hop"] = 2
            results.append(para)

        return results, hop2_query


def build_paragraph_dicts(example: Dict) -> List[Dict]:
    """
    Convert a HotpotQA example into a list of paragraph dicts.

    HotpotQA provides context as:
        context.title: list of 10 titles
        context.sentences: list of 10 lists of sentences
        supporting_facts.title: list of gold paragraph titles

    Args:
        example: A single HotpotQA validation example.

    Returns:
        List of 10 paragraph dicts with keys:
        title, text, sentences, is_gold, hop.
    """
    context = example["context"]
    gold_titles = set(example["supporting_facts"]["title"])

    paragraphs = []
    for title, sents in zip(context["title"], context["sentences"]):
        paragraphs.append({
            "title": title,
            "text": " ".join(sents),
            "sentences": list(sents),
            "is_gold": title in gold_titles,
            "hop": 0,  # populated during pipeline execution
        })

    return paragraphs
