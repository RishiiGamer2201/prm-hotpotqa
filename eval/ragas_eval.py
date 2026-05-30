"""
RAGAS evaluation for the multi-hop QA pipeline.

Evaluates pipeline outputs using 5 RAGAS metrics with bootstrap
confidence intervals. Calls Ollama directly via its REST API
for LLM-based metrics - no API keys needed.

Run:
    python eval/ragas_eval.py --llm ollama --t04 results/t0.4_raw.jsonl --t06 results/t0.6_raw.jsonl --output results/
"""

import os
import json
import csv
import argparse
import requests
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Global seed
SEED = 42
np.random.seed(SEED)

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(prompt: str, model: str = "mistral") -> str:
    """Call Ollama REST API directly. No wrappers, no async, no API keys."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        print(f"  [Ollama error] {e}")
        return ""


def score_faithfulness(question: str, answer: str, contexts: List[str]) -> float:
    """
    Faithfulness: are claims in the answer supported by the context?
    Returns a score between 0 and 1.
    """
    if not answer or not contexts:
        return 0.0

    context_str = "\n".join(contexts[:3])
    prompt = (
        f"Given the following context and answer, determine what fraction of "
        f"claims in the answer are supported by the context.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Answer: {answer}\n\n"
        f"Respond with ONLY a number between 0.0 and 1.0 where 1.0 means "
        f"all claims are supported and 0.0 means no claims are supported.\n"
        f"Score:"
    )
    raw = ollama_generate(prompt)
    return _parse_score(raw)


def score_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy: does the answer address the question?
    Returns a score between 0 and 1.
    """
    if not answer:
        return 0.0

    prompt = (
        f"Given the question and answer below, rate how relevant the answer "
        f"is to the question.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Respond with ONLY a number between 0.0 and 1.0 where 1.0 means "
        f"the answer perfectly addresses the question and 0.0 means it is "
        f"completely irrelevant.\n"
        f"Score:"
    )
    raw = ollama_generate(prompt)
    return _parse_score(raw)


def score_context_precision(contexts: List[str], gold_titles: set, all_titles: List[str]) -> float:
    """
    Context Precision: what fraction of retrieved paragraphs are relevant (gold)?
    This is computed directly without LLM calls.
    """
    if not all_titles:
        return 0.0
    relevant = sum(1 for t in all_titles if t in gold_titles)
    return relevant / len(all_titles)


def score_context_recall(contexts: List[str], gold_titles: set, kept_titles: set) -> float:
    """
    Context Recall: did retrieval cover all gold supporting facts?
    This is computed directly without LLM calls.
    """
    if not gold_titles:
        return 0.0
    found = sum(1 for t in gold_titles if t in kept_titles)
    return found / len(gold_titles)


def score_answer_correctness(question: str, answer: str, gold_answer: str) -> float:
    """
    Answer Correctness: is the answer factually correct?
    Returns a score between 0 and 1.
    """
    if not answer:
        return 0.0

    # Quick exact/substring match bonus
    if gold_answer.lower().strip() in answer.lower().strip():
        return 1.0

    prompt = (
        f"Compare the predicted answer with the gold answer for the given question.\n\n"
        f"Question: {question}\n"
        f"Gold Answer: {gold_answer}\n"
        f"Predicted Answer: {answer}\n\n"
        f"Rate the correctness of the predicted answer.\n"
        f"Respond with ONLY a number between 0.0 and 1.0 where 1.0 means "
        f"the predicted answer is fully correct and 0.0 means completely wrong.\n"
        f"Score:"
    )
    raw = ollama_generate(prompt)
    return _parse_score(raw)


def _parse_score(raw: str) -> float:
    """Parse a float score from LLM output, with fallback."""
    try:
        # Extract the first number from the response
        import re
        match = re.search(r"(0?\.\d+|1\.0|0|1)", raw)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        pass
    return 0.0


def load_results(path: str) -> List[Dict]:
    """Load pipeline results from JSONL."""
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def bootstrap_ci(
    scores: List[float],
    n_iter: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Compute bootstrap confidence interval for a list of scores.

    Args:
        scores: List of per-question metric scores.
        n_iter: Number of bootstrap iterations.
        ci: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with mean, ci_lower, ci_upper, ci_width.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(scores)
    boot_means = []
    for _ in range(n_iter):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return {
        "mean": float(arr.mean()),
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_width": upper - lower,
    }


def evaluate_threshold(
    results: List[Dict],
    threshold_label: str,
    checkpoint_dir: str = None,
) -> Dict:
    """
    Evaluate all 5 metrics for a single threshold's results.
    Supports checkpoint/resume - saves after each question.

    Args:
        results: Pipeline results for one threshold.
        threshold_label: Label like 't0.4' or 't0.6'.
        checkpoint_dir: Directory to save/load checkpoints.

    Returns:
        Dict mapping metric names to bootstrap CI dicts.
    """
    print(f"\n[Eval] Evaluating {threshold_label} ({len(results)} questions)...")

    all_scores = {m: [] for m in METRIC_NAMES}
    start_idx = 0

    # Resume from checkpoint if available
    ckpt_path = None
    if checkpoint_dir:
        safe_label = threshold_label.replace("=", "")
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{safe_label}.jsonl")
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        for m in METRIC_NAMES:
                            all_scores[m].append(entry[m])
                        start_idx += 1
            print(f"  [Resume] Loaded {start_idx} completed questions from checkpoint")

    for i in range(start_idx, len(results)):
        r = results[i]
        question = r["question"]
        answer = r["predicted_answer"]
        gold_answer = r["gold_answer"]
        contexts = [p["text"] for p in r.get("final_contexts", [])]

        # Get gold titles from hop1_retrieved
        gold_titles = set()
        for p in r.get("hop1_retrieved", []):
            if p.get("is_gold", False):
                gold_titles.add(p["title"])

        kept_titles = {p["title"] for p in r.get("final_contexts", [])}
        all_titles = [p["title"] for p in r.get("final_contexts", [])]

        print(f"\n  [{i+1}/{len(results)}] Q: {question[:80]}...")

        scores = {}

        # Score each metric
        print(f"    -> faithfulness...", end="", flush=True)
        scores["faithfulness"] = score_faithfulness(question, answer, contexts)
        all_scores["faithfulness"].append(scores["faithfulness"])
        print(f" {scores['faithfulness']:.2f}")

        print(f"    -> answer_relevancy...", end="", flush=True)
        scores["answer_relevancy"] = score_answer_relevancy(question, answer)
        all_scores["answer_relevancy"].append(scores["answer_relevancy"])
        print(f" {scores['answer_relevancy']:.2f}")

        print(f"    -> context_precision...", end="", flush=True)
        scores["context_precision"] = score_context_precision(contexts, gold_titles, all_titles)
        all_scores["context_precision"].append(scores["context_precision"])
        print(f" {scores['context_precision']:.2f}")

        print(f"    -> context_recall...", end="", flush=True)
        scores["context_recall"] = score_context_recall(contexts, gold_titles, kept_titles)
        all_scores["context_recall"].append(scores["context_recall"])
        print(f" {scores['context_recall']:.2f}")

        print(f"    -> answer_correctness...", end="", flush=True)
        scores["answer_correctness"] = score_answer_correctness(question, answer, gold_answer)
        all_scores["answer_correctness"].append(scores["answer_correctness"])
        print(f" {scores['answer_correctness']:.2f}")

        # Save checkpoint after each question
        if ckpt_path:
            with open(ckpt_path, "a") as f:
                f.write(json.dumps(scores) + "\n")

    # Compute bootstrap CIs
    metric_results = {}
    for name in METRIC_NAMES:
        scores = all_scores[name]
        metric_results[name] = bootstrap_ci(scores)
        print(
            f"  {name}: "
            f"{metric_results[name]['mean']:.4f} "
            f"[{metric_results[name]['ci_lower']:.4f}, "
            f"{metric_results[name]['ci_upper']:.4f}]"
        )

    return metric_results


def save_results_json(
    t04_metrics: Dict,
    t06_metrics: Dict,
    output_dir: str,
    n_questions: int = 500,
):
    """Save results as JSON."""
    results = {
        "seed": SEED,
        "n_questions": n_questions,
        "n_bootstrap_iterations": 1000,
        "results": {
            "t0.4": t04_metrics,
            "t0.6": t06_metrics,
        },
    }
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Eval] Saved results to {path}")


def save_results_csv(t04_metrics: Dict, t06_metrics: Dict, output_dir: str):
    """Save results as CSV."""
    path = os.path.join(output_dir, "results.csv")
    header = [
        "system", "faithfulness", "ans_rel", "ctx_prec", "ctx_rec", "ans_corr",
        "faith_ci", "ans_rel_ci", "ctx_prec_ci", "ctx_rec_ci", "ans_corr_ci",
    ]

    rows = []
    for label, metrics in [("PRM t=0.4", t04_metrics), ("PRM t=0.6", t06_metrics)]:
        row = [label]
        ci_cols = []
        for name in METRIC_NAMES:
            m = metrics[name]
            row.append(f"{m['mean']:.4f}")
            ci_cols.append(f"[{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]")
        row.extend(ci_cols)
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[Eval] Saved CSV to {path}")


def generate_plots(
    t04_results: List[Dict],
    t06_results: List[Dict],
    t04_metrics: Dict,
    t06_metrics: Dict,
    output_dir: str,
):
    """
    Generate evaluation plots:
    1. Bar chart comparing metrics across thresholds
    2. PRM score distributions
    3. Context recall by hop
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Plot 1: Metric bars ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(METRIC_NAMES))
    width = 0.35

    t04_means = [t04_metrics[m]["mean"] for m in METRIC_NAMES]
    t06_means = [t06_metrics[m]["mean"] for m in METRIC_NAMES]
    t04_errs = [
        (t04_metrics[m]["mean"] - t04_metrics[m]["ci_lower"],
         t04_metrics[m]["ci_upper"] - t04_metrics[m]["mean"])
        for m in METRIC_NAMES
    ]
    t06_errs = [
        (t06_metrics[m]["mean"] - t06_metrics[m]["ci_lower"],
         t06_metrics[m]["ci_upper"] - t06_metrics[m]["mean"])
        for m in METRIC_NAMES
    ]

    ax.bar(
        x - width / 2, t04_means, width, label="t=0.4",
        yerr=list(zip(*t04_errs)), capsize=3, color="#4C72B0", alpha=0.85
    )
    ax.bar(
        x + width / 2, t06_means, width, label="t=0.6",
        yerr=list(zip(*t06_errs)), capsize=3, color="#DD8452", alpha=0.85
    )

    ax.set_ylabel("Score")
    ax.set_title("RAGAS Metrics: PRM Threshold Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in METRIC_NAMES], rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metric_bars.png"), dpi=150)
    plt.close()
    print(f"[Plots] Saved metric_bars.png")

    # --- Plot 2: PRM Score Distributions ---
    def extract_prm_scores(results, hop_key):
        scores = []
        for r in results:
            for p in r.get(hop_key, []):
                if "prm_score" in p:
                    scores.append(p["prm_score"])
        return scores

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, results, color) in zip(axes, [
        ("t=0.4", t04_results, "#4C72B0"),
        ("t=0.6", t06_results, "#DD8452"),
    ]):
        hop1_scores = extract_prm_scores(results, "hop1_retrieved")
        hop2_scores = extract_prm_scores(results, "hop2_retrieved")

        ax.hist(hop1_scores, bins=50, alpha=0.6, label="Hop 1", color="#4C72B0")
        ax.hist(hop2_scores, bins=50, alpha=0.6, label="Hop 2", color="#DD8452")
        threshold = 0.4 if "0.4" in label else 0.6
        ax.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
        ax.set_xlabel("PRM Score")
        ax.set_ylabel("Count")
        ax.set_title(f"PRM Score Distribution ({label})")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "score_distributions.png"), dpi=150)
    plt.close()
    print(f"[Plots] Saved score_distributions.png")

    # --- Plot 3: Context Recall by Hop ---
    def compute_hop_recall(results):
        hop1_recalls, hop2_recalls = [], []
        for r in results:
            # Get all gold titles from the 10-paragraph pool
            gold_titles = set()
            for p in r.get("hop1_retrieved", []):
                if p.get("is_gold", False):
                    gold_titles.add(p["title"])

            # Titles kept after hop 1 PRM pruning
            hop1_kept_titles = {p["title"] for p in r.get("hop1_kept", [])}

            # Titles in final output (after both hops + dedup)
            final_titles = {p["title"] for p in r.get("final_contexts", [])}

            # For each gold title, check which hop found it
            for title in gold_titles:
                # Was this gold title found in hop 1 pruned results?
                if title in hop1_kept_titles:
                    # Found in hop 1 - did it survive to final?
                    if title in final_titles:
                        hop1_recalls.append(1.0)
                    else:
                        hop1_recalls.append(0.0)
                else:
                    # Not in hop 1 - was it rescued by hop 2?
                    if title in final_titles:
                        hop2_recalls.append(1.0)
                    else:
                        hop2_recalls.append(0.0)

        return (
            np.mean(hop1_recalls) if hop1_recalls else 0,
            np.mean(hop2_recalls) if hop2_recalls else 0,
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    t04_h1, t04_h2 = compute_hop_recall(t04_results)
    t06_h1, t06_h2 = compute_hop_recall(t06_results)

    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [t04_h1, t04_h2], width, label="t=0.4", color="#4C72B0")
    ax.bar(x + width / 2, [t06_h1, t06_h2], width, label="t=0.6", color="#DD8452")
    ax.set_ylabel("Recall")
    ax.set_title("Context Recall by Hop")
    ax.set_xticks(x)
    ax.set_xticklabels(["Hop 1", "Hop 2"])
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "context_recall_hop.png"), dpi=150)
    plt.close()
    print(f"[Plots] Saved context_recall_hop.png")


def main():
    """Main entry point for RAGAS evaluation."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--llm", choices=["ollama", "hf"], default="ollama",
        help="LLM backend: 'ollama' (default, free) or 'hf' (HuggingFace fallback)"
    )
    parser.add_argument(
        "--t04", required=True,
        help="Path to t=0.4 results JSONL"
    )
    parser.add_argument(
        "--t06", required=True,
        help="Path to t=0.6 results JSONL"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Verify Ollama is running
    if args.llm == "ollama":
        print("[Eval] Checking Ollama connection...")
        try:
            test = ollama_generate("Say OK", model="mistral")
            if test:
                print(f"[Eval] Ollama OK - response: {test[:50]}")
            else:
                print("[Eval] WARNING: Ollama returned empty response")
        except Exception as e:
            print(f"[Eval] ERROR: Cannot reach Ollama: {e}")
            print("[Eval] Make sure 'ollama serve' is running in another terminal")
            return

    # Load results
    t04_results = load_results(args.t04)
    t06_results = load_results(args.t06)
    print(f"[Eval] Loaded {len(t04_results)} results for t=0.4")
    print(f"[Eval] Loaded {len(t06_results)} results for t=0.6")

    # Evaluate both thresholds
    os.makedirs(args.output, exist_ok=True)
    ckpt_dir = os.path.join(args.output, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    t04_metrics = evaluate_threshold(t04_results, "t=0.4", checkpoint_dir=ckpt_dir)
    t06_metrics = evaluate_threshold(t06_results, "t=0.6", checkpoint_dir=ckpt_dir)

    # Save results
    n_questions = len(t04_results)
    save_results_json(t04_metrics, t06_metrics, args.output, n_questions)
    save_results_csv(t04_metrics, t06_metrics, args.output)

    # Generate plots
    generate_plots(t04_results, t06_results, t04_metrics, t06_metrics, args.output)

    print("\n[Eval] Evaluation complete!")


if __name__ == "__main__":
    main()
