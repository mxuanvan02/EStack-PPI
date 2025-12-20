"""
Generate feature-importance artifacts (PNG + CSV) for HybridStackPPI.

Usage:
  python -m experiments.generate_importance --dataset human --n-splits 5 --cache-version v3
"""

import argparse
from pathlib import Path

from experiments import run as exp_run


def main():
    parser = argparse.ArgumentParser(description="Generate HybridStackPPI feature-importance plots.")
    parser.add_argument("--dataset", choices=["human", "yeast"], default="human", help="Which bundled BioGrid dataset to use.")
    parser.add_argument("--pairing", choices=["concat", "avgdiff"], default="concat", help="Pairing strategy.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds (>=2 recommended).")
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D", help="ESM2 model name.")
    parser.add_argument("--h5-cache", default="cache/esm2_embeddings.h5", help="Path to ESM embedding cache (.h5).")
    parser.add_argument("--cache-version", default="v3", help="Cache version tag for feature matrices.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU parallelism for training.")
    args = parser.parse_args()

    base = Path("data/BioGrid/Human") if args.dataset == "human" else Path("data/BioGrid/Yeast")
    fasta_path = base / ("human_dict.fasta" if args.dataset == "human" else "yeast_dict.fasta")
    pairs_path = base / ("human_pairs.tsv" if args.dataset == "human" else "yeast_pairs.tsv")

    print(f"[Info] Using dataset: {args.dataset} | FASTA={fasta_path} | pairs={pairs_path}")
    print("[Info] CV folds:", args.n_splits)

    res = exp_run.run_experiment(
        fasta_path=str(fasta_path),
        pairs_path=str(pairs_path),
        h5_cache_path=args.h5_cache,
        model_factory=lambda n_jobs: exp_run.create_stacking_pipeline_for_notebook(  # type: ignore[attr-defined]
            pairing_strategy=args.pairing,
            n_jobs=n_jobs,
            h5_cache_path=args.h5_cache,
            esm_model_name=args.esm_model,
        ),
        pairing_strategy=args.pairing,
        n_splits=args.n_splits,
        esm_model_name=args.esm_model,
        n_jobs=args.n_jobs,
        cache_version=args.cache_version,
    )

    print("\n[Done] Importance artifacts should be saved as:")
    print(" - feature_importance_paper.png")
    print(" - feature_importance_hybrid.png")
    print(" - feature_importance_top.csv")
    print("\nMetrics summary:")
    print(res)


if __name__ == "__main__":
    main()
