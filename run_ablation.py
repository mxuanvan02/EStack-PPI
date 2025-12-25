#!/usr/bin/env python3
"""
E-StackPPI Ablation Runner
==========================
This script runs the "Embed-Only" (Protein Language Model features only) 
pipeline on Human and Yeast datasets.

Usage:
    python EStackPPI/run_ablation.py --dataset both
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path to allow imports from hybridstack
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run import run_experiment, define_stacking_columns
from hybridstack.feature_engine import FeatureEngine, EmbeddingComputer
from hybridstack.metrics import print_detailed_fold_table
from EStackPPI.pipeline import create_estack_pipeline

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    parser = argparse.ArgumentParser(description="E-StackPPI (Embedding-based) Ablation Study")
    parser.add_argument("--dataset", choices=["human", "yeast", "both"], default="both")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--esm-model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--h5-cache", default="/media/SAS/Van/HybridStackPPI/cache/esm2_embeddings.h5")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    print("=" * 100)
    print("🚀 E-STACKPPI: HIGH-DEFINITION ABLATION STUDY")
    print("=" * 100)
    
    # Dataset configurations (Focus exclusively on Yeast DIP as requested)
    all_datasets = [
        {
            "name": "Yeast DIP",
            "fasta": str(PROJECT_ROOT / "data/data/yeast/sequences.fasta"),
            "pairs": str(PROJECT_ROOT / "data/data/yeast/pairs.tsv"),
        }
    ]
    
    datasets = all_datasets # Only DIP remains

    for ds in datasets:
        print(f"\n" + "#" * 100)
        print(f"### DATASET: {ds['name']}")
        print("#" * 100)
        
        # 1. Initialize Engine once to get column names
        embedding_computer = EmbeddingComputer(model_name=args.esm_model)
        feature_engine = FeatureEngine(h5_cache_path=args.h5_cache, embedding_computer=embedding_computer)
        
        # Define variant columns
        _, all_embed_cols = define_stacking_columns(feature_engine, pairing_strategy="concat")
        global_only_cols = [c for c in all_embed_cols if "Global_ESM" in c]
        
        variants = [
            {"name": "E-StackPPI (Hybrid-Embed)", "cols": all_embed_cols},
            {"name": "E-StackPPI (ESM-Only)", "cols": global_only_cols},
        ]
        
        for var in variants:
            print(f"\n📦 RUNNING VARIANT: {var['name']}")
            print("-" * 100)
            
            try:
                # Factory for the refined stacking pipeline
                model_factory = lambda n_jobs=-1: create_estack_pipeline(var['cols'], n_jobs, use_selector=True)
                
                # run_experiment returns (avg_metrics, fold_list)
                result = run_experiment(
                    fasta_path=ds['fasta'],
                    pairs_path=ds['pairs'],
                    h5_cache_path=args.h5_cache,
                    model_factory=model_factory,
                    n_splits=args.n_splits,
                    n_jobs=args.n_jobs,
                    esm_model_name=args.esm_model,
                    pairing_strategy="concat",
                    experiment_title=f"{ds['name']} - {var['name']}"
                )
                
                # Display individual fold table
                if isinstance(result, tuple) and len(result) == 2:
                    avg_metrics, fold_list = result
                    print_detailed_fold_table(fold_list, title=f"Results: {ds['name']} - {var['name']}")
                else:
                    print(f"\n✅ Completed {var['name']}")
                    if result:
                        print(f"Summary Metrics: {result}")
                
            except Exception as e:
                print(f"❌ Error for {var['name']}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
