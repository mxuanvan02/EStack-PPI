#!/usr/bin/env python3
"""
ESM-2 Embedding Extraction Script for E-StackPPI

This script extracts ESM-2 embeddings from protein sequences and saves them
as numpy arrays for later use in PPI prediction.

Usage:
    python extract_esm2.py --dataset yeast   # Extract for Yeast-DIP
    python extract_esm2.py --dataset human   # Extract for Human-DIP
    python extract_esm2.py --dataset all     # Extract for both
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ESM-2 imports
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers not installed. Install with: pip install transformers")


def load_fasta(fasta_path: str) -> Dict[str, str]:
    """Load protein sequences from FASTA file."""
    sequences = {}
    with open(fasta_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line.split()[0][1:]
                sequences[header] = ""
            elif header:
                sequences[header] += line
    return sequences


def load_pairs(pairs_path: str) -> pd.DataFrame:
    """Load interaction pairs from TSV file."""
    pairs_df = pd.read_csv(
        pairs_path, sep="\t", header=None, 
        names=["protein1", "protein2", "label"]
    )
    return pairs_df


def canonicalize_pairs(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Sort protein IDs within each pair and drop duplicates."""
    if pairs_df.empty:
        return pairs_df
    
    df = pairs_df.copy()
    sorted_pairs = df[["protein1", "protein2"]].apply(
        lambda r: tuple(sorted([r["protein1"], r["protein2"]])), axis=1
    )
    df["protein1"], df["protein2"] = zip(*sorted_pairs)
    df["pair_key"] = df["protein1"] + "||" + df["protein2"]
    
    dup_count = df.duplicated("pair_key").sum()
    if dup_count > 0:
        print(f"  ⚠️ Removed {dup_count} duplicate pair rows")
    
    df = df.drop_duplicates(subset="pair_key", keep="first")
    df = df.drop(columns=["pair_key"]).reset_index(drop=True)
    return df


class ESM2Embedder:
    """ESM-2 Embedding Extractor using HuggingFace Transformers."""
    
    def __init__(
        self, 
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = None,
        max_length: int = 1024,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers package is required")
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔧 Loading ESM-2 model: {model_name}")
        print(f"   Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"   ✅ Model loaded successfully")
    
    def get_embedding(self, sequence: str) -> np.ndarray:
        """Extract mean-pooled embedding for a single sequence."""
        # Truncate if too long
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        
        # Tokenize
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean pooling over sequence length (excluding special tokens)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        
        # Mask padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        
        return mean_embedding.cpu().numpy().flatten()
    
    def get_embeddings_batch(
        self, 
        sequences: Dict[str, str], 
        batch_size: int = 8
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings for multiple sequences."""
        embeddings = {}
        protein_ids = list(sequences.keys())
        
        for i in tqdm(range(0, len(protein_ids), batch_size), desc="Extracting embeddings"):
            batch_ids = protein_ids[i:i+batch_size]
            batch_seqs = [sequences[pid] for pid in batch_ids]
            
            # Truncate sequences
            batch_seqs = [s[:self.max_length] for s in batch_seqs]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_seqs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            # Store results
            for j, pid in enumerate(batch_ids):
                embeddings[pid] = mean_embeddings[j].cpu().numpy()
        
        return embeddings


def build_pair_features(
    pairs_df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
) -> tuple:
    """Build feature matrix from protein pair embeddings."""
    X_rows = []
    y_rows = []
    valid_indices = []
    
    for idx, row in pairs_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]
        
        if p1 not in embeddings or p2 not in embeddings:
            continue
        
        # Concatenate embeddings
        v1 = embeddings[p1]
        v2 = embeddings[p2]
        pair_features = np.concatenate([v1, v2])
        
        X_rows.append(pair_features)
        y_rows.append(label)
        valid_indices.append(idx)
    
    X = np.vstack(X_rows)
    y = np.array(y_rows)
    
    return X, y, valid_indices


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESM-2 embeddings for E-StackPPI"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["yeast", "human", "all"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for embedding extraction"
    )
    parser.add_argument(
        "--model", type=str, default="facebook/esm2_t33_650M_UR50D",
        help="ESM-2 model to use"
    )
    args = parser.parse_args()
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    # Dataset configurations
    datasets = {
        "yeast": {
            "fasta_path": data_dir / "yeast" / "sequences.fasta",
            "pairs_path": data_dir / "yeast" / "pairs.tsv",
            "output_dir": data_dir / "yeast",
            "name": "DIP-Yeast",
        },
        "human": {
            "fasta_path": data_dir / "human" / "sequences.fasta",
            "pairs_path": data_dir / "human" / "pairs.tsv",
            "output_dir": data_dir / "human",
            "name": "DIP-Human",
        },
    }
    
    # Select datasets
    if args.dataset == "all":
        selected = ["yeast", "human"]
    else:
        selected = [args.dataset]
    
    # Initialize embedder
    embedder = ESM2Embedder(model_name=args.model)
    
    for dataset_key in selected:
        config = datasets[dataset_key]
        
        print(f"\n{'='*70}")
        print(f"📊 Processing {config['name']}")
        print(f"{'='*70}")
        
        # Check input files
        if not config["fasta_path"].exists():
            print(f"❌ FASTA file not found: {config['fasta_path']}")
            continue
        if not config["pairs_path"].exists():
            print(f"❌ Pairs file not found: {config['pairs_path']}")
            continue
        
        # Load data
        print(f"\n📂 Loading data...")
        sequences = load_fasta(str(config["fasta_path"]))
        pairs_df = load_pairs(str(config["pairs_path"]))
        pairs_df = canonicalize_pairs(pairs_df)
        
        print(f"   Proteins: {len(sequences):,}")
        print(f"   Pairs: {len(pairs_df):,}")
        
        # Extract embeddings
        print(f"\n🧬 Extracting ESM-2 embeddings...")
        embeddings = embedder.get_embeddings_batch(
            sequences, 
            batch_size=args.batch_size
        )
        
        # Build feature matrix
        print(f"\n🔧 Building feature matrix...")
        X, y, valid_indices = build_pair_features(pairs_df, embeddings)
        
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   Positive: {int(y.sum()):,}, Negative: {len(y) - int(y.sum()):,}")
        
        # Save outputs
        output_dir = config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        X_path = output_dir / "X_esm2.npy"
        y_path = output_dir / "y.npy"
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        print(f"\n💾 Saved outputs:")
        print(f"   {X_path}")
        print(f"   {y_path}")
    
    print(f"\n{'='*70}")
    print("🎉 ESM-2 EXTRACTION COMPLETED!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
