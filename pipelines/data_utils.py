import os
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pipelines.feature_engine import EmbeddingComputer, FeatureEngine
from pipelines.logger import PipelineLogger


def load_data(fasta_path: str, pairs_path: str):
    """Load protein sequences from FASTA and interaction pairs from TSV."""
    sequences: dict[str, str] = {}
    with open(fasta_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line.split()[0][1:]
                sequences[header] = ""
            elif header:
                sequences[header] += line

    pairs_df = pd.read_csv(pairs_path, sep="\t", header=None, names=["protein1", "protein2", "label"])
    return sequences, pairs_df


def canonicalize_pairs(
    pairs_df: pd.DataFrame,
    dataset_name: str | None = None,
    logger: "PipelineLogger | None" = None,  # type: ignore
) -> pd.DataFrame:
    """
    Sort protein IDs within each pair and drop duplicates to prevent inflated metrics.
    If conflicting labels exist for the same unordered pair, the first occurrence is kept.
    """
    if pairs_df.empty:
        return pairs_df

    df = pairs_df.copy()
    sorted_pairs = df[["protein1", "protein2"]].apply(lambda r: tuple(sorted([r["protein1"], r["protein2"]])), axis=1)
    df["protein1"], df["protein2"] = zip(*sorted_pairs)
    df["pair_key"] = df["protein1"] + "||" + df["protein2"]

    duplicate_mask = df.duplicated("pair_key", keep=False)
    dup_count = int(duplicate_mask.sum())
    conflict_keys = (
        df.loc[duplicate_mask].groupby("pair_key")["label"].nunique()
    )
    conflict_pairs = conflict_keys[conflict_keys > 1]

    df = df.drop_duplicates(subset="pair_key", keep="first")
    df = df.drop(columns=["pair_key"]).reset_index(drop=True)

    prefix = f"[{dataset_name}] " if dataset_name else ""
    if dup_count > 0:
        msg = f"{prefix}Removed {dup_count} duplicate pair rows (unordered)."
        if logger:
            logger.warning(msg)
        else:
            print(msg)
    if not conflict_pairs.empty:
        msg = (
            f"{prefix}Found {len(conflict_pairs)} pair(s) with conflicting labels; kept the first occurrence."
        )
        if logger:
            logger.warning(msg)
        else:
            print(msg)
    return df


def create_feature_matrix(
    pairs_df: pd.DataFrame, protein_features: dict, feature_names: List[str], pairing_strategy: str = "concat"
):
    """Build interaction-level feature matrix using concat or avg/diff strategy."""
    X_data, y_data, pair_indices = [], [], []

    if pairing_strategy == "concat":
        p1_feature_names = [f"P1_{name}" for name in feature_names]
        p2_feature_names = [f"P2_{name}" for name in feature_names]
        pair_feature_names = p1_feature_names + p2_feature_names
    elif pairing_strategy == "avgdiff":
        avg_feature_names = [f"Avg_{name}" for name in feature_names]
        diff_feature_names = [f"Diff_{name}" for name in feature_names]
        pair_feature_names = avg_feature_names + diff_feature_names
    else:
        raise ValueError(f"Unknown pairing strategy: {pairing_strategy}")

    for idx, row in pairs_df.iterrows():
        p1_id, p2_id, label = row["protein1"], row["protein2"], row["label"]

        # Canonicalize order to respect symmetry of PPI
        if p1_id > p2_id:
            p1_id, p2_id = p2_id, p1_id

        try:
            p1_feats = protein_features[p1_id].astype(np.float64)
            p2_feats = protein_features[p2_id].astype(np.float64)
        except KeyError:
            continue

        if pairing_strategy == "concat":
            pair_features = np.concatenate([p1_feats, p2_feats])
        elif pairing_strategy == "avgdiff":
            avg_feats = (p1_feats + p2_feats) / 2.0
            diff_feats = np.abs(p1_feats - p2_feats)
            pair_features = np.concatenate([avg_feats, diff_feats])

        X_data.append(pair_features)
        y_data.append(label)
        pair_indices.append(idx)

    X_df = pd.DataFrame(X_data, columns=pair_feature_names, index=pair_indices)
    y_s = pd.Series(y_data, index=pair_indices, name="label")
    return X_df, y_s


def get_cache_filename(
    pairs_path: str,
    pairing_strategy: str = "concat",
    esm_model_name: str = "esm2_t33_650M_UR50D",
    root_cache_dir: str = "cache",
    cache_version: str = "v2",
) -> str:
    """Build a dataset-specific cache filename from pairs path and ESM model."""
    parent_dir_name = os.path.basename(os.path.dirname(pairs_path)).lower()
    file_name_without_ext = os.path.splitext(os.path.basename(pairs_path))[0].lower()

    if parent_dir_name and parent_dir_name != "data":
        dataset_id = f"{parent_dir_name}_{file_name_without_ext}"
    else:
        dataset_id = file_name_without_ext

    safe_esm_model = esm_model_name.replace("/", "_").lower()
    filename = f"{dataset_id}_{safe_esm_model}_{pairing_strategy}_{cache_version}_features.h5"

    os.makedirs(root_cache_dir, exist_ok=True)
    return os.path.join(root_cache_dir, filename)


def save_feature_matrix_h5(X: pd.DataFrame, y: pd.Series, file_path: str):
    """Save X (DataFrame) and y (Series) to H5."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"  [Cache] Saving feature matrix to {file_path}...")
        with h5py.File(file_path, "w") as hf:
            hf.create_dataset("X_data", data=X.values)
            hf.create_dataset("X_cols", data=np.array(X.columns, dtype="S"))
            hf.create_dataset("X_index", data=X.index)

            hf.create_dataset("y_data", data=y.values)
            hf.create_dataset("y_index", data=y.index)
            hf.attrs["y_name"] = y.name if y.name else "label"
        print("  [Cache] Save complete.")
    except Exception as exc:  # noqa: BLE001
        print(f"  [Cache] ❌ ERROR saving cache to {file_path}: {exc}")


def load_feature_matrix_h5(file_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load X (DataFrame) and y (Series) from H5."""
    print(f"  [Cache] Loading feature matrix from {file_path}...")
    with h5py.File(file_path, "r") as hf:
        X_data = hf["X_data"][:]
        X_cols = [col.decode("utf-8") for col in hf["X_cols"][:]]
        X_index = hf["X_index"][:]
        X = pd.DataFrame(X_data, columns=X_cols, index=X_index)

        y_data = hf["y_data"][:]
        y_index = hf["y_index"][:]
        y_name = hf.attrs["y_name"]
        y = pd.Series(y_data, index=y_index, name=y_name)
    print(f"  [Cache] Load complete. X={X.shape}, y={y.shape}")
    return X, y


def split_pairs_no_overlap(pairs_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split train/test so no protein appears in both sets.
    Pairs containing proteins from different sets are dropped.
    """
    proteins = list(set(pairs_df["protein1"]).union(set(pairs_df["protein2"])))
    rng = np.random.RandomState(random_state)
    rng.shuffle(proteins)
    cutoff = max(1, int(len(proteins) * (1 - test_size)))
    train_proteins = set(proteins[:cutoff])
    test_proteins = set(proteins[cutoff:])

    train_idx, test_idx = [], []
    dropped = 0
    for idx, row in pairs_df.iterrows():
        p1, p2 = row["protein1"], row["protein2"]
        if p1 in train_proteins and p2 in train_proteins:
            train_idx.append(idx)
        elif p1 in test_proteins and p2 in test_proteins:
            test_idx.append(idx)
        else:
            dropped += 1
    return train_idx, test_idx, dropped


def make_protein_folds(pairs_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    """
    Create folds based on proteins (no overlap across folds).
    Pairs with proteins from different folds are dropped.
    """
    proteins = list(set(pairs_df["protein1"]).union(set(pairs_df["protein2"])))
    rng = np.random.RandomState(random_state)
    rng.shuffle(proteins)
    fold_assign = {prot: i % n_splits for i, prot in enumerate(proteins)}

    pair_fold: dict[int, int | None] = {}
    dropped = 0
    for idx, row in pairs_df.iterrows():
        f1 = fold_assign.get(row["protein1"])
        f2 = fold_assign.get(row["protein2"])
        if f1 is None or f2 is None or f1 != f2:
            pair_fold[idx] = None
            dropped += 1
        else:
            pair_fold[idx] = f1

    folds = []
    for fold_id in range(n_splits):
        val_idx = [idx for idx, f in pair_fold.items() if f == fold_id]
        train_idx = [idx for idx, f in pair_fold.items() if f is not None and f != fold_id]
        folds.append((train_idx, val_idx))

    return folds, dropped


def _get_or_compute_esm_global_only(seq_id: str, sequence: str, h5_path: str, embedding_computer: EmbeddingComputer):
    """Get or compute global ESM2 vector for a sequence."""
    seq_upper = sequence.upper()
    global_key = f"{seq_upper}_global_v2"

    with h5py.File(h5_path, "a") as h5f:
        if global_key in h5f:
            return h5f[global_key][:]

        _, global_vec = embedding_computer.compute_full_embeddings(sequence)
        h5f.create_dataset(global_key, data=global_vec)
        return global_vec


def build_esm_only_pair_matrix(fasta_path: str, pairs_path: str, h5_cache_path: str):
    """
    Build ESM2-only feature matrix using cached global vectors.
    """
    logger = PipelineLogger()
    logger.phase("Building ESM2-only feature matrix (READ CACHE ONLY)")

    sequences, pairs_df = load_data(fasta_path, pairs_path)
    pairs_df = canonicalize_pairs(pairs_df, dataset_name="ESM-only", logger=logger)
    logger.info(f"Số chuỗi protein: {len(sequences)}")
    logger.info(f"Số cặp tương tác: {len(pairs_df)}")

    if not os.path.exists(h5_cache_path):
        raise FileNotFoundError(
            f"H5 cache không tồn tại: {h5_cache_path}\n"
            "Bạn cần chạy feature extraction (ESM2) trước."
        )

    h5f = h5py.File(h5_cache_path, "r")
    global_vectors = {}

    for pid, seq in sequences.items():
        seq_upper = seq.upper()
        key = f"{seq_upper}_global_v2"

        if key not in h5f:
            raise KeyError(
                f"Global embedding của protein {pid} chưa có trong cache.\n"
                f"File cache: {h5_cache_path}"
            )

        global_vectors[pid] = h5f[key][:]

    h5f.close()

    X_rows, y_rows, idx_rows = [], [], []

    for idx, row in pairs_df.iterrows():
        p1, p2, label = row["protein1"], row["protein2"], row["label"]

        if p1 not in global_vectors or p2 not in global_vectors:
            continue

        v1 = global_vectors[p1]
        v2 = global_vectors[p2]
        X_rows.append(np.concatenate([v1, v2]))
        y_rows.append(label)
        idx_rows.append(idx)

    X_arr = np.vstack(X_rows)
    X_df = pd.DataFrame(X_arr, index=idx_rows, columns=[f"ESM_f{i}" for i in range(X_arr.shape[1])])
    y_s = pd.Series(y_rows, index=idx_rows, name="label")

    logger.result("ESM-only matrix shape", X_df.shape)
    return X_df, y_s


def get_protein_based_splits(pairs_df, n_splits=5, random_state=42):
    """
    Protein-level splits to avoid leakage: a protein appears in only one fold.
    """
    unique_proteins = list(set(pairs_df["protein1"]) | set(pairs_df["protein2"]))
    unique_proteins.sort()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    print(f"Generating {n_splits}-fold splits based on {len(unique_proteins)} unique proteins...")

    for fold_idx, (train_prot_idx, val_prot_idx) in enumerate(kf.split(unique_proteins)):
        train_prots = set(unique_proteins[i] for i in train_prot_idx)
        val_prots = set(unique_proteins[i] for i in val_prot_idx)

        train_mask = pairs_df.apply(
            lambda x: (x["protein1"] in train_prots) and (x["protein2"] in train_prots), axis=1
        )
        val_mask = pairs_df.apply(
            lambda x: (x["protein1"] in val_prots) and (x["protein2"] in val_prots), axis=1
        )

        train_indices = pairs_df[train_mask].index.to_numpy()
        val_indices = pairs_df[val_mask].index.to_numpy()
        splits.append((train_indices, val_indices))

        print(
            f"  Fold {fold_idx+1}: Train Pairs={len(train_indices)}, Val Pairs={len(val_indices)} "
            f"(Leakage Check: {len(set(train_indices) & set(val_indices))} overlap)"
        )

    return splits


def load_cluster_map(cluster_path: str) -> Dict[str, str]:
    """
    Load a protein->cluster mapping file.

    Accepts TSV/CSV with two columns: protein_id, cluster_id.
    Extra columns are ignored; duplicate proteins keep the first mapping.
    """
    if not os.path.exists(cluster_path):
        raise FileNotFoundError(f"Cluster map file not found: {cluster_path}")

    df = pd.read_csv(cluster_path, sep=None, engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Cluster map file must have at least 2 columns, found {df.shape[1]}")

    protein_col, cluster_col = 0, 1
    if {"protein", "cluster"}.issubset({c.lower() for c in df.columns.astype(str)}):
        protein_col = [c.lower() for c in df.columns].index("protein")
        cluster_col = [c.lower() for c in df.columns].index("cluster")

    mapping = {}
    for _, row in df.iterrows():
        protein_id = str(row[protein_col]).strip()
        cluster_id = str(row[cluster_col]).strip()
        if protein_id and protein_id not in mapping:
            mapping[protein_id] = cluster_id

    if not mapping:
        raise ValueError(f"No cluster mappings were parsed from {cluster_path}")

    return mapping


def get_cluster_based_splits(pairs_df: pd.DataFrame, cluster_map: Dict[str, str], n_splits: int = 5, random_state: int = 42):
    """
    Build splits ensuring proteins from the same sequence cluster never leak across folds.
    Unmapped proteins are treated as their own singleton clusters.
    """
    proteins = set(pairs_df["protein1"]).union(set(pairs_df["protein2"]))
    protein_to_cluster = {prot: cluster_map.get(prot, f"unmapped::{prot}") for prot in proteins}

    unique_clusters = sorted(set(protein_to_cluster.values()))
    if len(unique_clusters) < n_splits:
        raise ValueError(f"Not enough unique clusters ({len(unique_clusters)}) for {n_splits}-fold CV.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    for fold_idx, (train_cluster_idx, val_cluster_idx) in enumerate(kf.split(unique_clusters), start=1):
        train_clusters = {unique_clusters[i] for i in train_cluster_idx}
        val_clusters = {unique_clusters[i] for i in val_cluster_idx}

        train_mask = pairs_df.apply(
            lambda x: protein_to_cluster[x["protein1"]] in train_clusters
            and protein_to_cluster[x["protein2"]] in train_clusters,
            axis=1,
        )
        val_mask = pairs_df.apply(
            lambda x: protein_to_cluster[x["protein1"]] in val_clusters
            and protein_to_cluster[x["protein2"]] in val_clusters,
            axis=1,
        )

        train_indices = pairs_df[train_mask].index.to_numpy()
        val_indices = pairs_df[val_mask].index.to_numpy()
        splits.append((train_indices, val_indices))

        print(
            f"  Cluster Fold {fold_idx}: Train Pairs={len(train_indices)}, Val Pairs={len(val_indices)} "
            f"(clusters train={len(train_clusters)}, val={len(val_clusters)})"
        )

    return splits


__all__ = [
    "load_data",
    "canonicalize_pairs",
    "create_feature_matrix",
    "get_cache_filename",
    "save_feature_matrix_h5",
    "load_feature_matrix_h5",
    "split_pairs_no_overlap",
    "make_protein_folds",
    "build_esm_only_pair_matrix",
    "get_protein_based_splits",
    "get_cluster_based_splits",
    "load_cluster_map",
]
