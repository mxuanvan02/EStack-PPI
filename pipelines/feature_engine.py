import io
import re
from collections import Counter
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import requests
import torch
from filelock import FileLock
from tqdm.std import tqdm
from transformers import AutoModel, AutoTokenizer

# Constants
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

CTD_GROUPS = {
    "hydrophobicity": {"polar": "RKEDQN", "neutral": "GASTPHY", "hydrophobic": "CVLIMFW"},
    "normalized_vdw": {"small": "GASTPDC", "medium": "NVEQIL", "large": "MHKFRYW"},
    "polarity": {"neutral": "LIFWCMVY", "polar": "PATGS", "charged": "HQRKNED"},
    "polarizability": {"low": "GASDT", "medium": "CPNVEQIL", "high": "KMHFRYW"},
    "charge": {"positive": "KR", "neutral": "ANCQGHILMFPSTWYV", "negative": "DE"},
    "secondary_structure": {"helix": "EALMQKRH", "strand": "VIYW", "coil": "GNPSD"},
    "solvent_accessibility": {"buried": "ALFCGIVW", "exposed": "MSPTHY", "intermediate": "NQRKDE"},
}

PHYSICO_INDICES = {
    "hydrophobicity": {
        "A": 0.62,
        "C": 0.29,
        "D": -0.90,
        "E": -0.74,
        "F": 1.19,
        "G": 0.48,
        "H": -0.40,
        "I": 1.38,
        "K": -1.50,
        "L": 1.06,
        "M": 0.64,
        "N": -0.78,
        "P": 0.12,
        "Q": -0.85,
        "R": -2.53,
        "S": -0.18,
        "T": -0.05,
        "V": 1.08,
        "W": 0.81,
        "Y": 0.26,
    },
    "hydrophilicity": {
        "A": -0.5,
        "C": -1.0,
        "D": 3.0,
        "E": 3.0,
        "F": -2.5,
        "G": 0.0,
        "H": -0.5,
        "I": -1.8,
        "K": 3.0,
        "L": -1.8,
        "M": -1.3,
        "N": 0.2,
        "P": 0.0,
        "Q": 0.2,
        "R": 3.0,
        "S": 0.3,
        "T": -0.4,
        "V": -1.5,
        "W": -3.4,
        "Y": -2.3,
    },
}


class EmbeddingComputer:
    """Load ESM2 and provide embedding computation helpers."""

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        print(f"Loading protein language model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        print(f"ESM2 model loaded successfully on {self.device.upper()}.")

    @torch.no_grad()
    def compute_full_embeddings(self, sequence: str) -> tuple[np.ndarray, np.ndarray]:
        """Return full residue matrix and global vector."""
        inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1022).to(self.device)
        outputs = self.model(**inputs)

        matrix = outputs.last_hidden_state[0, 1:-1]

        if matrix.shape[0] == 0:
            return (
                np.zeros((0, self.embedding_dim), dtype=np.float32),
                np.zeros(self.embedding_dim, dtype=np.float32),
            )

        global_vec = matrix.mean(dim=0).cpu().numpy()
        return matrix.cpu().numpy(), global_vec


class BaseFeatureExtractor:
    def __init__(self):
        self.feature_names = []

    def compute(self, sequence: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def get_names(self, prefix: str = "") -> List[str]:
        return [f"{prefix}_{name}" for name in self.feature_names] if prefix else self.feature_names


class AACExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.feature_names = [f"AAC_{aa}" for aa in AMINO_ACIDS]

    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        length = len(seq) or 1
        counts = Counter(seq)
        return np.array([counts.get(aa, 0) / length for aa in AMINO_ACIDS], dtype=np.float32)


class DPCExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.dipeptides = [aa1 + aa2 for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS]
        self.feature_names = [f"DPC_{dp}" for dp in self.dipeptides]

    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        length = len(seq) - 1 or 1
        counts = Counter(seq[j : j + 2] for j in range(len(seq) - 1))
        return np.array([counts.get(dp, 0) / length for dp in self.dipeptides], dtype=np.float32)


class CTDExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.prop_mappers = {}
        for prop, groups in CTD_GROUPS.items():
            mapper: Dict[str, int] = {}
            for i, aas in enumerate(groups.values()):
                for aa in aas:
                    mapper[aa] = i
            self.prop_mappers[prop] = mapper

        for prop in CTD_GROUPS.keys():
            self.feature_names.extend([f"CTD_{prop}_C_{i+1}" for i in range(3)])
            self.feature_names.extend([f"CTD_{prop}_T_{t}" for t in ["12", "13", "23"]])
            for i in range(3):
                self.feature_names.extend([f"CTD_{prop}_D_{i+1}_{q}" for q in [0, 25, 50, 75, 100]])

    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        features = []
        seq_len = len(seq) or 1

        for prop, mapper in self.prop_mappers.items():
            seq_groups = [mapper.get(aa, -1) for aa in seq]

            counts = Counter(seq_groups)
            features.extend([counts.get(i, 0) / seq_len for i in range(3)])

            trans = {"12": 0, "13": 0, "23": 0}
            for i in range(seq_len - 1):
                g1, g2 = seq_groups[i], seq_groups[i + 1]
                if g1 != -1 and g2 != -1 and g1 != g2:
                    pair = tuple(sorted((g1, g2)))
                    if pair == (0, 1):
                        trans["12"] += 1
                    elif pair == (0, 2):
                        trans["13"] += 1
                    elif pair == (1, 2):
                        trans["23"] += 1
            features.extend([v / (seq_len - 1 or 1) for v in trans.values()])

            for gid in range(3):
                positions = [i for i, g in enumerate(seq_groups) if g == gid]
                if positions:
                    quartiles = np.percentile(positions, [0, 25, 50, 75, 100]) / (seq_len - 1 or 1)
                    features.extend(quartiles)
                else:
                    features.extend([0.0] * 5)
        return np.array(features, dtype=np.float32)


class PAACExtractor(BaseFeatureExtractor):
    def __init__(self, lambda_val: int = 10, weight: float = 0.05, aac_extractor: AACExtractor | None = None):
        super().__init__()
        self.lambda_val = lambda_val
        self.weight = weight
        self.aac_extractor = aac_extractor if aac_extractor else AACExtractor()
        self.feature_names = [f"PAAC_AAC_{aa}" for aa in AMINO_ACIDS]
        self.feature_names.extend([f"PAAC_lambda_{i+1}" for i in range(lambda_val)])

    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        seq_len = len(seq) or 1

        aac = self.aac_extractor.compute(seq)

        hydro = [PHYSICO_INDICES["hydrophobicity"].get(aa, 0) for aa in seq]
        hydrophil = [PHYSICO_INDICES["hydrophilicity"].get(aa, 0) for aa in seq]

        theta = []
        for lag in range(1, self.lambda_val + 1):
            if seq_len > lag:
                corr = sum(
                    (hydro[i] - hydro[i + lag]) ** 2 + (hydrophil[i] - hydrophil[i + lag]) ** 2
                    for i in range(seq_len - lag)
                )
                theta.append(corr / (seq_len - lag))
            else:
                theta.append(0.0)

        theta_sum = sum(theta)
        denominator = 1 + self.weight * theta_sum

        pseaac_part1 = aac / denominator
        pseaac_part2 = (self.weight * np.array(theta)) / denominator

        return np.concatenate([pseaac_part1, pseaac_part2]).astype(np.float32)


class MoranAutocorrelation(BaseFeatureExtractor):
    def __init__(self, max_lag: int = 30):
        super().__init__()
        self.max_lag = max_lag
        self.properties = list(PHYSICO_INDICES.keys())
        self.feature_names = [f"Moran_{p}_lag{l}" for p in self.properties for l in range(1, max_lag + 1)]

    def compute(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        seq_len = len(seq)
        all_features = []

        for prop_name in self.properties:
            prop_values = PHYSICO_INDICES[prop_name]
            prop_seq = np.array([prop_values.get(aa, 0) for aa in seq], dtype=np.float32)

            if seq_len < 2:
                all_features.extend([0.0] * self.max_lag)
                continue

            mean_prop = np.mean(prop_seq)
            std_dev = np.std(prop_seq) + 1e-9
            norm_prop_seq = (prop_seq - mean_prop) / std_dev

            prop_features = []
            for lag in range(1, self.max_lag + 1):
                if seq_len > lag:
                    numerator = np.sum(norm_prop_seq[:-lag] * norm_prop_seq[lag:])
                    moran = numerator / (seq_len - lag)
                    prop_features.append(moran)
                else:
                    prop_features.append(0.0)
            all_features.extend(prop_features)

        return np.array(all_features, dtype=np.float32)


class InterpretableFeatureExtractor:
    """Aggregate handcrafted feature extractors."""

    def __init__(self, use_aac=True, use_dpc=True, use_ctd=True, use_paac=True, use_moran=True):
        self.extractors: Dict[str, BaseFeatureExtractor] = {}
        aac_ext = AACExtractor() if use_aac or use_paac else None

        if use_aac:
            self.extractors["AAC"] = aac_ext
        if use_dpc:
            self.extractors["DPC"] = DPCExtractor()
        if use_ctd:
            self.extractors["CTD"] = CTDExtractor()
        if use_paac:
            self.extractors["PAAC"] = PAACExtractor(aac_extractor=aac_ext)
        if use_moran:
            self.extractors["Moran"] = MoranAutocorrelation()

    def extract(self, sequence: str) -> np.ndarray:
        features = [ext.compute(sequence) for ext in self.extractors.values()]
        return np.concatenate(features) if features else np.array([], dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        names: List[str] = []
        for name, ext in self.extractors.items():
            names.extend(ext.get_names(prefix=name))
        return names

    def summary(self) -> Dict[str, int]:
        return {name: len(ext.feature_names) for name, ext in self.extractors.items()}


class FeatureEngine:
    """
    Combine handcrafted, motif, and ESM2 embeddings (global + motif-local).
    """

    ELM_MOTIFS_URL = "http://elm.eu.org/elms/elms_index.tsv"

    def __init__(self, h5_cache_path: str, embedding_computer: EmbeddingComputer):
        print("Initializing Sequence-Only Hybrid Feature Engine...")

        self.handcraft_extractor = InterpretableFeatureExtractor()
        self.embedding_cache_path = h5_cache_path
        self.embedding_computer = embedding_computer
        self.embedding_dim = self.embedding_computer.embedding_dim

        self.motifs: Dict[str, re.Pattern] = {}
        self.motif_names: List[str] = []
        self._load_elm_motifs_from_api()

        self.global_emb_names = [f"Global_ESM_{i}" for i in range(self.embedding_dim)]
        self.local_emb_names = [f"Local_Motif_ESM_{i}" for i in range(self.embedding_dim)]

        print("Feature Engine ready.")

    def _load_elm_motifs_from_api(self):
        print(f"Fetching and compiling ELM motifs from API: {self.ELM_MOTIFS_URL}...")
        try:
            response = requests.get(self.ELM_MOTIFS_URL)
            response.raise_for_status()
            text_stream = io.StringIO(response.text)
            df = pd.read_csv(text_stream, sep="\t", comment="#")

            df = df.dropna(subset=["Regex"])
            for _, row in df.iterrows():
                elm_id = row["ELMIdentifier"]
                regex_pattern = row["Regex"]
                try:
                    self.motifs[elm_id] = re.compile(regex_pattern)
                except re.error as exc:
                    print(f"Warning: Could not compile regex for {elm_id}: '{regex_pattern}'. Error: {exc}")

            self.motif_names = [f"Motif_{name}" for name in self.motifs.keys()]
            print(f"✅ Successfully loaded and compiled {len(self.motifs)} motifs from ELM database.")

        except requests.exceptions.RequestException as exc:
            print(f"❌ ERROR: Failed to fetch ELM motifs from API. Error: {exc}. Motif features will be empty.")
        except Exception as exc:
            print(
                f"❌ ERROR: An unexpected error occurred while processing ELM motifs. Error: {exc}. "
                "Motif features will be empty."
            )

    def _get_or_compute_embeddings(self, seq_id: str, sequence: str, h5f: h5py.File):
        seq_upper = sequence.upper()
        matrix_key = f"{seq_upper}_matrix_v2"
        global_key = f"{seq_upper}_global_v2"

        if matrix_key in h5f and global_key in h5f:
            return h5f[matrix_key][:], h5f[global_key][:]

        if self.embedding_computer:
            matrix, global_vec = self.embedding_computer.compute_full_embeddings(sequence)
            h5f.create_dataset(matrix_key, data=matrix)
            h5f.create_dataset(global_key, data=global_vec)
            return matrix, global_vec

        length = len(sequence)
        dim = self.embedding_dim
        return np.zeros((length, dim), dtype=np.float32), np.zeros(dim, dtype=np.float32)

    def _extract_motif_and_local_embedding(self, sequence: str, embedding_matrix: np.ndarray):
        motif_binary_vector = []
        local_embedding_vectors = []

        if embedding_matrix.shape[0] == 0:
            return np.array([0] * len(self.motifs), dtype=np.float32), np.zeros(self.embedding_dim, dtype=np.float32)

        for pattern in self.motifs.values():
            match = pattern.search(sequence)
            if match:
                motif_binary_vector.append(1)
                start, end = match.span()
                start = min(start, embedding_matrix.shape[0] - 1)
                end = min(end, embedding_matrix.shape[0])
                if start < end:
                    motif_embs = embedding_matrix[start:end]
                    if motif_embs.shape[0] > 0:
                        local_embedding_vectors.append(motif_embs.max(axis=0))
            else:
                motif_binary_vector.append(0)

        if local_embedding_vectors:
            final_local_embedding = np.max(local_embedding_vectors, axis=0)
        else:
            final_local_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        return np.array(motif_binary_vector, dtype=np.float32), final_local_embedding

    def extract_all_features(self, sequences_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Extract handcraft, motif, global, and local embeddings for every sequence."""
        all_features = {}
        lock_path = self.embedding_cache_path + ".lock"

        with FileLock(lock_path):
            with h5py.File(self.embedding_cache_path, "a") as h5f:
                pbar = tqdm(sequences_dict.items(), desc="⚙️ Extracting Features", unit="seq")

                for seq_id, sequence in pbar:
                    pbar.set_postfix_str(f"{seq_id}: Handcraft...")
                    handcraft_feats = self.handcraft_extractor.extract(sequence)

                    pbar.set_postfix_str(f"{seq_id}: Embeddings...")
                    embedding_matrix, global_embedding = self._get_or_compute_embeddings(seq_id, sequence, h5f)

                    pbar.set_postfix_str(f"{seq_id}: Localizing Motifs...")
                    motif_binary_vector, local_motif_embedding = self._extract_motif_and_local_embedding(
                        sequence, embedding_matrix
                    )

                    combined_vector = np.concatenate(
                        [handcraft_feats, motif_binary_vector, global_embedding, local_motif_embedding]
                    )
                    all_features[seq_id] = combined_vector

                    pbar.set_postfix_str(f"{seq_id}: Done.")

        return all_features

    def get_feature_names(self) -> List[str]:
        handcraft_names = self.handcraft_extractor.get_feature_names()

        all_names = handcraft_names + self.motif_names + self.global_emb_names + self.local_emb_names
        return all_names
