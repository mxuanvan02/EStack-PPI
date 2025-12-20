import pandas as pd
import numpy as np

def make_pair_features(pairs_df: pd.DataFrame,
                       node_feats: pd.DataFrame,
                       ops=("concat", "absdiff", "hadamard")) -> pd.DataFrame:
    """
    pairs_df: có cột ['protein1','protein2'] (+ 'label' nếu có)
    node_feats: DataFrame index=protein_id, cột là các feature đã đặt tên (ví dụ motif__*)
    ops: chọn cách tạo đặc trưng cặp:
         - 'concat': ghép [f*_1, f*_2]
         - 'absdiff': |f*_1 - f*_2|
         - 'hadamard': f*_1 * f*_2
    """
    assert {"protein1","protein2"}.issubset(pairs_df.columns)
    F1 = node_feats.add_suffix("_1")
    F2 = node_feats.add_suffix("_2")

    df = pairs_df.copy()
    df = df.merge(F1, left_on="protein1", right_index=True, how="left")
    df = df.merge(F2, left_on="protein2", right_index=True, how="left")

    feat_cols_1 = [c for c in df.columns if c.endswith("_1")]
    feat_cols_2 = [c for c in df.columns if c.endswith("_2")]
    # đảm bảo cùng thứ tự
    feat_base = [c[:-2] for c in feat_cols_1]
    feat_cols_2 = [f"{b}_2" for b in feat_base]

    out_parts = []

    if "concat" in ops:
        out_parts.append(df[feat_cols_1 + feat_cols_2])

    if "absdiff" in ops:
        absdiff = (df[feat_cols_1].values - df[feat_cols_2].values)
        absdiff = np.abs(absdiff)
        absdiff = pd.DataFrame(absdiff, columns=[f"{b}__absdiff" for b in feat_base], index=df.index)
        out_parts.append(absdiff)

    if "hadamard" in ops:
        had = df[feat_cols_1].values * df[feat_cols_2].values
        had = pd.DataFrame(had, columns=[f"{b}__had" for b in feat_base], index=df.index)
        out_parts.append(had)

    X = pd.concat(out_parts, axis=1)
    keep_cols = [c for c in df.columns if c in ("protein1","protein2","label")]
    return pd.concat([df[keep_cols], X], axis=1)

