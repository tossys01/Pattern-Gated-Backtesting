from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .reasoner import classify_block, BlockLabel

def window_scale(df: pd.DataFrame) -> float:
    tr = (df["high"] - df["low"]).abs()
    atr = tr.mean()
    return float(max(atr, 1e-9))

def make_window_features(win: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, List[BlockLabel], str]:
    """
    Build the 15 minute window features:
      - KDTree vector uses normalized features for similarity
      - Labels use point based thresholds
      - Also return a compact banner key like D+|S+|Z
    """
    L = len(win)
    if L % 5 != 0:
        raise ValueError("pre context length must be a multiple of 5")
    B = L // 5
    scale = window_scale(win)

    o = win["open"].to_numpy(); h = win["high"].to_numpy(); l = win["low"].to_numpy(); c = win["close"].to_numpy()

    # normalized for vectors
    o0 = o[0]
    o_n = (o - o0) / scale; h_n = (h - o0) / scale; l_n = (l - o0) / scale; c_n = (c - o0) / scale

    # point offsets for labels (no normalization)
    o_p = o - o[0]; h_p = h - o[0]; l_p = l - o[0]; c_p = c - o[0]

    vectors: List[float] = []
    labels: List[BlockLabel] = []
    codes: List[str] = []

    for b in range(B):
        s = b * 5; e = s + 5

        # 4 features per 5 minute block, normalized
        cb = c_n[s:e]; hb = h_n[s:e]; lb = l_n[s:e]; ob = o_n[s:e]
        delta_oc = float(cb[-1] - ob[0])
        high_exc = float(np.max(hb) - max(ob[0], cb[-1]))
        low_exc = float(min(ob[0], cb[-1]) - np.min(lb))
        slope = float(delta_oc / 5.0)
        vectors.extend([delta_oc, high_exc, low_exc, slope])

        # Labels on point based arrays
        cbp = c_p[s:e]; hbp = h_p[s:e]; lbp = l_p[s:e]
        lab = classify_block(cbp, hbp, lbp, cfg)
        labels.append(lab)
        codes.append(lab.code)

    banner_key = "|".join(codes)
    return np.array(vectors, dtype=float), labels, banner_key
