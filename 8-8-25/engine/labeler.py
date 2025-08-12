from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from .vectorizer import make_window_features

def label_trades(
    minute_df: pd.DataFrame,
    trades: pd.DataFrame,
    pre_minutes: int,
    cfg: Dict[str, Any]
) -> Tuple[np.ndarray, List[list], List[int], List[str]]:
    """
    Returns:
      V:      N x 12 vector matrix for 15 minute windows
      labels: list of 3 code lists per trade
      kept_idx: indices in the original trades df that produced vectors
      keys_60m: banner_key string like D+|S+|Z per trade
    """
    # ensure datetime index with UTC
    if not isinstance(minute_df.index, pd.DatetimeIndex):
        minute_df = minute_df.set_index(pd.to_datetime(minute_df["time"], utc=True))
    else:
        minute_df = minute_df.tz_convert("UTC") if minute_df.index.tz is not None else minute_df.tz_localize("UTC")

    vecs: List[np.ndarray] = []
    labels_5m: List[list] = []
    kept_idx: List[int] = []
    keys_60m: List[str] = []

    for i, row in trades.iterrows():
        entry = pd.to_datetime(row["entry_time"], utc=True, errors="coerce")
        if pd.isna(entry):
            continue
        start = entry - pd.Timedelta(minutes=pre_minutes)
        end = entry - pd.Timedelta(minutes=1)
        win = minute_df.loc[start:end]
        if len(win) < pre_minutes:
            continue
        V, labs, key = make_window_features(win, cfg)
        vecs.append(V)
        labels_5m.append([lab.code for lab in labs])
        kept_idx.append(i)
        keys_60m.append(key)

    if not vecs:
        return np.empty((0,)), [], [], []
    return np.vstack(vecs), labels_5m, kept_idx, keys_60m
