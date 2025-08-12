# engine/pnl.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, Dict
import pandas as pd
import numpy as np

def ensure_money_col(trades: pd.DataFrame, dollars_per_point: float = 2.0) -> pd.DataFrame:
    """Guarantee pnl_money exists and is numeric."""
    out = trades.copy()
    if "pnl_points" not in out.columns:
        raise ValueError("trades must include pnl_points")
    out["pnl_points"] = pd.to_numeric(out["pnl_points"], errors="coerce").fillna(0.0)
    out["pnl_money"] = out["pnl_points"] * float(dollars_per_point)
    return out

def summary(trades: pd.DataFrame, dollars_per_point: float = 2.0) -> Dict[str, float]:
    """One line summary for a given slice of trades."""
    t = ensure_money_col(trades, dollars_per_point)
    n = len(t)
    if n == 0:
        return dict(n=0, wins=0, losses=0, win_rate=0.0, pts=0.0, usd=0.0, avg_pts=0.0, avg_usd=0.0,
                    max_win_pts=0.0, max_loss_pts=0.0)
    wins = int((t["pnl_points"] > 0).sum())
    losses = int((t["pnl_points"] < 0).sum())
    pts = float(t["pnl_points"].sum())
    usd = float(t["pnl_money"].sum())
    return dict(
        n=n,
        wins=wins,
        losses=losses,
        win_rate=(wins / n) * 100.0,
        pts=pts,
        usd=usd,
        avg_pts=float(t["pnl_points"].mean()),
        avg_usd=float(t["pnl_money"].mean()),
        max_win_pts=float(t["pnl_points"].max()),
        max_loss_pts=float(t["pnl_points"].min()),
    )

def groupby_summary(trades: pd.DataFrame, by: str, dollars_per_point: float = 2.0) -> pd.DataFrame:
    """Summaries for each group value, sorted by usd desc."""
    t = ensure_money_col(trades, dollars_per_point)
    if by not in t.columns:
        raise ValueError(f"column {by} missing for groupby")
    g = t.groupby(by, dropna=False)
    out = g.apply(lambda df: pd.Series(summary(df, dollars_per_point)))
    out = out.sort_values("usd", ascending=False)
    return out.reset_index()

def slice_by_indices(trades: pd.DataFrame, indices: Iterable[int]) -> pd.DataFrame:
    """Return a stable slice by integer indices from the original trades order."""
    idx = list(indices)
    valid = [i for i in idx if 0 <= i < len(trades)]
    return trades.iloc[valid].copy()
