# engine/chart.py
from __future__ import annotations
from typing import Dict, Iterable, Tuple, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------
# Utility: SMA
# ---------------------------
def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

# ---------------------------
# Continuous-day x-axis via rangebreaks
# ---------------------------
def _rangebreaks_for_session(tz: str, session_start: str, session_end: str) -> list:
    # Remove weekends + hours outside session
    return [
        dict(bounds=["sat", "mon"]),  # hide weekends
        dict(bounds=[session_end, session_start], pattern="hour"),  # hide non-session hours
    ]

# ---------------------------
# Build reference times for a selected trade's pre-context
# ---------------------------
def build_ref_times_for_context(minutes: pd.DataFrame, entry_time: pd.Timestamp, pre_minutes: int) -> List[pd.Timestamp]:
    start = entry_time - pd.Timedelta(minutes=pre_minutes-1)
    end = entry_time
    # ensure they exist; if missing, we still build minute marks at UTC
    idx = pd.date_range(start=start, end=end, freq="1min", tz="UTC")
    return list(idx)

# ---------------------------
# Envelope statistics across matches
# returns per-offset arrays (length = pre_minutes)
# ---------------------------
def envelope_from_matches(minutes: pd.DataFrame, entry_times: Iterable[pd.Timestamp], pre_minutes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lows = []
    highs = []
    for et in entry_times:
        start = et - pd.Timedelta(minutes=pre_minutes-1)
        seg = minutes.loc[start: et]
        if len(seg) == pre_minutes:
            lows.append(seg["low"].to_numpy(dtype=float))
            highs.append(seg["high"].to_numpy(dtype=float))
    if not lows:
        offsets = np.arange(pre_minutes)
        return offsets, np.zeros(pre_minutes), np.zeros(pre_minutes)
    lows = np.vstack(lows)
    highs = np.vstack(highs)
    min_low = lows.min(axis=0)
    max_high = highs.max(axis=0)
    offsets = np.arange(pre_minutes)  # 0 .. pre-1 (oldest .. newest)
    return offsets, min_low, max_high

# ---------------------------
# Add envelope traces aligned to provided reference times
# ---------------------------
def add_envelope_traces(fig: go.Figure, ref_times: List[pd.Timestamp], min_low: np.ndarray, max_high: np.ndarray) -> go.Figure:
    if not ref_times or len(min_low) == 0 or len(max_high) == 0:
        return fig
    x = ref_times
    fig.add_trace(go.Scatter(
        x=x, y=max_high, mode="lines", line=dict(width=1), name="Envelope High", opacity=0.6,
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x, y=min_low, mode="lines", line=dict(width=1), name="Envelope Low", opacity=0.6,
        fill="tonexty", hoverinfo="skip", showlegend=False
    ))
    return fig

# ---------------------------
# Vertical symmetric padding like TV (little headroom/footroom)
# ---------------------------
def symmetric_yaxis(fig: go.Figure, series: pd.Series, pad_frac: float = 0.01) -> go.Figure:
    if series is None or series.empty:
        return fig
    lo = float(series.min()); hi = float(series.max())
    mid = 0.5 * (lo + hi)
    span = max(hi - mid, mid - lo)
    lo2, hi2 = mid - span * (1 + pad_frac), mid + span * (1 + pad_frac)
    fig.update_yaxes(range=[lo2, hi2])
    return fig

# ---------------------------
# Pre-context highlighting helpers
# ---------------------------
def _add_precontext_vrects(fig: go.Figure, entry_time: pd.Timestamp, pre_minutes: int):
    # Full pre-context block
    s_full = entry_time - pd.Timedelta(minutes=pre_minutes-1)
    e_full = entry_time
    fig.add_vrect(x0=s_full, x1=e_full, fillcolor="rgba(200,200,255,0.12)", line_width=0, layer="below")

    # 5-min sub-blocks
    blocks = pre_minutes // 5
    for b in range(blocks):
        b_start = s_full + pd.Timedelta(minutes=b*5)
        b_end   = b_start + pd.Timedelta(minutes=4)
        fig.add_vrect(x0=b_start, x1=b_end, fillcolor="rgba(255,255,255,0.05)" if b % 2 == 0 else "rgba(255,255,255,0.08)", line_width=0, layer="below")

# ---------------------------
# Main candle with trades overlay
# ---------------------------
def make_candles_with_trades(
    minutes: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    sl_pts: float,
    tp_pts: float,
    pre_minutes: int,
    show_context: bool,
    selected_trade_idx: int | None,
    view_after_minutes: int,
    tz: str,
    show_sma: Dict[str, bool] | None = None,
    continuous_days: bool = True,
    max_initial_hours: int = 36,
) -> go.Figure:
    show_sma = show_sma or {"ma5": False, "ma21": False, "ma50": False, "ma200": False}

    df = minutes.copy()
    df = df.sort_index()
    # Optional SMAs
    if show_sma.get("ma5"):   df["ma5"] = _sma(df["close"], 5)
    if show_sma.get("ma21"):  df["ma21"] = _sma(df["close"], 21)
    if show_sma.get("ma50"):  df["ma50"] = _sma(df["close"], 50)
    if show_sma.get("ma200"): df["ma200"] = _sma(df["close"], 200)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", showlegend=False
    ))

    # SMAs
    for k, col in [("SMA 5","ma5"),("SMA 21","ma21"),("SMA 50","ma50"),("SMA 200","ma200")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=k, line=dict(width=1)))

    # Pre-context highlight for selected trade
    ref_entry = None
    if trades is not None and len(trades) and selected_trade_idx is not None and 0 <= selected_trade_idx < len(trades):
        ref_entry = pd.to_datetime(trades.iloc[selected_trade_idx]["entry_time"], utc=True, errors="coerce")
        if show_context and not pd.isna(ref_entry):
            _add_precontext_vrects(fig, ref_entry, pre_minutes)

    # Draw SL/TP bands for selected trade
    if ref_entry is not None and 0 <= selected_trade_idx < len(trades):
        row = trades.iloc[selected_trade_idx]
        side = str(row.get("side", "")).upper()
        entry_price = float(row.get("entry_price", np.nan))
        if not np.isnan(entry_price):
            if side == "LONG":
                tp = entry_price + tp_pts
                sl = entry_price - sl_pts
            else:
                tp = entry_price - tp_pts
                sl = entry_price + sl_pts
            # Horizontal lines
            for y, nm, col in [(tp, "TP","rgba(0,200,0,0.5)"), (entry_price,"Entry","rgba(255,255,255,0.5)"), (sl,"SL","rgba(200,0,0,0.5)")]:
                fig.add_hline(y=y, line_width=1, line_color=col, opacity=0.8)

    # Layout — remove time gaps; initial zoom = last 36h; TV-like padding
    session_start = "08:30"
    session_end   = "15:30"
    rbs = _rangebreaks_for_session(tz, session_start, session_end) if continuous_days else []

    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(
            rangeselector=dict(visible=False),
            rangeslider=dict(visible=False),
            showgrid=False,
            rangebreaks=rbs,
        ),
        yaxis=dict(showgrid=False, automargin=True),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        font=dict(color="#FFFFFF"),
        hovermode="x unified",
    )

    # Default view: last max_initial_hours (or full slice if shorter)
    if len(df.index) > 0 and max_initial_hours:
        end_ts = df.index[-1]
        start_ts = max(df.index[0], end_ts - pd.Timedelta(hours=max_initial_hours))
        fig.update_xaxes(range=[start_ts, end_ts])

    return fig
