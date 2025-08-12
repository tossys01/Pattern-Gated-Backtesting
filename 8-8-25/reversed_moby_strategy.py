#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, smooth_d: int = 3):
    lowest_low = low.rolling(length, min_periods=length).min()
    highest_high = high.rolling(length, min_periods=length).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return k, d

def resample_sma(close: pd.Series, rule: str, length: int) -> pd.Series:
    resampled = close.resample(rule).last()
    ht_sma = resampled.rolling(length, min_periods=length).mean()
    aligned = ht_sma.reindex(resampled.index)
    return aligned.reindex(close.index, method="ffill")

def ensure_tz(dt_index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if dt_index.tz is None:
        return dt_index.tz_localize("UTC").tz_convert(tz)
    return dt_index.tz_convert(tz)

@dataclass
class Config:
    consolidation_threshold: float = 0.005
    stop_loss_points: float = 10.0
    take_profit_points: float = 10.0
    start_hour: int = 8
    start_minute: int = 30
    end_hour: int = 15
    end_minute: int = 30
    tz: str = "US/Eastern"

@dataclass
class Columns:
    time: str = "time"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: Optional[str] = None

def run_strategy(df: pd.DataFrame, cfg: Config, cols: Columns) -> Dict[str, pd.DataFrame]:
    if cols.time not in df.columns:
        raise ValueError(f"Missing timestamp column: {cols.time}")
    df[cols.time] = pd.to_datetime(df[cols.time], utc=True)
    df.set_index(cols.time, inplace=True)
    df.index = ensure_tz(df.index, cfg.tz)

    o = df[cols.open].astype(float)
    h = df[cols.high].astype(float)
    l = df[cols.low].astype(float)
    c = df[cols.close].astype(float)

    df["ma5"] = sma(c, 5)
    df["ma21"] = sma(c, 21)
    df["ma50"] = sma(c, 50)
    df["ma200"] = sma(c, 200)
    df["ma200_1h"] = resample_sma(c, "60T", 200)
    df["ma200_daily"] = resample_sma(c, "1D", 200)
    df["k"], df["d"] = stochastic_oscillator(h, l, c, 14, 3)

    def ma_dist(a, b): return (a - b).abs() / c

    thresh = cfg.consolidation_threshold
    df["consolidation"] = (
        (ma_dist(df["ma5"], df["ma21"]) < thresh)
        & (ma_dist(df["ma5"], df["ma50"]) < thresh)
        & (ma_dist(df["ma5"], df["ma200"]) < thresh)
        & (ma_dist(df["ma21"], df["ma50"]) < thresh)
        & (ma_dist(df["ma21"], df["ma200"]) < thresh)
        & (ma_dist(df["ma50"], df["ma200"]) < thresh)
    )

    trade_signal = pd.Series(np.nan, index=df.index, dtype=float)
    cond_long = (df["k"] < df["d"]) & (df["k"] > 80)
    cond_short = (df["k"] > df["d"]) & (df["k"] < 20)
    for i in range(len(df)):
        if df["consolidation"].iloc[i]:
            if cond_long.iloc[i]:
                trade_signal.iloc[i] = 1.0
            elif cond_short.iloc[i]:
                trade_signal.iloc[i] = -1.0
    trade_signal = trade_signal.ffill().fillna(1.0)
    df["trade_signal"] = trade_signal
    df["prev_trade_signal"] = df["trade_signal"].shift(1).fillna(df["trade_signal"])

    idx = df.index
    day_base = idx.normalize()
    start_series = day_base + pd.to_timedelta(cfg.start_hour, unit="h") + pd.to_timedelta(cfg.start_minute, unit="m")
    end_series   = day_base + pd.to_timedelta(cfg.end_hour,   unit="h") + pd.to_timedelta(cfg.end_minute, unit="m")
    df["start_time"] = pd.Series(start_series, index=df.index)
    df["end_time"]   = pd.Series(end_series,   index=df.index)
    df["within_time"] = (df.index >= df["start_time"]) & (df.index <= df["end_time"])
    df["market_close_bar"] = df.index >= df["end_time"]

    position = 0
    last_entry_price = np.nan
    entry_time = None
    trades: List[Dict] = []

    for i in range(len(df)):
        ts = df.index[i]
        bar_high = h.iloc[i]
        bar_low  = l.iloc[i]
        price_close = c.iloc[i]

        if not df["within_time"].iloc[i]:
            if df["market_close_bar"].iloc[i] and position != 0:
                trades.append({
                    "entry_time": entry_time, "exit_time": ts, "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": last_entry_price, "exit_price": price_close,
                    "reason": "Market close",
                    "pnl_points": (price_close - last_entry_price) * position
                })
                position = 0
                last_entry_price = np.nan
                entry_time = None
            continue

        if position != 0 and not np.isnan(last_entry_price):
            if position == 1:
                stop_hit = bar_low <= last_entry_price - cfg.stop_loss_points
                tp_hit   = bar_high >= last_entry_price + cfg.take_profit_points
            else:
                stop_hit = bar_high >= last_entry_price + cfg.stop_loss_points
                tp_hit   = bar_low  <= last_entry_price - cfg.take_profit_points

            if stop_hit or tp_hit:
                reason = "Stop loss" if stop_hit else "Take profit"
                exit_price = (
                    last_entry_price - cfg.stop_loss_points if stop_hit and position == 1 else
                    last_entry_price + cfg.take_profit_points if tp_hit and position == 1 else
                    last_entry_price + cfg.stop_loss_points if stop_hit and position == -1 else
                    last_entry_price - cfg.take_profit_points
                )
                trades.append({
                    "entry_time": entry_time, "exit_time": ts, "side": "LONG" if position == 1 else "SHORT",
                    "entry_price": last_entry_price, "exit_price": exit_price,
                    "reason": reason, "pnl_points": (exit_price - last_entry_price) * position
                })
                position = 0
                last_entry_price = np.nan
                entry_time = None
                continue

        prev_sig = df["prev_trade_signal"].iloc[i]
        curr_sig = df["trade_signal"].iloc[i]
        if curr_sig == 1.0 and prev_sig != 1.0 and position == 0:
            position = 1; last_entry_price = price_close; entry_time = ts
        elif curr_sig == -1.0 and prev_sig != -1.0 and position == 0:
            position = -1; last_entry_price = price_close; entry_time = ts

    if position != 0 and entry_time is not None:
        ts = df.index[-1]; price_close = c.iloc[-1]
        trades.append({
            "entry_time": entry_time, "exit_time": ts, "side": "LONG" if position == 1 else "SHORT",
            "entry_price": last_entry_price, "exit_price": price_close,
            "reason": "Dataset end", "pnl_points": (price_close - last_entry_price) * position
        })

    trades_df = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    if not trades_df.empty:
        trades_df["cum_pnl_points"] = trades_df["pnl_points"].cumsum()
        equity = trades_df[["exit_time","cum_pnl_points"]].rename(columns={"exit_time":"time"}).set_index("time")
    else:
        equity = pd.DataFrame(columns=["cum_pnl_points"])

    return {"df": df, "trades": trades_df, "equity": equity}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--timestamp_col", default="time")
    p.add_argument("--open_col", default="open")
    p.add_argument("--high_col", default="high")
    p.add_argument("--low_col", default="low")
    p.add_argument("--close_col", default="close")
    p.add_argument("--tz", default="US/Eastern")
    p.add_argument("--cons_threshold", type=float, default=0.005)
    p.add_argument("--sl", type=float, default=10.0)
    p.add_argument("--tp", type=float, default=10.0)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(consolidation_threshold=args.cons_threshold, stop_loss_points=args.sl, take_profit_points=args.tp, tz=args.tz)
    cols = Columns(time=args.timestamp_col, open=args.open_col, high=args.high_col, low=args.low_col, close=args.close_col)
    df = pd.read_csv(args.csv)
    res = run_strategy(df, cfg, cols)
    res["trades"].to_csv("trades_reversed_moby.csv", index=False)
    res["equity"].to_csv("equity_reversed_moby.csv")

if __name__ == "__main__":
    main()
