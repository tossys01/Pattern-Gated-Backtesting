# ui/app.py — Streamlit only (with KDTree similarity; no "Compare patterns")

# ---------- bootstrap project root for `engine` imports ----------
import sys
from pathlib import Path
_UI = Path(__file__).resolve()
ROOT = _UI.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENGINE = ROOT / "engine"
ENGINE.mkdir(exist_ok=True)
(ENGINE / "__init__.py").touch()

# ---------- std libs ----------
import json
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- project deps ----------
from engine.vectorizer import make_window_features  # uses engine.reasoner.classify_block
from engine.kdtree_index import VectorIndex

# ---------- config ----------
CFG = json.load(open(ROOT / "config.json"))
TZ = CFG.get("timezone", "US/Eastern")
DEFAULT_CSV = ROOT / CFG["data"]["minute_csv"]
LABEL_MAP = json.load(open(ROOT / CFG["models"]["label_map"]))
MONEY_PER_POINT = float(CFG.get("money_per_point", 2.0))

# Pattern colors (keep chart visuals as you liked)
COLORS = {
    "D+": "#00ff66",   # bright green
    "S+": "#00aa44",   # green
    "Z":  "#ffd21f",   # yellow
    "S-": "#ff7a7a",   # light red
    "D-": "#cc0000",   # dark red
}
# Table banner
EMOJI = {"D+": "🟩", "S+": "🟢", "Z": "🟨", "S-": "🟠", "D-": "🟥", "R+": "R+", "R-": "R-"}

# ---------- utilities ----------
@st.cache_data
def load_minutes(csv_path: str, session_start: str, session_end: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ("time", "open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"CSV missing column '{c}'")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # keep only regular session by local time
    local = df["time"].dt.tz_convert(TZ)
    st_time = pd.to_datetime(session_start).time()
    en_time = pd.to_datetime(session_end).time()
    lt = local.dt.time
    mask = (lt >= st_time) & (lt <= en_time)
    return df.loc[mask, ["time", "open", "high", "low", "close"]].copy()


def slice_by_days(df: pd.DataFrame, tz: str, start_d: date | None, end_d: date | None) -> pd.DataFrame:
    if df.empty: return df
    local = df["time"].dt.tz_convert(tz)
    days = local.dt.date
    s = start_d or days.iloc[0]
    e = end_d or days.iloc[-1]
    if e < s: e = s
    keep = (days >= s) & (days <= e)
    return df.loc[keep].copy()


def to_continuous_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bar"] = np.arange(len(out), dtype=int)
    return out


def y_range_tight(highs: np.ndarray, lows: np.ndarray, pad_ratio: float = 0.01):
    lo = float(np.min(lows)); hi = float(np.max(highs))
    rng = max(1e-9, hi - lo); pad = rng * pad_ratio
    return lo - pad, hi + pad


def attach_patterns(minutes: pd.DataFrame, trades: pd.DataFrame, pre_minutes: int):
    """
    Build vectors + 5-min labels per trade (uses engine.vectorizer + LABEL_MAP thresholds).
    Adds: trade_id, Pattern, PatternViz, PatternList
    Returns: updated_trades, V (NxF), labels (list[list]), kept_idx (list of trade indices kept)
    """
    idx_collected, vecs, all_codes = [], [], []
    m = minutes.set_index("time")

    for i, row in trades.iterrows():
        entry = pd.to_datetime(row.get("entry_time"), utc=True, errors="coerce")
        if pd.isna(entry):
            continue
        win = m.loc[entry - pd.Timedelta(minutes=pre_minutes): entry - pd.Timedelta(minutes=1)]
        if len(win) < pre_minutes:
            continue

        w = win[["open", "high", "low", "close"]].copy()
        res = make_window_features(w, LABEL_MAP)
        if not isinstance(res, tuple) or len(res) < 2:
            raise ValueError("make_window_features() returned unexpected shape")
        v, labs = res[0], res[1]
        vecs.append(v)

        # robust code extraction
        codes = []
        for lab in labs:
            if hasattr(lab, "code"):
                codes.append(lab.code)
            elif isinstance(lab, dict) and "code" in lab:
                codes.append(lab["code"])
            else:
                codes.append(str(lab))
        # eliminate UNK -> treat as Z so your table never shows ⬜
        codes = [("Z" if c in ("UNK", "None", "nan") else c) for c in codes]

        all_codes.append(codes)
        idx_collected.append(int(i))

    T = trades.copy().reset_index(drop=True)
    for col in ("trade_id", "Pattern", "PatternViz", "PatternList"):
        if col not in T.columns:
            T[col] = None
    T["trade_id"] = T.index

    def _codes_to_emoji(codes_):
        # Reversals are “R+ / R-” text, others colored squares
        out = []
        for c in codes_:
            out.append(EMOJI.get(c, "🟨" if c == "Z" else "⬜"))
        return " ".join(out)

    for j, tidx in enumerate(idx_collected):
        codes = all_codes[j]
        T.at[tidx, "Pattern"] = "|".join(codes)
        T.at[tidx, "PatternViz"] = _codes_to_emoji(codes)
        T.at[tidx, "PatternList"] = codes

    V = np.vstack(vecs) if vecs else np.empty((0,))
    return T, V, all_codes, idx_collected


def add_pre_context_blocks(fig: go.Figure, entry_bar: int, pre_minutes: int, codes: list[str] | None):
    """Draw 5-min blocks behind focused trade."""
    if entry_bar is None: return
    total = pre_minutes
    if total <= 0: return
    blocks = total // 5
    start_bar = entry_bar - total
    if start_bar < 0: start_bar = 0

    for b in range(blocks):
        x0 = start_bar + b * 5
        x1 = x0 + 4
        code = codes[b] if codes and b < len(codes) else None
        if code in ("R+", "R-"):
            # outline & text
            fig.add_vrect(
                x0=x0, x1=x1,
                line=dict(color="#AAAAAA", width=1, dash="dot"),
                fillcolor="rgba(0,0,0,0)",
                annotation_text=code, annotation_position="top left",
                layer="below"
            )
        else:
            base = COLORS.get(code, "#888888")
            if base.startswith("#") and len(base) == 7:
                r = int(base[1:3], 16); g = int(base[3:5], 16); b_ = int(base[5:7], 16)
                fill = f"rgba({r},{g},{b_},0.18)"
            else:
                fill = "rgba(128,128,128,0.18)"
            fig.add_vrect(x0=x0, x1=x1, fillcolor=fill, line_width=0, layer="below")


def draw_sl_tp(fig: go.Figure, entry_price: float, side: str, sl_pts: float, tp_pts: float):
    if side.upper() == "LONG":
        sl = entry_price - sl_pts
        tp = entry_price + tp_pts
    else:
        sl = entry_price + sl_pts
        tp = entry_price - tp_pts
    fig.add_hline(y=sl, annotation_text="SL", annotation_position="right", line=dict(width=1))
    fig.add_hline(y=tp, annotation_text="TP", annotation_position="right", line=dict(width=1))


def make_chart(df_cont: pd.DataFrame,
               trades: pd.DataFrame | None,
               focus_idx: int | None,
               pre_minutes: int,
               sl_pts: float,
               tp_pts: float,
               auto_zoom_last_pre: bool):
    """Continuous bar chart (no session gaps) with tight vertical padding and pre-context overlays."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_cont["bar"],
        open=df_cont["open"], high=df_cont["high"], low=df_cont["low"], close=df_cont["close"],
        name="price", showlegend=False
    ))
    ylo, yhi = y_range_tight(df_cont["high"].to_numpy(), df_cont["low"].to_numpy(), pad_ratio=0.01)
    fig.update_yaxes(range=[ylo, yhi])

    if trades is not None and not trades.empty and focus_idx is not None and 0 <= focus_idx < len(trades):
        t = trades.iloc[focus_idx]
        et = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
        if not pd.isna(et) and et in df_cont.index:
            entry_bar = int(np.where(df_cont.index.values == np.datetime64(et))[0][0])
            codes = t["PatternList"] if isinstance(t.get("PatternList", None), list) else None
            add_pre_context_blocks(fig, entry_bar, pre_minutes, codes)
            draw_sl_tp(fig, float(t["entry_price"]), str(t["side"]), float(sl_pts), float(tp_pts))
            if auto_zoom_last_pre:
                x0 = max(0, entry_bar - pre_minutes)
                x1 = min(int(df_cont["bar"].iloc[-1]), entry_bar + max(5, pre_minutes // 3))
                fig.update_xaxes(range=[x0, x1])

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=8, r=8, t=24, b=8),
        xaxis=dict(type="linear", rangeslider=dict(visible=False)),
        dragmode="pan",
        hovermode="x unified",
        showlegend=False,
    )
    return fig

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Pattern Gated Backtest", layout="wide")

ss = st.session_state
ss.setdefault("minutes", None)
ss.setdefault("trades", None)
ss.setdefault("focus_idx", None)
ss.setdefault("auto_zoom", True)
# KDTree state
ss.setdefault("V", None)            # feature matrix (np.ndarray) or None
ss.setdefault("kept_idx", None)     # list of trade indices used to build V
ss.setdefault("nn_table", None)     # last nearest-neighbor results (pd.DataFrame)

st.title("Pattern-Gated Backtest")

with st.sidebar:
    st.subheader("Import & slice")
    col_a, col_b = st.columns(2)
    with col_a:
        start_day = st.date_input("Start day", value=None, key="start_day_ui")
    with col_b:
        end_day = st.date_input("End day", value=None, key="end_day_ui")
    pre_ctx = st.selectbox("Pre-context minutes", [15,20,25,30,35,40,45,50,55,60], index=0)

    use_upload = st.toggle("Upload CSV instead of default", value=False)
    uploaded = st.file_uploader("CSV (time,open,high,low,close)", type=["csv"]) if use_upload else None

    st.subheader("Reverse Moby")
    sl_pts = st.number_input("Stop loss (pts)", value=10.0)
    tp_pts = st.number_input("Take profit (pts)", value=10.0)
    cons_thr = st.number_input("Consolidation threshold", value=0.0025, format="%f")  # 50% of 0.005

    st.toggle("Auto zoom to pre-context", value=True, key="auto_zoom")

    c1, c2 = st.columns(2)
    with c1:
        btn_import = st.button("Import data", use_container_width=True)
    with c2:
        btn_run = st.button("Run strategy", use_container_width=True)

left, right = st.columns([3,2], gap="large")

with left:
    st.subheader("Chart")
    chart_ph = st.empty()

with right:
    st.subheader("Trades")
    table_ph = st.empty()
    focus_sel = st.empty()

# -------- Import action --------
if btn_import:
    try:
        if use_upload and uploaded is not None:
            tmp = pd.read_csv(uploaded)
            tmp["time"] = pd.to_datetime(tmp["time"], utc=True)
            tmp = tmp[["time","open","high","low","close"]]
        else:
            tmp = load_minutes(str(DEFAULT_CSV), CFG["session"]["start"], CFG["session"]["end"])
        tmp = slice_by_days(tmp, TZ,
                            start_day if isinstance(start_day, date) else None,
                            end_day if isinstance(end_day, date) else None)
        if tmp.empty:
            st.error("No data after slicing — widen the date range.")
        else:
            ss.minutes = tmp
            ss.trades = None
            ss.focus_idx = None
            ss.V = None
            ss.kept_idx = None
            ss.nn_table = None
            st.success(f"Loaded {len(tmp)} minute bars.")
    except Exception as e:
        st.error(f"Import failed: {e}")

# -------- Run strategy --------
if btn_run:
    if ss.minutes is None or ss.minutes.empty:
        st.error("Load data first.")
    else:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("reversed_moby_strategy", str(ROOT / "reversed_moby_strategy.py"))
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)

            raw = ss.minutes.copy()
            res = mod.run_strategy(
                raw,
                mod.Config(consolidation_threshold=float(cons_thr),
                           stop_loss_points=float(sl_pts),
                           take_profit_points=float(tp_pts),
                           tz=TZ),
                mod.Columns(time="time", open="open", high="high", low="low", close="close"),
            )
            trades = res["trades"].copy().reset_index(drop=True)

            # ensure tz-aware
            for col in ("entry_time","exit_time"):
                if col in trades.columns:
                    trades[col] = pd.to_datetime(trades[col], utc=True, errors="coerce")

            # PnL in dollars
            if "pnl_points" in trades.columns:
                trades["pnl_money"] = trades["pnl_points"].fillna(0.0) * MONEY_PER_POINT

            # ----- attach patterns (Pattern, PatternViz, PatternList) -----
            trades, V, labels, kept_idx = attach_patterns(ss.minutes, trades, int(pre_ctx))

            ss.trades = trades
            ss.focus_idx = 0 if len(trades) else None
            ss.V = V if (isinstance(V, np.ndarray) and V.size) else None
            ss.kept_idx = kept_idx
            ss.nn_table = None  # clear previous search

            st.success(f"Trades: {len(trades)}  |  Patterned: {len(kept_idx)}")
        except Exception as e:
            st.error(f"Run failed: {e}")

# -------- Draw --------
def redraw():
    if ss.minutes is None or ss.minutes.empty:
        chart_ph.info("Load data to see the chart.")
        table_ph.empty(); focus_sel.empty()
        return

    df = ss.minutes.copy()
    df_cont = to_continuous_index(df).set_index("time")

    fig = make_chart(df_cont, ss.trades, ss.focus_idx,
                     pre_minutes=int(pre_ctx),
                     sl_pts=float(sl_pts),
                     tp_pts=float(tp_pts),
                     auto_zoom_last_pre=bool(st.session_state.auto_zoom))

    chart_ph.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # trades table numbered + banners
    if ss.trades is not None and not ss.trades.empty:
        t = ss.trades.copy()
        t["trade_id"] = t.index  # explicit numbering
        show = ["trade_id","entry_time","exit_time","side","entry_price","exit_price","pnl_points","pnl_money","Pattern","PatternViz"]
        show = [c for c in show if c in t.columns]
        table_ph.dataframe(t[show], use_container_width=True, hide_index=True)

        # focus chooser
        sel = focus_sel.selectbox("Focus trade (trade_id)", options=t.index.tolist(),
                                  index=int(ss.focus_idx or 0))
        if sel != ss.focus_idx:
            ss.focus_idx = int(sel)
            st.rerun()
    else:
        table_ph.info("Run the strategy to see the trades.")
        focus_sel.empty()

redraw()

# -------- KDTree Similarity (replaces Compare patterns) --------
st.markdown("---")
st.subheader("Similarity search (KDTree)")

if ss.trades is None or ss.trades.empty or ss.V is None or (isinstance(ss.V, np.ndarray) and ss.V.size == 0) or not ss.kept_idx:
    st.info("Run strategy to build vectors and patterns first.")
else:
    # Choose base trade only among those that have vectors (kept_idx)
    patterned_ids = list(map(int, ss.kept_idx))
    cols = st.columns([2, 2, 1])
    with cols[0]:
        base_trade = st.selectbox("Base trade (patterned only)", options=patterned_ids, index=0, key="nn_base")
    with cols[1]:
        topN = st.slider("Top-N neighbors", min_value=5, max_value=100, value=50, step=1)
    with cols[2]:
        do_search = st.button("Find nearest", use_container_width=True)

    if do_search:
        try:
            V = ss.V
            kept_idx = ss.kept_idx
            # map trade_id -> row in V
            t2r = {int(tid): r for r, tid in enumerate(kept_idx)}
            if base_trade not in t2r:
                st.error("Selected trade has no vector (not enough pre-context).")
            else:
                vi = VectorIndex(V)
                qrow = t2r[base_trade]
                dist, nn = vi.query_by_vector(V[qrow], k=int(topN))
                # simple similarity score
                sim = 1.0 / (1.0 + dist)
                match_trade_ids = [int(kept_idx[j]) for j in nn]
                out = ss.trades.iloc[match_trade_ids][["entry_time", "side", "pnl_points", "pnl_money", "Pattern", "PatternViz"]].copy()
                out.insert(0, "trade_id", match_trade_ids)
                out.insert(1, "similarity", np.round(sim, 4))
                ss.nn_table = out.reset_index(drop=True)
        except Exception as e:
            st.error(f"KDTree search failed: {e}")

    if ss.nn_table is not None and not ss.nn_table.empty:
        st.dataframe(ss.nn_table, use_container_width=True, hide_index=True)
        snap_id = st.selectbox("Snap chart to neighbor trade_id", options=ss.nn_table["trade_id"].tolist(), index=0, key="nn_snap")
        if st.button("Snap", key="nn_snap_btn"):
            ss.focus_idx = int(snap_id)
            st.rerun()
