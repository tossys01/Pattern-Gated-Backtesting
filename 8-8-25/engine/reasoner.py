from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class BlockLabel:
    code: str
    reason: Dict[str, Any]

def _runs(signs):
    runs = []
    if len(signs) == 0:
        return runs
    curr = signs[0]; ln = 1
    for s in signs[1:]:
        if s == curr:
            ln += 1
        else:
            runs.append((curr, ln))
            curr = s; ln = 1
    runs.append((curr, ln))
    return runs

def classify_block(cb_pts, hb_pts, lb_pts, cfg: Dict[str, Any]) -> BlockLabel:
    """
    Classify a 5 minute block using point based thresholds.
    Inputs are offsets in points relative to the first open of the block.
    """
    c = np.asarray(cb_pts, dtype=float)
    h = np.asarray(hb_pts, dtype=float)
    l = np.asarray(lb_pts, dtype=float)

    # per minute close deltas
    d = np.array([c[1]-c[0], c[2]-c[1], c[3]-c[2], c[4]-c[3]])
    net = float(c[-1] - c[0])
    span = float(np.max(h) - np.min(l))  # highest high - lowest low within block
    signs = np.sign(d)
    flips = int(np.sum(np.abs(np.diff(signs)) > 0))
    bars_net = int(np.sum(np.sign(d) == np.sign(net))) if net != 0 else int(np.sum(np.sign(d) >= 0))

    # origin excursions around 0 reference
    path = np.array([0.0, d[0], d[0]+d[1], d[0]+d[1]+d[2], net])
    max_above = float(np.max(path))
    max_below = float(np.min(path))
    max_origin_exc = float(max(max_above, -max_below))

    # Consolidation
    cons = cfg["consolidation"]
    if span <= cons["max_span"] and abs(net) <= cons["max_net_abs"] \
       and flips >= cons["min_turn_points"] and max_origin_exc <= cons["max_origin_excursion"]:
        return BlockLabel("Z", {
            "stage": "Z", "span": span, "net": net, "flips": flips, "max_origin_exc": max_origin_exc
        })

    # Reversal
    rev = cfg["reversal"]
    if span >= rev["min_span"] and flips == 1:
        # pivot at 3rd or 4th bar boundary
        change_idx = int(np.where(np.diff(signs) != 0)[0][0]) + 2  # 2-based minute index in 1..4 -> pivot 3 or 4
        if change_idx in rev["pivot_allowed"]:
            if change_idx == 3:
                leg1 = float(c[2] - c[0])
                leg2 = float(c[4] - c[2])
            else:
                leg1 = float(c[3] - c[0])
                leg2 = float(c[4] - c[3])

            crosses_origin = (max_above > 0 and max_below < 0)
            if abs(leg1) >= rev["min_leg"] and abs(leg2) >= rev["min_leg"] \
               and (crosses_origin if rev.get("require_origin_cross", True) else True):
                if net > 0 and leg1 < 0 and leg2 > 0:
                    return BlockLabel("R+", {"stage": "R", "pivot": change_idx, "leg1": leg1, "leg2": leg2, "span": span, "net": net})
                if net < 0 and leg1 > 0 and leg2 < 0:
                    return BlockLabel("R-", {"stage": "R", "pivot": change_idx, "leg1": leg1, "leg2": leg2, "span": span, "net": net})

    # Drive
    drv = cfg["drive"]
    if abs(net) >= drv["min_net"] and flips <= drv["max_turn_points"] and bars_net >= drv["min_bars_in_net"]:
        # bar rates in direction of net and max counter bar
        dir_net = np.sign(net) if net != 0 else 1
        dir_rates = np.abs(d[np.sign(d) == dir_net])
        counter_rates = np.abs(d[np.sign(d) == -dir_net])
        if len(dir_rates) >= 1 and np.all(dir_rates >= drv["min_bar_rate"]) and (len(counter_rates) == 0 or np.max(counter_rates) <= drv["max_counter_bar"]):
            return BlockLabel("D+" if net > 0 else "D-", {"stage": "D", "net": net, "span": span, "flips": flips})

    # Step
    st = cfg["step"]
    if st["min_net"] <= abs(net) < st["max_net"]:
        dir_net = 1 if net >= 0 else -1
        runs = _runs(list(np.sign(d)))
        pushes = sum(1 for s, ln in runs if s == dir_net)
        retraces = sum(1 for s, ln in runs if s == -dir_net)
        push_rates = [abs(val) for val in d if np.sign(val) == dir_net]
        retr_rates = [abs(val) for val in d if np.sign(val) == -dir_net]
        avg_push = float(np.mean(push_rates)) if push_rates else 0.0
        avg_retr = float(np.mean(retr_rates)) if retr_rates else 1e-9
        retrace_exc_ok = max_origin_exc <= st["max_retrace_excursion"] or avg_retr <= st["max_retrace_excursion"]
        if pushes >= st["min_pushes"] and retraces >= st["min_retraces"] \
           and avg_push >= st["push_over_retrace_strength"] * avg_retr and retrace_exc_ok:
            return BlockLabel("S+" if net > 0 else "S-", {"stage": "S", "net": net, "span": span, "pushes": pushes, "retraces": retraces})

    # Fallback
    return BlockLabel("UNK", {"stage": "F", "net": net, "span": span, "flips": flips})
