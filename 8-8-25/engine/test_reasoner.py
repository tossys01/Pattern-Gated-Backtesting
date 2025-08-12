import numpy as np
from engine.reasoner import classify_block

CFG = {
    "eps": 0.01,
    "consolidation": {
        "excursion_max": 7.0,
        "net_abs_max": 2.0,
        "min_turns": 2,
        "one_min_abs_max": 2.0,
        "pullback_max": 3.0
    },
    "reversal": {
        "pivot_allowed": [2, 3, 4],
        "leg_min": 3.5,
        "origin_cross_min": 0.5,
        "excursion_min": 7.0,
        "final_net_abs_min": 1.0
    },
    "drive": {
        "net_min": 9.0,
        "max_turns": 1,
        "rate_dir_min": 1.8,
        "min_bars_in_dir": 4,
        "counter_bar_max": 1.0
    },
    "step": {
        "net_min": 5.0,
        "net_max": 9.0,
        "min_bars_in_dir": 3,
        "pullback_max": 1.5,
        "avg_dir_bar_min": 1.0,
        "avg_counter_bar_max": 1.0,
        "turns_allowed": [1, 2]
    }
}


def make_block_from_path(path_vals, base=100.0):
    """
    Helper to build synthetic 5 minute arrays from cumulative close path.
    path_vals must be length 5 with path[0] == 0.
    Creates closes at base + path, highs and lows with small wiggle to control excursion as needed.
    """
    c = base + np.array(path_vals, dtype=float)
    # default highs and lows match closes unless bumped by caller
    h = c.copy()
    l = c.copy()
    return c, h, l


def test_consolidation_basic():
    # small path inside 2 net and 7 excursion, choppy turns
    path = [0, +0.5, -0.2, +0.3, -0.1]  # many turns, tiny moves
    c, h, l = make_block_from_path(path, 100)
    # give excursion 6.0
    h[2] += 3.0
    l[1] -= 3.0
    lab = classify_block(c, h, l, CFG)
    assert lab.code == "Z"


def test_reversal_up_pivot3():
    # leg1 down about 4.0, leg2 up about 5.0 crossing origin
    path = [0, -2.0, -4.0, -4.0, +1.5]  # deltas: -2, -2, 0, +5.5 pivot at delta 4 -> minute 4
    c, h, l = make_block_from_path(path, 100)
    # ensure excursion >= 7
    h[4] += 1.0
    l[2] -= 2.5
    lab = classify_block(c, h, l, CFG)
    assert lab.code == "R+"


def test_drive_down_strong():
    # net about -10 with low turns and small counters
    path = [0, -2.5, -5.0, -7.5, -10.0]  # deltas -2.5 each
    c, h, l = make_block_from_path(path, 100)
    lab = classify_block(c, h, l, CFG)
    assert lab.code == "D-"


def test_step_up_clean():
    # net about +6 with moderate small pullbacks
    path = [0, +1.5, +3.0, +2.2, +6.0]  # deltas +1.5, +1.5, -0.8, +3.8
    c, h, l = make_block_from_path(path, 100)
    lab = classify_block(c, h, l, CFG)
    assert lab.code == "S+"


def test_buffers_no_overlap_z_vs_r():
    # build exc just below 7 so must be Z, not R
    path = [0, +1.0, -1.0, +1.0, -1.5]
    c, h, l = make_block_from_path(path, 100)
    h[1] += 3.0
    l[2] -= 3.9  # total excursion < 7
    lab = classify_block(c, h, l, CFG)
    assert lab.code == "Z"

    # now push excursion just above 7 to allow reversal if legs meet thresholds
    h[4] += 4.0
    # craft legs
    c2 = c.copy()
    c2[1] = c2[0] + 2.0
    c2[2] = c2[0] + 4.0
    c2[3] = c2[0] + 4.0
    c2[4] = c2[0] - 1.5  # cross below origin by more than 0.5
    lab2 = classify_block(c2, h, l, CFG)
    # might be UNK if legs fail, but should not be Z
    assert lab2.code in {"R-", "UNK"}


def test_buffers_no_overlap_step_vs_drive():
    # Step top boundary just under 9
    path = [0, +2.0, +4.0, +6.5, +8.99]
    c, h, l = make_block_from_path(path, 100)
    lab = classify_block(c, h, l, CFG)
    assert lab.code in {"S+", "UNK"}  # should not be Drive

    # Drive floor just above 9
    path2 = [0, +2.0, +4.0, +6.5, +9.2]
    c2, h2, l2 = make_block_from_path(path2, 100)
    lab2 = classify_block(c2, h2, l2, CFG)
    assert lab2.code in {"D+", "UNK"}  # should not be Step
