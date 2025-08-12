import pandas as pd

def compute_mae_mfe(minute_df: pd.DataFrame, entry_time, exit_time, side: str):
    seg = minute_df.loc[entry_time: exit_time]
    if seg.empty:
        return 0.0, 0.0
    prices = seg['close']
    if side.upper() == 'LONG':
        mae = float((prices.min() - prices.iloc[0]))
        mfe = float((prices.max() - prices.iloc[0]))
    else:
        mae = float((prices.max() - prices.iloc[0])) * -1.0
        mfe = float((prices.min() - prices.iloc[0])) * -1.0
    return mae, mfe