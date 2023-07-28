import numpy as np
import pandas as pd


def calc_num_objects(hist: pd.Series):
    n_ref = hist.sum() // 10e4
    n_depots = hist.sum() // 20e3
    clip_ref = np.clip(n_ref, 0, 5)
    clip_depots = np.clip(n_depots, 0, 25)
    return int(clip_ref), int(clip_depots)
