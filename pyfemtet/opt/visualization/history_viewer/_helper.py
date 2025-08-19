import numpy as np
import pandas as pd
from pyfemtet.opt.history import *


def has_full_bound(history: History, prm_name, df: pd.DataFrame = None) -> bool:
    if df is None:
        df = history.get_df()

    lb_name = CorrespondingColumnNameRuler.prm_lower_bound_name(prm_name)
    if lb_name in df.columns:
        lb = df[lb_name].dropna().min()
    else:
        lb = df[prm_name].dropna().min()

    ub_name = CorrespondingColumnNameRuler.prm_upper_bound_name(prm_name)
    if ub_name in df.columns:
        ub = df[ub_name].dropna().max()
    else:
        ub = df[prm_name].dropna().max()

    if lb == ub:
        return False

    if np.isnan(float(lb)) or np.isnan(float(ub)):
        return False

    return True


def control_visibility_by_style(visible: bool, current_style: dict):

    visibility = 'inline' if visible else 'none'
    part = {'display': visibility}

    if current_style is None:
        return part

    else:
        current_style.update(part)
        return current_style
