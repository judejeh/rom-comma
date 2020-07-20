import time
from numpy import full
import romcomma.model as model
import numpy as np
import pandas as pd
from romcomma.data import Store, Fold, Frame
from romcomma.typing_ import NP, Union, Tuple, Sequence
from shutil import rmtree, copytree
from pathlib import Path
from scipy import optimize
from pyDOE import lhs
from scipy.stats.distributions import norm
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


def load_data(source: str, store_name: str, origin_csv_parameters=None) -> Store:
    if origin_csv_parameters is None:
        origin_csv_parameters = {'index_col': 0}
    store = Store.from_csv(BASE_PATH / store_name, BASE_PATH / source, **origin_csv_parameters)
    # from here X is a dataframe of inputs and Y is a dataframe of outputs.
    return store


def plot_data(store: Store, store_name: str, hist_num_name: str = "Data_Num_Hists",
              hist_str_name: str = "Data_Str_Hists_"):
    df = store.X
    for y in df.columns:
        if df[y].dtype == np.float64 or df[y].dtype == np.int64:
            # df.hist(sharey=True, bins=40)
            df.hist(bins=40)
            plt.tight_layout()
            fig_name = hist_num_name + ".pdf"
            plt.savefig(BASE_PATH / store_name / fig_name)
            plt.clf()
        else:
            df[y].value_counts().plot(kind="bar")
            fig_name = hist_str_name + str(y) + ".pdf"
            plt.savefig(BASE_PATH / store_name / fig_name)
    return


BASE_PATH = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Silica")
STORE_NAME = "Test_Store"
Store = load_data("Data.csv", STORE_NAME)
plot_data(Store, STORE_NAME)
