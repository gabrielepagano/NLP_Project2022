import numpy as np
import pandas as pd


def pd_pearcorr(df):
    correlation = df.corr()

    print("\nThis is corr :\n")
    print(correlation)


def np_pearcorr(np1, np2):
    corr = np.corrcoef(np1, np2)

    print("\nThis is corr :\n")
    print(corr)
