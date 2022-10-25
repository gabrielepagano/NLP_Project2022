import numpy as np
import pandas as pd

def pd_pearcorr_rating_sentiment(df):
    correlation = df.corr()

    print("\nThis is corr :\n")
    print(correlation)

def np_pearcorr_rating_sentiment(np1, np2):
    corr = np.corrcoef(np1, np2)

    print("\nThis is corr :\n")
    print(corr)
