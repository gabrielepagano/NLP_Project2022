def pd_pear_corr(df, p):
    """
        Calculates the Pearson Correlation between columns of the provided dataframe

        Args:
            df: the dataframe for the Pearson Correlation calculation
            p: boolean, if True it prints results

        Returns:
            correlation: the correlation df
    """
    correlation = df.corr()

    if p:
        print("\nThis is corr :\n")
        print(correlation)

    return correlation
