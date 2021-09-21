def clean_data_Q(df, col: str):
    df_clean = df[df[col] < np.percentile(df[col], 95)]

    df_clean = df_clean[df_clean['Q'] >= 0]

    return df_clean

def clean_data_top5(df, col :str, col1 :str):
    df_clean = df[(df[col] < np.percentile(df[col], 95)) & (df[col1] < np.percentile(df[col1], 95))]
    return df_clean

def clean_data(df, col :str, col1 :str):
    df_clean = df[(df[col] < np.percentile(df[col], 97.5)) & (df[col] > np.percentile(df[col], 2.5)) & (df[col1] < np.percentile(df[col1], 97.5)) & (df[col1] > np.percentile(df[col1], 2.5))]
    return df_clean
