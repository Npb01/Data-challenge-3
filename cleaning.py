import pandas as pd
import numpy as np


df_all_weirs = pd.read_csv(r'C:\\Users\\20193727\\Downloads\\data_for_students\\data\\all_weirs.csv')


def clean_data_Q(df):
    #df_clean = df[df[col] < np.percentile(df[col], 95)]

    df_clean = df[df['Q'] >= 0]

    return df_clean

def clean_data_top5(df, col :str, col1 :str):
    df_clean = df[(df[col] < np.percentile(df[col], 95)) & (df[col1] < np.percentile(df[col1], 95))]
    return df_clean

def clean_data(df, col :str, col1 :str):
    df_clean = df[(df[col] < np.percentile(df[col], 97.5)) & (df[col] > np.percentile(df[col], 2.5)) & (df[col1] < np.percentile(df[col1], 97.5)) & (df[col1] > np.percentile(df[col1], 2.5))]
    return df_clean

df_all_weirs = pd.read_csv(r'C:\\Users\\20193727\\Downloads\\data_for_students\\data\\all_weirs.csv')

df = clean_data_Q(df_all_weirs)
df_clean = clean_data(df, 'Q', 'VERSCHIL')

#pd.DataFrame.to_csv(df_clean)
#df_clean.to_csv(r'C:\Users\20193727\PycharmProjects\Data-challenge-3\all_weirs_clean.csv')