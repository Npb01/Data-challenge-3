import numpy as np
import pandas as pd
import lmfit
import datetime
from datetime import date
from sklearn import linear_model
import datetime
import argparse
pd.options.mode.chained_assignment = None  # default='warn'
from current_model import get_model, calc_vegetation, predict_vegetation, negative_backwater_to_zero

#---------------Please adjust variables here or in the command line------------------------------------------------------------
data_path=r'C:\\Users\\20193727\\Downloads\\data_for_students\\data\\feature_tables\\'
          #r"  #'C:\\Users\\Bringer\\Documents\\JADS\\Aa-en-Maas\\Features\\' #(--data_path)
weir='211C_211B' #(--weir)
risk_date='2021-04-01' # (--risk_date)
prediction=True # True for prediction (--prediction)
last_days=7 # (--last_days) For prediction: Defines how many days the linear model takes into account to predict the next 21 days
avg_temp=22 # (--avg_temp) For prediction: Average Temperature adjusts the prediction +/- 20%
#---------------End of adjust variables-------------------------------------------------------------------------------------------

def get_data(weir,data_path,date_format=False):
    ''' Get the feature data of the individual weir
    Keyword arguments:
    weir -- the weir name as string
    date_format -- date_format boolean
    data_path -- the local path of the weir feature data csv's
    Returns: data as dataframe'''
    datapath=data_path+weir+'_feature_table.csv'
    data=pd.read_csv(datapath,index_col="TIME",parse_dates=True)
    if date_format:
        data.index=data.index.strftime('%Y-%m-%d')
    return data

