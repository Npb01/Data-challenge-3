import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit
import datetime
from datetime import date
from sklearn import linear_model
import datetime
#from datetime import datetime
import argparse
pd.options.mode.chained_assignment = None  # default='warn'
from current_model import get_model, calc_vegetation, negative_backwater_to_zero
from scipy.signal import argrelextrema
import json

#---------------Please adjust variables here or in the command line------------------------------------------------------------
data_path=r'C:\\Users\\20193727\\Downloads\\data_for_students\\data\\feature_tables\\'
          #r"  #'C:\\Users\\Bringer\\Documents\\JADS\\Aa-en-Maas\\Features\\' #(--data_path)
weir='103BIB_103BIC' #(--weir)
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

def predict_vegetation(weir, train_days,avg_temp,data_path, pred_date_idx):
    '''Predict the vegetation of the next 21 days based on the last 7 days with linear model
    Keyword arguments:
    weir -- the weir name as string
    last_days -- the number of days the linear model should base the prediction on
    avg_temp -- the average temperature adjusting the predictions by +/- 20%
    data_path -- the local path of the weir feature data csv's
    Returns: Dataframe of the backwater predictions of the next 21 days'''
    data=get_data(weir,data_path,date_format=True)
    data.reset_index(inplace=True)
    # Get the last data points depending on number of last_days
    last_data=data[pred_date_idx-train_days:pred_date_idx]
    # Get last day to calculate
    last_day = datetime.datetime.strptime(last_data.iloc[-1]['TIME'], "%Y-%m-%d") #setting the -1 to -21 might allow us to predict 21 days that are already in the data
                                                                                  # thus allowing us to compare data to our predictions.
    # Get dates of the next 21 days
    new_dates=[last_day+datetime.timedelta(days=i) for i in range(1,5)]
    # Calculate back water by vegetation for the last days
    last_data['vegetation']=last_data['TIME'].apply(lambda row:calc_vegetation(weir,get_data(weir,data_path,date_format=True),row,data_path))
    last_data.reset_index(inplace=True)
    # Define linear model
    reg = linear_model.LinearRegression()
    # Take index and the back water by vegetation as training data
    x_train=last_data.index.to_numpy().reshape(-1, 1)
    y_train=last_data['vegetation'].to_numpy().reshape(-1, 1)
    # Fit the linear model on the last days
    reg.fit(x_train,y_train)
    # Get index for the next 21 days
    x_test=[x_train[-1]+i for i in range(1,5)]
    # Predict the vegetation for the next 21 days
    predictions=reg.predict(x_test)
    # Format
    predictions= [item for elem in predictions.tolist() for item in elem]
    # Depending on the temperature add multplication value to adjust values
    try:
        if (avg_temp > 25):
            predictions=[pred*1.2 for pred in predictions]
        elif (avg_temp<20):
            predictions=[pred*0.8 for pred in predictions]
    except:
        print("The Temperature was not available")
    data = {'TIME':  new_dates,'Predicted backwater by vegetation': predictions}
    df = pd.DataFrame (data, columns = ['TIME','Predicted backwater by vegetation'])
    #print(len(df))
    #df['Q'] = data['Q'][-20:]
    #print(df)
    return df

def predict_whole_df(weir, last_days):
    parser = argparse.ArgumentParser(description='Arguments get parsed via --commands')
    parser.add_argument('--weir', type=str, default=weir)
    parser.add_argument('--risk_date', type=str, default=risk_date)
    parser.add_argument('--data_path', type=str, default=data_path)
    parser.add_argument('--prediction', type=bool, default=prediction)
    parser.add_argument('--last_days', type=int, default=last_days)
    parser.add_argument('--avg_temp', type=int, default=avg_temp)
    args = parser.parse_args()

    df = get_data(args.weir, args.data_path)
    final_df = predict_vegetation(weir=args.weir, train_days=args.last_days, avg_temp=args.avg_temp,
                             data_path=args.data_path, pred_date_idx=(args.last_days+1))

    for idx in range(8,len(df)):
        df1 = predict_vegetation(weir=args.weir, train_days=args.last_days, avg_temp=args.avg_temp,
                             data_path=args.data_path, pred_date_idx=idx)
        final_df = final_df.append(df1.loc[1], ignore_index=True)

    return final_df

#df_for_weir = predict_whole_df('103BIB_103BIC', 7)

#plt.plot(df_for_weir['Predicted backwater by vegetation'])
#plt.show()

#df_for_year.reset_index(inplace=True)

#array_for_maxima = np.array(df_for_year['Predicted backwater by vegetation'])
#max_idxs = argrelextrema(array_for_maxima, np.greater)
#max_idxs_list = list(max_idxs[0])
#peak_dates = df_for_year.loc[max_idxs_list, ['TIME']]
#print(peak_dates)

with open(r'C:\Users\20193727\Downloads\mowing_dates_dict.json') as json_file:
    mow_data = json.load(json_file)

def comparison(mow_dates, weir):
    df_whole_weir = predict_whole_df(weir, 7)

    array_for_maxima = np.array(df_whole_weir['Predicted backwater by vegetation'])
    max_idxs = argrelextrema(array_for_maxima, np.greater)
    max_idxs_list = list(max_idxs[0])
    peak_dates = df_whole_weir.loc[max_idxs_list, ['TIME']]
    peak_dates = [pd.to_datetime(i) for i in peak_dates['TIME']]
    #datetime.datetime.strptime(i, '%y-%m-%d')

    mowing_dates = [pd.to_datetime(i) for i in mow_dates[weir]]

    correct_pred = 0
    for D in peak_dates:
        res = min(mowing_dates, key=lambda sub: abs(sub - D))
        min_diff = res-D
        if min_diff.days < 7:
            correct_pred += 1

    acc = correct_pred / len(peak_dates)

    print(acc)

comparison(mow_data, '103BIB_103BIC')

# consider that if all 7 days before it ar 0 then the prediction will be zero consider this in analysis