import pandas as pd
from KID1_Prediction_Class import NN_KID1_model
import numpy as np


def MakeInputData(df_KID, KID_NAME):
    input_current_state=df_KID[['CPU_utilization','FAN (%)', 'CPU_temperature(degC)', 'PS']].reset_index()
    input_job=df_KID[KID_NAME+'_jobs'].reset_index()
    Inp_np=np.empty((0, 5), float)
    for i in range(len(df_KID)-1):
        Inp_series=pd.concat([input_current_state.ix[i], input_job.ix[i+1]]).drop('Elapsed time (s)')
        tmp_np=np.array(Inp_series)
        Inp_np=np.append(Inp_np, [tmp_np], axis=0)
    input_data_np=Inp_np
    return input_data_np


df_KID1_ml=pd.read_pickle("/home/nakajo/GT_Kids/Results/ML_data/vailidation_ver1/ml_data/df_KID1_ml.pkl")
NN_KID1_model=NN_KID1_model()
input_data=MakeInputData(df_KID1_ml, "KID1")

results=NN_KID1_model.predict(input_data[1:3, :])

print(results[:,0])
