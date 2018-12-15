import tensorflow as tf
import numpy as np
import pandas as pd
import os


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

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

def MakeOutputData(df_KID, KID_NAME):
    output_candidates=df_KID[['CPU_temperature(degC)', KID_NAME+'_serve_time']].reset_index()
    Oup_np=np.empty((0,2), float)
    oup_series=output_candidates.drop('Elapsed time (s)', axis=1)
    tmp_np=np.array(oup_series)
    Oup_np=np.append(Oup_np,tmp_np, axis=0)
    output_data_np=Oup_np[1:] # skip the initital state 
    return output_data_np

def DetermineNeuron(nInput, nOutput):
    global W_fc1, b_fc1, h_fc1, W_fc2, b_fc2, h_fc2, keep_prob, W_fc3, b_fc3, y
    W_fc1 = weight_variable([nInput,30])
    b_fc1 = bias_variable([30])
    h_fc1 = tf.nn.relu(tf.matmul(x,W_fc1) + b_fc1)

    W_fc2 = weight_variable([30,10])
    b_fc2 = bias_variable([10])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

    keep_prob = tf.placeholder(tf.float32)

    W_fc3 = weight_variable([10,nOutput])
    b_fc3 = bias_variable([nOutput])
    y = tf.nn.relu(tf.matmul(h_fc2,W_fc3) + b_fc3)

def DefinePlaceHolder(nInput, nOutput):
    global x, y_, w, b
    x= tf.placeholder(tf.float32, shape=[None, nInput], name = "input")
