import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


data_total = pd.DataFrame()
for task in ["_1","_2","_3"]:
    task_data_tmp = pd.read_csv("python/dataset/task" + task + ".txt",header=None)
    data_total = pd.concat([data_total,task_data_tmp],axis=0)

data = data_total.iloc[:,:-1].apply(lambda x:np.round(x).astype(int))
label = data_total.iloc[:,-1]

train_data, test_data, train_label, test_label = train_test_split(data,label,test_size=0.2,random_state=42)
train_EEG = train_data.iloc[:,:30]
test_EEG = test_data.iloc[:,:30]
train_act = train_data.iloc[:,30:]
test_act = test_data.iloc[:,30:]

save_path = "data/processed/"
train_EEG.to_csv(save_path + "train_EEG.csv",index=False)
test_EEG.to_csv(save_path + "test_EEG.csv",index=False)
train_act.to_csv(save_path + "train_act.csv",index=False)
test_act.to_csv(save_path + "test_act.csv",index=False)
train_label.to_csv(save_path + "train_label.csv",index=False)
test_label.to_csv(save_path + "test_label.csv",index=False)