# !git clone http://github.com/AccSrd/multimodal-Parkinson-data-processing.git

import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose,Resize,CenterCrop,Normalize,ToTensor
import clip
from transformers import BertTokenizer
from tqdm import tqdm
import pickle


# data_total = []
# for task in ["_1","_2","_3"]:
#     task_data_tmp = pd.read_csv("python/dataset/task" + task + ".txt",header=None)
#     data_total.append(task_data_tmp)
# data_total = pd.concat(data_total,ignore_index=True)

# X_data = data_total.iloc[:,2:-1].apply(lambda x:np.round(x).astype(int)).to_numpy()
# y_data = data_total.iloc[:,-1].to_numpy()

# X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.2,random_state=42)

# print(X_train)
# train_data = pd.DataFrame({'sample':X_train.to_list(),'label':y_train})
# train_data['sample'] = train_data['sample'].apply(lambda x: ' '.join(map(str,x)),)
# train_data.to_csv("train.csv",index=False)

# test_data = pd.DataFrame({'sample':X_test.to_list(),'label':y_test})
# test_data['sample'] = test_data['sample'].apply(lambda x: ' '.join(map(str,x)),)
# test_data.to_csv("test.csv",index=False)

# df = pd.read_csv("python/dataset/test_EEG.csv")
# result = []
# for i in range(len(df)):
#     result.append(df.iloc[i,0].split(" "))
# result_df = pd.DataFrame(result)
# cols = ["FP1","FP2","F3","F4","C3","C4","P3","P4","01","02"
#         ,"F7","F8","P7","P8","Fz","Cz","Pz","FC1","FC2","CP1"
#         ,"CP2","FC5","FC6","CP5","CP6","EMG1","EMG2","IO","EMG3","EMG4"]
# result_df.columns = cols
# result_df["label"] = df["label"]
# print(result_df)

# result_df.to_csv("python/dataset/test_EEG.csv")
print("Now it is successed")
