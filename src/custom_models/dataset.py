from torch.utils.data import Dataset
import pickle
import pandas as pd
import torch
import numpy as np

def set_seed(seed):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

set_seed(980616)

class MultiModalDataset_ti(Dataset):
    '''
    treat eeg as txt, act as img; ti means txt + img
    '''
    def __init__(self,eeg_txt_path,act_img_path,label_df_path):
        with open(eeg_txt_path, 'rb') as f:
            self.eeg_txt = pickle.load(f)
        with open(act_img_path, 'rb') as f:
            self.act_img = pickle.load(f)
        self.label = pd.read_csv(label_df_path)['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        eeg_txt_input = torch.tensor(self.eeg_txt[idx]['input_ids']) # torch.Size([1, 512])
        eeg_txt_mask = torch.tensor(self.eeg_txt[idx]['attention_mask'])
        act_img_input = torch.tensor(self.act_img[idx]).unsqueeze(0) # torch.Size([1, 1,512])
        act_img_mask = torch.tensor([1])
        label = self.label[idx]
        if pd.isnull(label):
            label = 0
        label = torch.LongTensor([label])
        return eeg_txt_input,eeg_txt_mask,act_img_input,act_img_mask,label

class MultiModalDataset_tt(Dataset):
    '''
    treat eeg as txt, act as txt; tt means txt + txt
    '''
    def __init__(self,eeg_txt_path,act_txt_path,label_df_path):
        with open(eeg_txt_path, 'rb') as f:
            self.eeg_txt = pickle.load(f)
        with open(act_txt_path, 'rb') as f:
            self.act_txt = pickle.load(f)
        self.label = pd.read_csv(label_df_path)['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        eeg_txt_input = torch.tensor(self.eeg_txt[idx]['input_ids']) # torch.Size([1, 512])
        eeg_txt_mask = torch.tensor(self.eeg_txt[idx]['attention_mask'])
        act_txt_input = torch.tensor(self.act_txt[idx]['attention_mask'])
        act_txt_mask = torch.tensor(self.act_txt[idx]['attention_mask'])
        label = self.label[idx]
        if pd.isnull(label):
            label = 0
        label = torch.LongTensor([label])
        return eeg_txt_input,eeg_txt_mask,act_txt_input,act_txt_mask,label
    
class MultiModalDataset_it(Dataset):
    '''
    treat eeg as img, act as txt; ti means img + txt
    '''
    def __init__(self,eeg_img_path,act_txt_path,label_df_path):
        with open(eeg_img_path, 'rb') as f:
            self.eeg_img = pickle.load(f)
        with open(act_txt_path, 'rb') as f:
            self.act_txt = pickle.load(f)
        self.label = pd.read_csv(label_df_path)['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        eeg_img_input = torch.tensor(self.eeg_img[idx]).unsqueeze(0) # torch.Size([1, 1,512])
        eeg_img_mask = torch.tensor([1])
        act_txt_input = torch.tensor(self.act_txt[idx]['input_ids']) # torch.Size([1, 512])
        act_txt_mask = torch.tensor(self.act_txt[idx]['attention_mask'])
        
        label = self.label[idx]
        if pd.isnull(label):
            label = 0
        label = torch.LongTensor([label])
        return eeg_img_input,eeg_img_mask,act_txt_input,act_txt_mask,label

class MultiModalDataset_ii(Dataset):
    '''
    treat eeg as img, act as img; ti means img + img
    '''
    def __init__(self,eeg_img_path,act_img_path,label_df_path):
        with open(eeg_img_path, 'rb') as f:
            self.eeg_img = pickle.load(f)
        with open(act_img_path, 'rb') as f:
            self.act_img = pickle.load(f)
        self.label = pd.read_csv(label_df_path)['label']

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        eeg_img_input = torch.tensor(self.eeg_img[idx]).unsqueeze(0) # torch.Size([1, 1,512])
        eeg_img_mask = torch.tensor([1])
        act_img_input = torch.tensor(self.act_img[idx]).unsqueeze(0) # torch.Size([1, 1,512])
        act_img_mask = torch.tensor([1])
        
        label = self.label[idx]
        if pd.isnull(label):
            label = 0
        label = torch.LongTensor([label])
        return eeg_img_input,eeg_img_mask,act_img_input,act_img_mask,label
