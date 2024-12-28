import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import clip
from torchvision.models import resnet34
from transformers import BertTokenizer
from tqdm import tqdm
import pickle
from scipy.interpolate import interp1d


# image_embedding
class TransferToImage(Dataset):
    def __init__(self,data_path,modal_type):
        self.tensor = []
        df = pd.read_csv(data_path)
        op_upsample = nn.Upsample(scale_factor=74,mode='nearest')
        pad = nn.ZeroPad2d(padding=(1,1,1,1))
        for i in range(len(df)):
            if modal_type == "act":
                # a try for an idea for transforming to image
                df_list = df.loc[i].tolist()
                df_list = df_list + [df_list[-1]]*2
                df_tensor = torch.Tensor(df_list).reshape(3,3,3).permute(2,0,1).unsqueeze(0)
                df_tensor = pad(op_upsample(df_tensor)).squeeze(0) # torch.size(3,224,224)
                self.tensor.append(df_tensor)
            if modal_type == "EEG":
                # a try for another idea for transforming to image
                df_data = df.loc[i]
                df_data = (df_data-df_data.min())/(df_data.max()-df_data.min())
                df_array = df_data.to_numpy()
                x_original = np.linspace(0,1,len(df_array))
                x_new = np.linspace(0,1,224*224)
                interpolator = interp1d(x_original,df_array,kind="linear")
                interpolated_data = interpolator(x_new)
                df_reshape = interpolated_data.reshape((224,224))
                df_img = np.stack([df_reshape]*3,axis=0)
                df_tensor = torch.from_numpy(df_img).float() # torch.size(3,224,224)
                self.tensor.append(df_tensor)
    def __len__(self):
        return len(self.tensor)
    def __getitem__(self, index):
        return torch.FloatTensor(self.tensor[index])
    
class GetEmbedding(object):
    def __init__(self,modal_list,data_train_test_list):
        self.modal_list = modal_list
        self.data_train_test_list = data_train_test_list
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def img_encode(self,data_path,modal_type,process_model,coef_model):
        """
        note: an example to use gpu to encode img
        """
        device = self.device
        
        if process_model == "clip":
            model, preprocess = clip.load(coef_model)
            model = model.to(device)
            dataset = TransferToImage(data_path,modal_type)
            dataloader = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=False)
            encode=[]
            with torch.no_grad():
                for data in tqdm(dataloader):
                    encode.append(model.encode_image(data.to(device)))
            encode_array = np.array(torch.cat(encode,dim=0).detach().cpu())
        if process_model == "resnet":
            if coef_model == "resnet34":
                model = resnet34()
                model.load_state_dict(torch.load("models/pretrained/resnet34.pth"))
                model.eval()
                model.fc = torch.nn.Identity()
                dataset = TransferToImage(data_path,modal_type)
                dataloader = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=False)
                encode=[]
                with torch.no_grad():
                    for data in tqdm(dataloader):
                        encode.append(model(data))
                encode_array = np.array(torch.cat(encode,dim=0).detach())
        print(encode_array.shape)
        del encode
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return encode_array
    
    def get_img_encode(self,img_process_coef_model_list):
        for modal in self.modal_list:
            for data_train_test in self.data_train_test_list:
                data_path = "data/processed/" + data_train_test + "_" + modal +".csv"
                for process_model,coef_model in img_process_coef_model_list:
                    img_encode_array = self.img_encode(data_path,modal,process_model,coef_model)
                    print(modal, process_model, coef_model, data_train_test)
                    coef_model_path = coef_model.replace("/","_").replace("-","_")
                    save_path = "data/embedding"+modal+"/"+"img/"+ process_model + "_" + coef_model_path + "/" 
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    with open(save_path + data_train_test +".pickle", 'wb') as f:
                        img_encode_array.dump(f)

    def text_encode(self,data_path,process_model,coef_model):
        """
        note: an example to use cpu to encode text
        """
        df = pd.read_csv(data_path)
        text_embedding=[]
        if process_model == "bert":
            tokenizer = BertTokenizer.from_pretrained(coef_model)
            for i in range(len(df)):
                sentence = " ".join([str(j) for j in df.loc[i].tolist()])
                text_encode = tokenizer(sentence,padding="max_length",truncation=True,max_length=512)
                text_embedding.append(text_encode)
        return text_embedding

    def get_text_encode(self,txt_process_coef_model_list):
        for modal in self.modal_list:
            for data_train_test in self.data_train_test_list:
                data_path = "data/processed/" + data_train_test + "_" + modal +".csv"
                for process_model,coef_model in txt_process_coef_model_list:
                    txt_embedding = self.text_encode(data_path,process_model,coef_model)
                    print(modal, process_model, coef_model, data_train_test)
                    coef_model_path = coef_model.replace("/","_").replace("-","_")
                    save_path = "data/embedding"+modal+"/"+"txt/"+ process_model + "_" + coef_model_path + "/" 
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    with open(save_path + data_train_test +".pickle", 'wb') as f:
                        pickle.dump(txt_embedding,f)
    
    def run(self,img_process_coef_model_list,txt_process_coef_model_list):
        self.get_img_encode(img_process_coef_model_list)
        self.get_text_encode(txt_process_coef_model_list)

if __name__ == '__main__':
    modal_list= ["act","EEG"]
    data_train_test_list = ["train","test"]
    img_process_coef_model_list = [['clip','ViT-B/16'],['clip','ViT-B/32'],['resnet','resnet34']]
    txt_process_coef_model_list = [['bert','bert-base-uncased'],['bert','bert-base-cased']]

    python_job = GetEmbedding(modal_list,data_train_test_list)
    python_job.run(img_process_coef_model_list,txt_process_coef_model_list)
