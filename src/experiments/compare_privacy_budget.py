import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from base_train import TrainAndTest

"""
compare:
    privacy budget: epsilon = np.logspace(np.log10(0.01), np.log10(5.0), 20)
fix: 
    multimodal_type: "ti"
    txt_model: bert, txt_model_coef:"bert-base-uncased"
    img_model: clip, img_model_coef:"ViT-B/32"
    cross_modal_type: double_stream
    DP scheme: feature-level element-wise DP dropout
"""
import torch
import numpy as np
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(980616)

class CompareModal(object):
    def __init__(self,
                 train_type = "compare_privacy_budget",
                 multimodal_type = "ti",
                 txt_model="bert",
                 txt_model_coef="bert-base-uncased",
                 img_model="clip",
                 img_model_coef="ViT-B/32",
                 cross_atn_type="double_stream",
                 dp_mode="lapacian_dropout"):
        self.train_type = train_type
        self.multimodal_type = multimodal_type
        self.txt_model = txt_model
        self.txt_model_coef = txt_model_coef
        self.img_model = img_model
        self.img_model_coef = img_model_coef
        self.cross_atn_type = cross_atn_type
        self.dp_mode = dp_mode
        self.python_job = TrainAndTest()
        set_seed(980616)
    def run_eps_list(self):
        set_seed(980616)
        epsilon_list = np.logspace(np.log10(0.01), np.log10(5.0), 20)
        epsilon_list = np.around(epsilon_list, decimals=3)
        for epsilon in epsilon_list:
            path_suffix = "eps_list/" + str(epsilon)+'/'
            self.python_job.train(self.train_type,path_suffix,self.multimodal_type,self.dp_mode,self.txt_model,self.txt_model_coef,self.img_model,self.img_model_coef,self.cross_atn_type,epsilon)
    def run_representative_list(self):
        set_seed(980616)
        epsilon_list = [0.01,0.1,1.0]
        for epsilon in epsilon_list:
            path_suffix = "eps_representative/" + str(epsilon)+'/'
            self.python_job.train(self.train_type,path_suffix,self.multimodal_type,self.dp_mode,self.txt_model,self.txt_model_coef,self.img_model,self.img_model_coef,self.cross_atn_type,epsilon)



if __name__ == "__main__":
    print("I'm running with different privacy budgets")
    set_seed(980616)
    python_job = CompareModal()
    python_job.run_eps_list()
    python_job.run_representative_list()
