import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from base_train import TrainAndTest

"""
compare:
    multimodal_type: "ti", "tt", "it", "ii"
fix: 
    txt_model: bert, txt_model_coef:"bert-base-uncased"
    img_model: clip, img_model_coef:"ViT-B/32"
    cross_modal_type: double_stream
    DP scheme: feature-level element-wise DP dropout
    privacy budget: epsilon = 0.1
"""
import torch
import numpy as np
import random
def set_seed(seed):
    # 设置 PyTorch 的随机种子
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # # 设置 NumPy 的随机种子
    # np.random.seed(seed)

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
                 train_type = "compare_modal",
                 txt_model="bert",
                 txt_model_coef="bert-base-uncased",
                 img_model="clip",
                 img_model_coef="ViT-B/32",
                 cross_atn_type="double_stream",
                 dp_mode="lapacian_dropout",
                 epsilon=0.1):
        self.train_type = train_type
        self.txt_model = txt_model
        self.txt_model_coef = txt_model_coef
        self.img_model = img_model
        self.img_model_coef = img_model_coef
        self.cross_atn_type = cross_atn_type
        self.dp_mode = dp_mode
        self.epsilon = epsilon
        self.python_job = TrainAndTest()
        set_seed(980616)
    def test_ti(self):
        set_seed(980616)
        multimodal_type = "ti"
        path_suffix = multimodal_type + "/"
        eeg_model = self.txt_model
        eeg_model_coef = self.txt_model_coef
        act_model = self.img_model
        act_model_coef = self.img_model_coef
        self.python_job.train(self.train_type,path_suffix,multimodal_type,self.dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,self.cross_atn_type,self.epsilon)
    def test_tt(self):
        set_seed(980616)
        multimodal_type = "tt"
        path_suffix = multimodal_type + "/"
        eeg_model = self.txt_model
        eeg_model_coef = self.txt_model_coef
        act_model = self.txt_model
        act_model_coef = self.txt_model_coef
        self.python_job.train(self.train_type,path_suffix,multimodal_type,self.dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,self.cross_atn_type,self.epsilon)
    def test_it(self):
        set_seed(980616)
        multimodal_type = "it"
        path_suffix = multimodal_type + "/"
        eeg_model = self.img_model
        eeg_model_coef = self.img_model_coef
        act_model = self.txt_model
        act_model_coef = self.txt_model_coef
        self.python_job.train(self.train_type,path_suffix,multimodal_type,self.dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,self.cross_atn_type,self.epsilon)
    def test_ii(self):
        set_seed(980616)
        multimodal_type = "ii"
        path_suffix = multimodal_type + "/"
        eeg_model = self.img_model
        eeg_model_coef = self.img_model_coef
        act_model = self.img_model
        act_model_coef = self.img_model_coef
        self.python_job.train(self.train_type,path_suffix,multimodal_type,self.dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,self.cross_atn_type,self.epsilon)
    def run(self):
        set_seed(980616)
        self.test_ti() # In demo model
        self.test_tt()
        self.test_it()
        self.test_ii()

if __name__ == "__main__":
    print("I'm running to compare modal choices within ti,tt,it,ii...")
    set_seed(980616)
    python_job = CompareModal()
    python_job.run()
