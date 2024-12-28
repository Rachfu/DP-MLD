import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from base_train import TrainAndTest

"""
    A demo of our proposed DP-MLD method

    multimodal_type: "ti"
    cross_modal_type: double_stream
    txt_model: bert, txt_model_coef:"bert-base-uncased"
    img_model: clip, img_model_coef:"ViT-B/32"
    DP scheme: feature-level element-wise DP dropout
    privacy budget: epsilon = 0.1
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

class Demo(object):
    def __init__(self,
                 train_type = "demo",
                 multimodal_type = "ti",
                 cross_atn_type = "double_stream",
                 txt_model="bert",
                 txt_model_coef="bert-base-uncased",
                 img_model="clip",
                 img_model_coef="ViT-B/32",
                 dp_mode = "lapacian_dropout",
                 epsilon=0.1):
        self.train_type = train_type
        self.multimodal_type = multimodal_type
        self.cross_atn_type = cross_atn_type
        self.txt_model = txt_model
        self.txt_model_coef = txt_model_coef
        self.img_model = img_model
        self.img_model_coef = img_model_coef
        self.dp_mode = dp_mode
        self.epsilon = epsilon
        self.python_job = TrainAndTest()
        set_seed(980616)
    def demo(self):
        set_seed(980616)
        path_suffix="DPMLD/"
        self.python_job.train(self.train_type,path_suffix,self.multimodal_type,self.dp_mode,self.txt_model,self.txt_model_coef,self.img_model,self.img_model_coef,self.cross_atn_type,self.epsilon)

if __name__ == "__main__":
    print("I'm running a demo of DP-MLD")
    set_seed(980616)
    python_job = Demo()
    python_job.demo()
