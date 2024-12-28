import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from base_train import TrainAndTest

"""
compare:
    DP scheme: feature-level element-wise DP dropout (our proposed method);
    DPSGD;
    Laplacian dropout with feature-level equal Laplacian noises and dropout rates (one special case of our proposed method);
    NDP-MLD (nonprivate version of our proposed method)
fix: 
    multimodal_type: "ti"
    cross_modal_type: double_stream
    txt_model: bert, txt_model_coef:"bert-base-uncased"
    img_model: clip, img_model_coef:"ViT-B/32"
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

class CompareCrossModalType(object):
    def __init__(self,
                 train_type = "compare_private_scheme",
                 multimodal_type = "ti",
                 cross_atn_type = "double_stream",
                 txt_model="bert",
                 txt_model_coef="bert-base-uncased",
                 img_model="clip",
                 img_model_coef="ViT-B/32",
                 epsilon=0.1):
        self.train_type = train_type
        self.multimodal_type = multimodal_type
        self.cross_atn_type = cross_atn_type
        self.txt_model = txt_model
        self.txt_model_coef = txt_model_coef
        self.img_model = img_model
        self.img_model_coef = img_model_coef
        self.epsilon = epsilon
        self.python_job = TrainAndTest()
        set_seed(980616)
    def LaplacianDropout(self):
        set_seed(980616)
        dp_mode = "lapacian_dropout"
        path_suffix = dp_mode + "/"
        self.python_job.train(self.train_type,path_suffix,self.multimodal_type,dp_mode,self.txt_model,self.txt_model_coef,self.img_model,self.img_model_coef,self.cross_atn_type,self.epsilon)
    def DPSGD(self):
        set_seed(980616)
        dp_mode = "DPSGD"
        path_suffix = dp_mode + "/"
        self.python_job.train(self.train_type,path_suffix,self.multimodal_type,dp_mode,self.txt_model,self.txt_model_coef,self.img_model,self.img_model_coef,self.cross_atn_type,self.epsilon)
    def LaplacianDropoutEquaWei(self):
        set_seed(980616)
        dp_mode = "lapacian_dropout_equal_weight"
        path_suffix = dp_mode + "/"
        self.python_job.train(self.train_type,path_suffix,self.multimodal_type,dp_mode,self.txt_model,self.txt_model_coef,self.img_model,self.img_model_coef,self.cross_atn_type,self.epsilon)
    def NonPrivate(self):
        set_seed(980616)
        dp_mode = "NDP"
        path_suffix = dp_mode + "/"
        self.python_job.train(self.train_type,path_suffix,self.multimodal_type,dp_mode,self.txt_model,self.txt_model_coef,self.img_model,self.img_model_coef,self.cross_atn_type,self.epsilon)
    def run(self):
        set_seed(980616)
        self.LaplacianDropout() # in demo 
        self.DPSGD()
        self.LaplacianDropoutEquaWei()
        self.NonPrivate()

if __name__ == "__main__":
    print("I'm running to compare private scheme")
    set_seed(980616)
    python_job = CompareCrossModalType()
    python_job.run()
