import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from base_train import TrainAndTest

"""
compare:
    models with different initial weights:
        bert: "bert-base-uncased", "bert-base-cased"
        vit: "ViT-B/32", "ViT-B/16"
        resnet: resnet34
fix:
    multimodal_type: "ti"
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


class CompareModelIniWeight(object):
    def __init__(self,
                 train_type = "compare_model_ini_wight",
                 multimodal_type = "ti",
                 cross_atn_type="double_stream",
                 dp_mode="lapacian_dropout",
                 epsilon=0.1):
        self.train_type = train_type
        self.multimodal_type = multimodal_type
        self.cross_atn_type = cross_atn_type
        self.dp_mode = dp_mode
        self.epsilon = epsilon
        self.python_job = TrainAndTest()
        set_seed(980616)
    def run(self):
        set_seed(980616)
        for txt_model,txt_coef,img_model,img_coef in [
            ["bert","bert-base-uncased","clip","ViT-B/16"],
            ["bert","bert-base-uncased","resnet","resnet34"],
            ["bert","bert-base-cased","clip","ViT-B/32"],
            ["bert","bert-base-cased","clip","ViT-B/16"],
            ["bert","bert-base-cased","resnet","resnet34"]
        ]:
            print(txt_model,txt_coef,img_model,img_coef)
            path_suffix = txt_coef.replace("/","_").replace("-","_") + "&" + img_coef.replace("/","_").replace("-","_") +"/"
            self.python_job.train(self.train_type,path_suffix,self.multimodal_type,self.dp_mode,txt_model,txt_coef,img_model,img_coef,self.cross_atn_type,self.epsilon)

if __name__ == "__main__":
    set_seed(980616)
    print("I'm running to compare model coef choices within diff initial weights of bert, clip, and plus test for resnet")
    python_job = CompareModelIniWeight()
    python_job.run()