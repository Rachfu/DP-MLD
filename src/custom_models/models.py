import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer,TransformerEncoder,TransformerEncoderLayer
from transformers import BertModel
from tqdm import tqdm
import warnings
from sklearn.metrics import f1_score
import os
from opacus import PrivacyEngine
import numpy as np
import torch.nn.functional as F


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

class TICA_LapDropout(nn.Module):
    """
    --- Our mainly proposed model in paper ---
    Ti: treat eeg as txt, act as img; ti means txt + img
    CA: cross-attention for cross modal feature
    LapDropout: proposed feature-level Laplacian dropout
    """
    def __init__(self,bert_coef):
        """
        bert_coef: 
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        self.DP = nn.parameter.Parameter(torch.zeros(1, 768 * 3))
        self.noiser = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        
    def forward(self, eeg_txt_input,eeg_txt_mask,act_img_input,act_img_mask,epsilon,hard):
        device = self.device
        eps = torch.tensor(epsilon)
        eeg_txt_semantics_embedding, eeg_txt_feature = self.bert(input_ids= eeg_txt_input, 
                                                                 attention_mask=eeg_txt_mask,
                                                                 return_dict=False) #768
        act_img_embedding = self.visual_encoder(act_img_input)
        act_img_feature = act_img_embedding.squeeze(1)
        cross_attn_result = self.multi_head_decoder(tgt=act_img_embedding.permute(1,0,2), 
                                                    memory=eeg_txt_semantics_embedding.permute(1,0,2),
                                                    tgt_key_padding_mask=act_img_mask==0, 
                                                    memory_key_padding_mask=eeg_txt_mask==0)
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_txt_feature, act_img_feature,cross_attn_result),dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)
        w = F.sigmoid(self.DP)
        noise = self.noiser.sample(feature.shape).view(*feature.shape).to(device)
        eps_hat = 1/(((eps.exp() - w) / (1 - w)).log())  # fix
        feature = feature + noise * eps_hat
        mask = F.gumbel_softmax(torch.stack((w, 1 - w)).repeat(1, feature.shape[0], 1), 
                                hard=hard, dim=0)
        feature = (feature * mask).sum(0)
        feature = self.fc_layers(feature)
        prediction = self.classifier(feature)
        return prediction

class TTCA_LapDropout(nn.Module):
    def __init__(self,bert_coef):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        self.DP = nn.parameter.Parameter(torch.zeros(1, 768 * 3))
        self.noiser = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        
    def forward(self, eeg_txt_input,eeg_txt_mask,act_txt_input,act_txt_mask,epsilon,hard):
        device = self.device
        eps = torch.tensor(epsilon)
        eeg_txt_semantics_embedding, eeg_txt_feature = self.bert(input_ids= eeg_txt_input, 
                                                                 attention_mask=eeg_txt_mask,
                                                                 return_dict=False) #768
        act_txt_semantics_embedding, act_txt_feature = self.bert(input_ids= act_txt_input, 
                                                                 attention_mask=act_txt_mask,
                                                                 return_dict=False) #768
        
        cross_attn_result = self.multi_head_decoder(tgt=act_txt_semantics_embedding.permute(1,0,2), 
                                                    memory=eeg_txt_semantics_embedding.permute(1,0,2))
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_txt_feature, act_txt_feature,cross_attn_result),dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)
        w = F.sigmoid(self.DP)
        noise = self.noiser.sample(feature.shape).view(*feature.shape).to(device)
        eps_hat = 1/(((eps.exp() - w) / (1 - w)).log())  # fix
        feature = feature + noise * eps_hat
        mask = F.gumbel_softmax(torch.stack((w, 1 - w)).repeat(1, feature.shape[0], 1), 
                                hard=hard, dim=0)
        feature = (feature * mask).sum(0)
        feature = self.fc_layers(feature)
        prediction = self.classifier(feature)
        return prediction

class ITCA_LapDropout(nn.Module):
    def __init__(self,bert_coef):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        self.DP = nn.parameter.Parameter(torch.zeros(1, 768 * 3))
        self.noiser = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        
    def forward(self, eeg_img_input,eeg_img_mask,act_txt_input,act_txt_mask,epsilon,hard):
        device = self.device
        eps = torch.tensor(epsilon)
        eeg_img_embedding = self.visual_encoder(eeg_img_input)
        eeg_img_feature = eeg_img_embedding.squeeze(1)
        act_txt_semantics_embedding, act_txt_feature = self.bert(input_ids= act_txt_input, 
                                                                 attention_mask=act_txt_mask,
                                                                 return_dict=False) #768
        cross_attn_result = self.multi_head_decoder(tgt=eeg_img_embedding.permute(1,0,2), 
                                                    memory=act_txt_semantics_embedding.permute(1,0,2),
                                                    tgt_key_padding_mask=eeg_img_mask==0, 
                                                    memory_key_padding_mask=act_txt_mask==0)
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_img_feature, act_txt_feature,cross_attn_result),dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)
        w = F.sigmoid(self.DP)
        noise = self.noiser.sample(feature.shape).view(*feature.shape).to(device)
        eps_hat = 1/(((eps.exp() - w) / (1 - w)).log())  # fix
        feature = feature + noise * eps_hat
        mask = F.gumbel_softmax(torch.stack((w, 1 - w)).repeat(1, feature.shape[0], 1), 
                                hard=hard, dim=0)
        feature = (feature * mask).sum(0)
        feature = self.fc_layers(feature)
        prediction = self.classifier(feature)
        return prediction

class IICA_LapDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        self.DP = nn.parameter.Parameter(torch.zeros(1, 768 * 3))
        self.noiser = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        
    def forward(self, eeg_img_input,eeg_img_mask,act_img_input,act_img_mask,epsilon,hard):
        device = self.device
        eps = torch.tensor(epsilon)
        eeg_img_embedding = self.visual_encoder(eeg_img_input)
        eeg_img_feature = eeg_img_embedding.squeeze(1)
        act_img_embedding = self.visual_encoder(act_img_input)
        act_img_feature = act_img_embedding.squeeze(1)
        cross_attn_result = self.multi_head_decoder(tgt=eeg_img_embedding.permute(1,0,2), 
                                                    memory=act_img_embedding.permute(1,0,2))
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_img_feature, act_img_feature,cross_attn_result),dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)
        w = F.sigmoid(self.DP)
        noise = self.noiser.sample(feature.shape).view(*feature.shape).to(device)
        eps_hat = 1/(((eps.exp() - w) / (1 - w)).log())  # fix
        feature = feature + noise * eps_hat
        mask = F.gumbel_softmax(torch.stack((w, 1 - w)).repeat(1, feature.shape[0], 1), 
                                hard=hard, dim=0)
        feature = (feature * mask).sum(0)
        feature = self.fc_layers(feature)
        prediction = self.classifier(feature)
        return prediction

class TISC_LapDropout(nn.Module):
    """
    Ti: treat eeg as txt, act as img; ti means txt + img
    SC: single attention for cross modal feature
    LapDropout: proposed feature-level Laplacian dropout
    """
    def __init__(self,bert_coef):
        """
        bert_coef: 
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_encoderlayer = TransformerEncoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_encoder = TransformerEncoder(self.multi_head_encoderlayer,num_layers=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        self.DP = nn.parameter.Parameter(torch.zeros(1, 768 * 3))
        self.noiser = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        
    def forward(self, eeg_txt_input,eeg_txt_mask,act_img_input,act_img_mask,epsilon,hard):
        device = self.device
        eps = torch.tensor(epsilon)
        eeg_txt_semantics_embedding, eeg_txt_feature = self.bert(input_ids= eeg_txt_input, 
                                                                 attention_mask=eeg_txt_mask,
                                                                 return_dict=False) #768
        act_img_embedding = self.visual_encoder(act_img_input)
        act_img_feature = act_img_embedding.squeeze(1)
        eeg_txt_semantics_embedding_mean = eeg_txt_semantics_embedding.mean(dim=1).unsqueeze(1)
        concat_attn_embedding = torch.cat((eeg_txt_semantics_embedding_mean,act_img_embedding),dim=1)
        concat_attn_embedding = concat_attn_embedding.permute(1,0,2)
        concat_attn_result = self.multi_head_encoder(concat_attn_embedding).mean(dim=0) # torch.Size([8, 768])
        feature_concat = torch.cat((eeg_txt_feature, act_img_feature,concat_attn_result),dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)
        w = F.sigmoid(self.DP)
        noise = self.noiser.sample(feature.shape).view(*feature.shape).to(device)
        eps_hat = 1/(((eps.exp() - w) / (1 - w)).log())  # fix
        feature = feature + noise * eps_hat
        mask = F.gumbel_softmax(torch.stack((w, 1 - w)).repeat(1, feature.shape[0], 1), 
                                hard=hard, dim=0)
        feature = (feature * mask).sum(0)
        feature = self.fc_layers(feature)
        prediction = self.classifier(feature)
        return prediction

class TICA_DPSGD(nn.Module):
    """
    Ti: treat eeg as txt, act as img; ti means txt + img
    CA: cross-attention for cross modal feature
    DPSGD: DP achieved with perturbation on gradients
    """
    def __init__(self,bert_coef):
        """
        bert_coef: 
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_coef)
        self.visual_encoder = nn.Linear(512, 768)
        self.fc_layers = nn.Sequential(
            nn.Linear(2*768, 2*768),
            nn.ReLU(),
            nn.Linear(2*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        
    def forward(self, eeg_txt_input,eeg_txt_mask,act_img_input,act_img_mask):
        eeg_txt_semantics_embedding, eeg_txt_feature = self.bert(input_ids= eeg_txt_input, 
                                                                 attention_mask=eeg_txt_mask,
                                                                 return_dict=False) #768
        act_img_embedding = self.visual_encoder(act_img_input)
        act_img_feature = act_img_embedding.squeeze(1)
        feature_concat = torch.cat((eeg_txt_feature, act_img_feature),dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)
        feature = self.fc_layers(feature)
        prediction = self.classifier(feature)
        return prediction

class TICA_NonPrivate(nn.Module):
    """
    --- Our mainly proposed model in paper ---
    Ti: treat eeg as txt, act as img; ti means txt + img
    CA: cross-attention for cross modal feature
    NonPrivate: nonprivate version
    """
    def __init__(self,bert_coef):
        """
        bert_coef: 
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        
    def forward(self, eeg_txt_input,eeg_txt_mask,act_img_input,act_img_mask):
        eeg_txt_semantics_embedding, eeg_txt_feature = self.bert(input_ids= eeg_txt_input, 
                                                                 attention_mask=eeg_txt_mask,
                                                                 return_dict=False) #768
        act_img_embedding = self.visual_encoder(act_img_input)
        act_img_feature = act_img_embedding.squeeze(1)
        cross_attn_result = self.multi_head_decoder(tgt=act_img_embedding.permute(1,0,2), 
                                                    memory=eeg_txt_semantics_embedding.permute(1,0,2),
                                                    tgt_key_padding_mask=act_img_mask==0, 
                                                    memory_key_padding_mask=eeg_txt_mask==0)
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_txt_feature, act_img_feature,cross_attn_result),dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)
        feature = self.fc_layers(feature)
        prediction = self.classifier(feature)
        return prediction
    
class TISC_LapDropoutEquWeight(nn.Module):
    """
    Ti: treat eeg as txt, act as img; ti means txt + img
    SC: single attention for cross modal feature
    LapDropoutEquWeight: feature-level equally weightted Laplacian noise + dropout; an example case included in our 
                         proposed DP-MLD; but weaker privacy protection
    """
    def __init__(self,bert_coef,dropout_rate):
        """
        bert_coef: 
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_coef)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = nn.Linear(512, 768)
        bert_output_size = 768
        self.multi_head_decoderlayer = TransformerDecoderLayer(d_model=bert_output_size, nhead=12)
        self.multi_head_decoder = TransformerDecoder(self.multi_head_decoderlayer, num_layers=3)
        self.fc_layers = nn.Sequential(
            nn.Linear(3*768, 3*768),
            nn.ReLU(),
            nn.Linear(3*768, 768),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(768, 2)
        self.dropout_rate = dropout_rate 
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, eeg_txt_input,eeg_txt_mask,act_img_input,act_img_mask,epsilon):
        device = self.device
        eeg_txt_semantics_embedding, eeg_txt_feature = self.bert(input_ids= eeg_txt_input, 
                                                                 attention_mask=eeg_txt_mask,
                                                                 return_dict=False) #768
        act_img_embedding = self.visual_encoder(act_img_input)
        act_img_feature = act_img_embedding.squeeze(1)
        cross_attn_result = self.multi_head_decoder(tgt=act_img_embedding.permute(1,0,2), 
                                                    memory=eeg_txt_semantics_embedding.permute(1,0,2),
                                                    tgt_key_padding_mask=act_img_mask==0, 
                                                    memory_key_padding_mask=eeg_txt_mask==0)
        cross_attn_result = cross_attn_result.permute(1,0,2).mean(dim=1) #768
        feature_concat = torch.cat((eeg_txt_feature, act_img_feature,cross_attn_result),dim=1)
        feature_min = torch.min(feature_concat, dim=-1, keepdims=True)[0]
        feature_max = torch.max(feature_concat, dim=-1, keepdims=True)[0]
        feature = (feature_concat - feature_min) / (feature_max - feature_min)

        feature = self.dropout(feature)        
        dropout_rate = self.dropout_rate
        eps_hat = 1/(np.log(((np.exp(epsilon) - dropout_rate) / (1 - dropout_rate))))
        lap_sigma=1/eps_hat
        m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([lap_sigma]))
        noise = m.sample([feature.shape[0]]).to(device)
        feature += noise.view(-1, 1)
        feature = self.fc_layers(feature)
        prediction = self.classifier(feature)
        return prediction