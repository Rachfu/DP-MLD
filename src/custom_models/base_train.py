import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MultiModalDataset_ti,MultiModalDataset_tt,MultiModalDataset_it,MultiModalDataset_ii
from models import TICA_LapDropout,TTCA_LapDropout,ITCA_LapDropout,IICA_LapDropout,TISC_LapDropout,TICA_DPSGD,TICA_NonPrivate,TISC_LapDropoutEquWeight
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
import warnings
from sklearn.metrics import f1_score
from opacus import PrivacyEngine
import time
from datetime import datetime
import random
import torch
import numpy as np
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

warnings.filterwarnings('ignore')

class TrainAndTest(object):
    def __init__(self,
                 batch_size=8,
                 learning_rate= 1e-6,
                 epochs = 50,
                 ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(980616)

    def cal_loss(self,prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    def train(self,train_type,path_suffix,multimodal_type,dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,cross_atn_type,epsilon):
        """
        multimodal_type = "ti","tt","it","ii"
        dp_mode: "lapacian_dropout"
        """
        set_seed(980616)
        batch_size = self.batch_size
        eeg_model_coef_standardized = eeg_model_coef.replace("/","_").replace("-","_")
        act_model_coef_standardized = act_model_coef.replace("/","_").replace("-","_")

        # dataloader
        if multimodal_type == "ti":
            eeg_txt_path = "data/embedding/EEG/txt/" + eeg_model + "_" + eeg_model_coef_standardized + "/"
            act_img_path = "data/embedding/act/img/" + act_model + "_" + act_model_coef_standardized + "/"
            label_path = "data/processed/"
            train_dataset = MultiModalDataset_ti(eeg_txt_path + "train.pickle",
                                                act_img_path + "train.pickle",
                                                label_path+"train_label.csv")
            test_dataset = MultiModalDataset_ti(eeg_txt_path + "test.pickle",
                                                act_img_path + "test.pickle",
                                                label_path+"test_label.csv")
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        if multimodal_type == "tt":
            eeg_txt_path = "data/embedding/EEG/txt/" + eeg_model + "_" + eeg_model_coef_standardized + "/"
            act_txt_path = "data/embedding/act/txt/" + act_model + "_" + act_model_coef_standardized + "/"
            label_path = "data/processed/"
            train_dataset = MultiModalDataset_tt(eeg_txt_path + "train.pickle",
                                                act_txt_path + "train.pickle",
                                                label_path+"train_label.csv")
            test_dataset = MultiModalDataset_tt(eeg_txt_path + "test.pickle",
                                                act_txt_path + "test.pickle",
                                                label_path+"test_label.csv")
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        if multimodal_type == "it":
            eeg_img_path = "data/embedding/EEG/img/" + eeg_model + "_" + eeg_model_coef_standardized + "/"
            act_txt_path = "data/embedding/act/txt/" + act_model + "_" + act_model_coef_standardized + "/"
            label_path = "data/processed/"
            train_dataset = MultiModalDataset_it(eeg_img_path + "train.pickle",
                                                act_txt_path + "train.pickle",
                                                label_path+"train_label.csv")
            test_dataset = MultiModalDataset_it(eeg_img_path + "test.pickle",
                                                act_txt_path + "test.pickle",
                                                label_path+"test_label.csv")
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        if multimodal_type == "ii":
            eeg_img_path = "data/embedding/EEG/img/" + eeg_model + "_" + eeg_model_coef_standardized + "/"
            act_img_path = "data/embedding/act/img/" + act_model + "_" + act_model_coef_standardized + "/"
            label_path = "data/processed/"
            train_dataset = MultiModalDataset_ii(eeg_img_path + "train.pickle",
                                                act_img_path + "train.pickle",
                                                label_path+"train_label.csv")
            test_dataset = MultiModalDataset_ii(eeg_img_path + "test.pickle",
                                                act_img_path + "test.pickle",
                                                label_path+"test_label.csv")
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            
        # model
        if cross_atn_type == "double_stream":
            if multimodal_type == "ti":
                if dp_mode == 'lapacian_dropout':
                    model = TICA_LapDropout(bert_coef = eeg_model_coef)
                if dp_mode == "DPSGD":
                    model = TICA_DPSGD(bert_coef = eeg_model_coef)
                if dp_mode == "NDP":
                    model = TICA_NonPrivate(bert_coef = eeg_model_coef)
                if dp_mode == "lapacian_dropout_equal_weight":
                    model = TISC_LapDropoutEquWeight(bert_coef = eeg_model_coef,dropout_rate=0.5) # choose as the initial weight of DP-MLD, i.e. 0.5
            if multimodal_type == "tt":
                if dp_mode == 'lapacian_dropout':
                    model = TTCA_LapDropout(bert_coef = eeg_model_coef)
            if multimodal_type == "it":
                if dp_mode == 'lapacian_dropout':
                    model = ITCA_LapDropout(bert_coef = eeg_model_coef)
            if multimodal_type == "ii":
                if dp_mode == 'lapacian_dropout':
                    model = IICA_LapDropout()    
        if cross_atn_type == "single_stream":
            if multimodal_type == "ti":
                if dp_mode == 'lapacian_dropout':
                    model = TISC_LapDropout(bert_coef = eeg_model_coef)

        # training settings
        learning_rate = self.learning_rate
        epochs = self.epochs
        model_path = "models/custom/"+ train_type +"/" + path_suffix
        log_path = "logs/" + train_type +"/" + path_suffix
        
        for path in [model_path,log_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        save_model_path = model_path + "best_f1.pickle"
        whole_log_path = log_path+"whole_record.txt"
        best_log_path = log_path+"best_record.txt"
        f1_score_best = 0.5

        # train
        if dp_mode == "lapacian_dropout":
            DP_params = [p for n, p in model.named_parameters() if 'DP' in n]
            model_params = [p for n, p in model.named_parameters() if 'DP' not in n]
            model_optimizer = Adam(model_params, lr=learning_rate)
            DP_optimizer = Adam(DP_params, lr=learning_rate)
            device = self.device
            model = model.to(device)
            # training
            for epoch in range(epochs):
                start_time = time.time()
                epoch_acc_train,epoch_loss_train,epoch_acc_test,epoch_loss_test,sample_size_train,sample_size_test = [0]*6

                model.train()
                for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(train_dataloader):
                    sample_size_train+=1
                    model.train()
                    DP_optimizer.zero_grad()
                    eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                    if multimodal_type == "ti":
                        prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask,epsilon,hard = False)
                    elif multimodal_type == "it":
                        prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input,act_mask,epsilon,hard = False)
                    elif multimodal_type == "ii":
                        prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input.to(torch.float32),act_mask,epsilon,hard = False)
                    else:
                        prediction = model(eeg_input,eeg_mask,act_input,act_mask,epsilon,hard = False)
                    loss, accuracy, _, _ = self.cal_loss(prediction,label)  
                    loss.backward()
                    DP_optimizer.step()

                    model_optimizer.zero_grad()
                    if multimodal_type == "ti":
                        prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask,epsilon,hard = True)
                    elif multimodal_type == "it":
                        prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input,act_mask,epsilon,hard = True)
                    elif multimodal_type == "ii":
                        prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input.to(torch.float32),act_mask,epsilon,hard = True)
                    else:
                        prediction = model(eeg_input,eeg_mask,act_input,act_mask,epsilon,hard = True)
                    loss, accuracy, _, _ = self.cal_loss(prediction,label) 
                    epoch_loss_train += loss.item()
                    epoch_acc_train += accuracy.item()
                    loss.backward()
                    model_optimizer.step()
                    
                prediction_all = []
                label_all = []
                model.eval()
                with torch.no_grad():
                    for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(test_dataloader):
                        sample_size_test +=1
                        eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                        if multimodal_type == "ti":
                            prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask,epsilon,hard = True)
                        elif multimodal_type == "it":
                            prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input,act_mask,epsilon,hard = True)
                        elif multimodal_type == "ii":
                            prediction = model(eeg_input.to(torch.float32),eeg_mask,act_input.to(torch.float32),act_mask,epsilon,hard = True)
                        else:
                            prediction = model(eeg_input,eeg_mask,act_input,act_mask,epsilon,hard = True)
                        loss, accuracy, pred_label_id, label_id = self.cal_loss(prediction,label)
                        prediction_all.extend(pred_label_id.cpu().numpy())
                        label_all.extend(label_id.cpu().numpy())
                        epoch_loss_test += loss.item()
                        epoch_acc_test += accuracy.item()

                f1_score_epoch = f1_score(prediction_all,label_all)
                end_time = time.time()
                time_cost = end_time-start_time
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                record = f'''Epochs: {epoch + 1}
                | Train Loss: {epoch_loss_train/sample_size_train: .3f}
                | Train Accuracy: {epoch_acc_train/sample_size_train: .3f}
                | Test Loss: {epoch_loss_test/sample_size_test: .3f}
                | Test Accuracy: {epoch_acc_test/sample_size_test: .3f}
                | f_1 Score: {f1_score_epoch: .3f}
                | Time Cost: {time_cost: .1f}
                | Record Time: {formatted_datetime} \n'''
                print(record)
                with open(whole_log_path, "a") as file:
                    file.write(record)

                if f1_score_epoch > f1_score_best:
                    torch.save(model.state_dict(), save_model_path)
                    f1_best_record = record
                    f1_score_best = f1_score_epoch
                    with open(best_log_path, "w") as file:
                        file.write(f1_best_record)
        

        if dp_mode == "DPSGD":
            # progressive training
            # epochs_progressive = 1
            # optimizer = Adam(model.parameters(), lr=learning_rate)
            # device = self.device
            # model = model.to(device)
            # for epoch in range(epochs_progressive):
            #     start_time = time.time()
            #     epoch_acc_train,epoch_loss_train,epoch_acc_test,epoch_loss_test,sample_size_train,sample_size_test = [0]*6

            #     model.train()
            #     for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(train_dataloader):
            #         sample_size_train+=1
            #         model.train()
            #         optimizer.zero_grad()
            #         eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
            #         prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask)
            #         loss, accuracy, _, _ = self.cal_loss(prediction,label)  
            #         loss.backward()
            #         optimizer.step()
            #         epoch_loss_train += loss.item()
            #         epoch_acc_train += accuracy.item()
                    
            #     prediction_all = []
            #     label_all = []
            #     model.eval()
            #     with torch.no_grad():
            #         for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(test_dataloader):
            #             sample_size_test +=1
            #             eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
            #             prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask)
            #             loss, accuracy, pred_label_id, label_id = self.cal_loss(prediction,label)
            #             prediction_all.extend(pred_label_id.cpu().numpy())
            #             label_all.extend(label_id.cpu().numpy())
            #             epoch_loss_test += loss.item()
            #             epoch_acc_test += accuracy.item()

            #     f1_score_epoch = f1_score(prediction_all,label_all)
            #     end_time = time.time()
            #     time_cost = end_time-start_time
            #     current_datetime = datetime.now()
            #     formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            #     record = f'''Epochs: {epoch + 1}
            #     | Train Loss: {epoch_loss_train/sample_size_train: .3f}
            #     | Train Accuracy: {epoch_acc_train/sample_size_train: .3f}
            #     | Test Loss: {epoch_loss_test/sample_size_test: .3f}
            #     | Test Accuracy: {epoch_acc_test/sample_size_test: .3f}
            #     | f_1 Score: {f1_score_epoch: .3f}
            #     | Time Cost: {time_cost: .1f}
            #     | Record Time: {formatted_datetime} \n'''
            #     print(record)
            #     with open(whole_log_path, "a") as file:
            #         file.write(record)

            #     if f1_score_epoch > f1_score_best:
            #         torch.save(model.state_dict(), save_model_path)
            #         f1_best_record = record
            #         f1_score_best = f1_score_epoch
            #         with open(best_log_path, "w") as file:
            #             file.write(f1_best_record)


            model.to("cpu")
            model.train()
            trainable_layers = [model.bert.encoder.layer[-1],model.bert.pooler,model.fc_layers,model.visual_encoder,model.classifier]
            # print(model)
            total_params = 0
            trainable_params = 0
            for p in model.parameters():
                p.requires_grad = False
                total_params += p.numel()

            for layer in trainable_layers:
                for p in layer.parameters():
                    p.requires_grad = True
                    trainable_params += p.numel()
            print(f"Total parameters count: {total_params:,}")
            print(f"Trainable parameters count: {trainable_params:,}")
            optimizer = Adam(model.parameters(), lr=learning_rate)
            DELTA = 1 / len(train_dataloader) # Parameter for privacy accounting. Probability of not achieving privacy guarantees
            MAX_GRAD_NORM = 0.1
            privacy_engine = PrivacyEngine()
            model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                target_delta=DELTA,
                target_epsilon=epsilon, 
                epochs=epochs,
                max_grad_norm=MAX_GRAD_NORM,
            )

            # trainable_layers = [model.fc_layers,model.classifier]
            # params_DP = []
            # for layer in trainable_layers:
            #     params_DP.extend(layer.parameters())
            # print(params_DP)
            # dp_optimizer = Adam(params_DP, lr=learning_rate)
            # privacy_engine = PrivacyEngine()
            # model, dp_optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            #     module=model,
            #     optimizer=dp_optimizer,
            #     data_loader=train_dataloader,
            #     target_delta= 1 / len(train_dataloader), # Parameter for privacy accounting. Probability of not achieving privacy guarantees
            #     target_epsilon=epsilon, 
            #     epochs=epochs,
            #     max_grad_norm=0.1,
            # )
            # print("222")
            # optimizer = Adam(model.parameters(), lr=learning_rate)
            
            # optimizer= optim.SGD(model.parameters(),lr=learning_rate)
            # privacy_engine = PrivacyEngine(
            #     model,
            #     sample_rate = batch_size/len(train_dataset),
            #     noise_multiplier = 0.1,
            #     max_grad_norm =1.0,
            # )
            # privacy_engine.attach(optimizer)
            
            device = self.device
            model = model.to(device)
            # training
            for epoch in range(epochs):
                start_time = time.time()
                epoch_acc_train,epoch_loss_train,epoch_acc_test,epoch_loss_test,sample_size_train,sample_size_test = [0]*6

                model.train()
                for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(train_dataloader):
                    sample_size_train+=1
                    model.train()
                    optimizer.zero_grad()
                    eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                    prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask)
                    loss, accuracy, _, _ = self.cal_loss(prediction,label)  
                    loss.backward()
                    optimizer.step()
                    epoch_loss_train += loss.item()
                    epoch_acc_train += accuracy.item()
                    
                prediction_all = []
                label_all = []
                model.eval()
                with torch.no_grad():
                    for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(test_dataloader):
                        sample_size_test +=1
                        eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                        prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask)
                        loss, accuracy, pred_label_id, label_id = self.cal_loss(prediction,label)
                        prediction_all.extend(pred_label_id.cpu().numpy())
                        label_all.extend(label_id.cpu().numpy())
                        epoch_loss_test += loss.item()
                        epoch_acc_test += accuracy.item()

                f1_score_epoch = f1_score(prediction_all,label_all)
                end_time = time.time()
                time_cost = end_time-start_time
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                record = f'''Epochs: {epoch + 1}
                | Train Loss: {epoch_loss_train/sample_size_train: .3f}
                | Train Accuracy: {epoch_acc_train/sample_size_train: .3f}
                | Test Loss: {epoch_loss_test/sample_size_test: .3f}
                | Test Accuracy: {epoch_acc_test/sample_size_test: .3f}
                | f_1 Score: {f1_score_epoch: .3f}
                | Time Cost: {time_cost: .1f}
                | Record Time: {formatted_datetime} \n'''
                print(record)
                with open(whole_log_path, "a") as file:
                    file.write(record)

                if f1_score_epoch > f1_score_best:
                    torch.save(model.state_dict(), save_model_path)
                    f1_best_record = record
                    f1_score_best = f1_score_epoch
                    with open(best_log_path, "w") as file:
                        file.write(f1_best_record)

        if dp_mode == "NDP":
            optimizer = Adam(model.parameters(), lr=learning_rate)
            device = self.device
            model = model.to(device)
            for epoch in range(epochs):
                start_time = time.time()
                epoch_acc_train,epoch_loss_train,epoch_acc_test,epoch_loss_test,sample_size_train,sample_size_test = [0]*6

                model.train()
                for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(train_dataloader):
                    sample_size_train+=1
                    model.train()
                    optimizer.zero_grad()
                    eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                    prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask)
                    loss, accuracy, _, _ = self.cal_loss(prediction,label)  
                    loss.backward()
                    optimizer.step()
                    epoch_loss_train += loss.item()
                    epoch_acc_train += accuracy.item()
                    
                prediction_all = []
                label_all = []
                model.eval()
                with torch.no_grad():
                    for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(test_dataloader):
                        sample_size_test +=1
                        eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                        prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask)
                        loss, accuracy, pred_label_id, label_id = self.cal_loss(prediction,label)
                        prediction_all.extend(pred_label_id.cpu().numpy())
                        label_all.extend(label_id.cpu().numpy())
                        epoch_loss_test += loss.item()
                        epoch_acc_test += accuracy.item()

                f1_score_epoch = f1_score(prediction_all,label_all)
                end_time = time.time()
                time_cost = end_time-start_time
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                record = f'''Epochs: {epoch + 1}
                | Train Loss: {epoch_loss_train/sample_size_train: .3f}
                | Train Accuracy: {epoch_acc_train/sample_size_train: .3f}
                | Test Loss: {epoch_loss_test/sample_size_test: .3f}
                | Test Accuracy: {epoch_acc_test/sample_size_test: .3f}
                | f_1 Score: {f1_score_epoch: .3f}
                | Time Cost: {time_cost: .1f}
                | Record Time: {formatted_datetime} \n'''
                print(record)
                with open(whole_log_path, "a") as file:
                    file.write(record)

                if f1_score_epoch > f1_score_best:
                    torch.save(model.state_dict(), save_model_path)
                    f1_best_record = record
                    f1_score_best = f1_score_epoch
                    with open(best_log_path, "w") as file:
                        file.write(f1_best_record)

        if dp_mode == "lapacian_dropout_equal_weight":
            print("yeah")
            optimizer = Adam(model.parameters(), lr=learning_rate)
            device = self.device
            model = model.to(device)
            for epoch in range(epochs):
                start_time = time.time()
                epoch_acc_train,epoch_loss_train,epoch_acc_test,epoch_loss_test,sample_size_train,sample_size_test = [0]*6

                model.train()
                for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(train_dataloader):
                    sample_size_train+=1
                    model.train()
                    optimizer.zero_grad()
                    eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                    prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask,epsilon)
                    loss, accuracy, _, _ = self.cal_loss(prediction,label)  
                    loss.backward()
                    optimizer.step()
                    epoch_loss_train += loss.item()
                    epoch_acc_train += accuracy.item()
                    
                prediction_all = []
                label_all = []
                model.eval()
                with torch.no_grad():
                    for eeg_input,eeg_mask,act_input,act_mask,label in tqdm(test_dataloader):
                        sample_size_test +=1
                        eeg_input,eeg_mask,act_input,act_mask,label = eeg_input.to(device), eeg_mask.to(device),act_input.to(device), act_mask.to(device), label.to(device)
                        prediction = model(eeg_input,eeg_mask,act_input.to(torch.float32),act_mask,epsilon)
                        loss, accuracy, pred_label_id, label_id = self.cal_loss(prediction,label)
                        prediction_all.extend(pred_label_id.cpu().numpy())
                        label_all.extend(label_id.cpu().numpy())
                        epoch_loss_test += loss.item()
                        epoch_acc_test += accuracy.item()

                f1_score_epoch = f1_score(prediction_all,label_all)
                end_time = time.time()
                time_cost = end_time-start_time
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                record = f'''Epochs: {epoch + 1}
                | Train Loss: {epoch_loss_train/sample_size_train: .3f}
                | Train Accuracy: {epoch_acc_train/sample_size_train: .3f}
                | Test Loss: {epoch_loss_test/sample_size_test: .3f}
                | Test Accuracy: {epoch_acc_test/sample_size_test: .3f}
                | f_1 Score: {f1_score_epoch: .3f}
                | Time Cost: {time_cost: .1f}
                | Record Time: {formatted_datetime} \n'''
                print(record)
                with open(whole_log_path, "a") as file:
                    file.write(record)

                if f1_score_epoch > f1_score_best:
                    torch.save(model.state_dict(), save_model_path)
                    f1_best_record = record
                    f1_score_best = f1_score_epoch
                    with open(best_log_path, "w") as file:
                        file.write(f1_best_record)
if __name__ == "__main__":
    set_seed(980616)
    # print("I'm running to for compare our corresponding nonprivate version NDP-MLD")
    print("I'm running to for compare Laplacian dropout with feature-level equal Laplacian noises and dropout rates")
    python_job = TrainAndTest()
    train_type = "compare_lapacian_dropout_equal_weight"
    multimodal_type = "ti"
    dp_mode  ="lapacian_dropout_equal_weight"
    eeg_model="bert"
    eeg_model_coef="bert-base-uncased"
    act_model="clip"
    act_model_coef="ViT-B/32"
    cross_atn_type="double_stream"
    epsilon=0.1
    python_job.train(train_type,multimodal_type,dp_mode,eeg_model,eeg_model_coef,act_model,act_model_coef,cross_atn_type,epsilon)

