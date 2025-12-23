import os, sys, gc, random
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataset import DynamicLabelDataset, DynamicLabelDataset_pred
from utils.loss import sim_loss, binary_soft_ce_loss
from utils.VI_cal import get_pseudo_label_with_threshold



def seed_torch(seed=128):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True



def get_pseudo_label(x_train, y_train, site, m='max'):

    st = 12 if site in ['site_bz', 'site_ag'] else 16 if site in ['site_hl'] else 14
    
    custom_thresholds = {
        'gwcci':   (0.17, 0.17),
        'gwcci_2': (0.56, 0.56),
        'smci':    (5.15, 5.15)
    }

    VI1_label = get_pseudo_label_with_threshold(
        x_train, vi='gwcci', 
        thresholds = custom_thresholds, 
        start_time = 12,
        end_time= 12+6,
        m = m
        )
    print('---gwcci伪标签精度:',np.mean(VI1_label[VI1_label != -1] == y_train[VI1_label != -1]))
    print('伪标签 1个数:', np.sum(VI1_label == 1))
    print('伪标签 0个数:', np.sum(VI1_label == 0))

    VI2_label = get_pseudo_label_with_threshold(
        x_train, 
        vi='gwcci_2', 
        thresholds = custom_thresholds, 
        start_time = st,
        end_time= st+4,
        m = m
        )
    print('---gwcci_2伪标签精度:',np.mean(VI2_label[VI2_label != -1] == y_train[VI2_label != -1]))
    print('伪标签 1个数:', np.sum(VI2_label == 1))
    print('伪标签 0个数:', np.sum(VI2_label == 0))

    VI3_label = get_pseudo_label_with_threshold(
        x_train, 
        vi='smci', 
        thresholds = custom_thresholds, 
        start_time = st,
        end_time= st+4,
        m = m
        )
    print('---smci伪标签精度:',np.mean(VI3_label[VI3_label != -1] == y_train[VI3_label != -1]))
    print('伪标签 1个数:', np.sum(VI3_label == 1))
    print('伪标签 0个数:', np.sum(VI3_label == 0))

    pseudo_label = VI1_label + VI2_label + VI3_label
    pseudo_label[pseudo_label < 2] = 0
    pseudo_label[pseudo_label >= 2] = 1
    print('pseudo_label精度：',np.mean(pseudo_label[pseudo_label != -1] == y_train[pseudo_label != -1]))
    print('伪标签1个数：', np.sum(pseudo_label == 1))
    print('伪标签0个数：', np.sum(pseudo_label == 0))
    print('==============================')
    
    return pseudo_label
    





def built_model(model_name, configs):
    '''
    built model based on model name and configurations
    '''
    
    if model_name == 'PEACE_Net':
        from model.PEACE_Net import PEACE_Net
        model = PEACE_Net(
            SEED = configs.seed,
            input_feature_size = configs.in_channels,
            seq_len = configs.seq_len,
            pe_tau = configs.pe_tau,
            d_model = configs.d_model,
            dim_feedforward = configs.dim_feedforward,
            projection_dim = configs.projection_dim,
            nhead = configs.nhead,
            num_layers = configs.num_layers,
            n_classes = configs.num_classes,
            dropout = configs.dropout,
            pooling = configs.pooling
        )

    elif model_name == 'transformer':
        from model.transformer import TransformerClassifier
        model = TransformerClassifier(
            SEED = configs.seed,
            input_feature_size = configs.in_channels,
            seq_len = configs.seq_len,
            d_model = configs.d_model,
            dim_feedforward = configs.dim_feedforward,
            projection_dim = configs.projection_dim,
            nhead = configs.nhead,
            num_layers = configs.num_layers,
            n_classes = configs.num_classes,
            dropout = configs.dropout,
            pooling = configs.pooling
        )
    
    elif model_name == 'DCM':
        from model.DCM import DCM
        model = DCM(
            SEED = configs.seed,
            input_feature_size = configs.in_channels,
            hidden_size = configs.LSTM_hidden_size,
            num_layers = configs.LSTM_num_layers,
            bidirectional = configs.bidirectional,
            dropout = configs.LSTM_dropout,
            num_classes = configs.num_classes
        )
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model
