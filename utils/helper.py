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
from utils.tools import print_metrics


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




def get_pseudo_label_with_tw(x_train, y_train, start_t_gwcci, start_t_gwcci_2, start_t_smci, m='max'):

    custom_thresholds = {
        'gwcci':   (0.17, 0.17),
        'gwcci_2': (0.56, 0.56),
        'smci':    (3.25, 3.25)
    }

    VI1_label = get_pseudo_label_with_threshold(
        x_train, vi='gwcci', 
        thresholds = custom_thresholds, 
        start_time = start_t_gwcci,
        end_time= start_t_gwcci+7,
        m = m
        )
    print('---gwcci_acc:')
    print_metrics(y_train, VI1_label)

    VI2_label = get_pseudo_label_with_threshold(
        x_train, 
        vi='gwcci_2', 
        thresholds = custom_thresholds, 
        start_time = start_t_gwcci_2,
        end_time= start_t_gwcci_2 + 5,
        m = m
        )
    print('---gwcci_2_acc:')
    print_metrics(y_train, VI2_label)

    VI3_label = get_pseudo_label_with_threshold(
        x_train, 
        vi='smci', 
        thresholds = custom_thresholds, 
        start_time = start_t_smci,
        end_time= start_t_smci + 5,
        m = m
        )
    print('---smci_acc:')
    print_metrics(y_train, VI3_label)

    pseudo_label = VI1_label + VI2_label + VI3_label
    pseudo_label[pseudo_label < 2] = 0
    pseudo_label[pseudo_label >= 2] = 1
    print('---pseudo_label_acc:')
    print_metrics(y_train, pseudo_label)
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
            pe_tau = configs.pe_tau,
            SEED = configs.seed,
            input_feature_size = configs.in_channels,
            seq_len = configs.seq_len,
            d_model = configs.d_model,
            dim_feedforward = configs.dim_feedforward,
            #projection_dim = configs.projection_dim,
            nhead = configs.nhead,
            num_layers = configs.num_layers,
            n_classes = configs.num_classes,
            dropout = configs.dropout,
            #pooling = configs.pooling
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
