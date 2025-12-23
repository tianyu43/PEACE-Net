import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import numpy as np
import torch
from tqdm import tqdm
from configs import configs
from torch.utils.data import DataLoader
from dataset.dataset import DynamicLabelDataset_pred
from model.PEACE_Net import PEACE_Net
from utils.helper import seed_torch
from utils.tools import save_to_npy, print_metrics

SITE = 'site_usa_ia'
YEAR = 2023
i = 0

DATA_SR_PATH = f'F:/reginal/S2_data_IOWA/grid_0/x_sr.npy'
DATA_LABEL_PATH = f'F:/reginal/S2_data_IOWA/grid_0/label.npy'
OUT_PATH = f"./result/{SITE}/out_peace-net_{YEAR}.npy"
os.makedirs(OUT_PATH, exist_ok=True)


seed_torch(configs.seed)
SEED = configs.seed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====================================== 1.load data ======================================

print(f'================== {SITE}__{YEAR}__grid_{i} ==================')

data_x = np.load(DATA_SR_PATH)
data_y = np.load(DATA_LABEL_PATH)

print(DEVICE)
print('data_x', data_x.shape)
print('data_Y',data_y.shape) 

# ====================================== 2.model set up ======================================

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


model.load_state_dict(torch.load(f'./train/models/PEACE_-net/{SITE}_grid_{i}_{YEAR}/peace-model-best.pth'))
model.to(DEVICE)
model.eval()

# ====================================== 3.dataset ======================================
dataset_pred = DynamicLabelDataset_pred(data_x, data_y)
loader_pred =  DataLoader(dataset_pred,
                          batch_size = int(pow(2, 15)), 
                          shuffle=False, drop_last=False,
                          )
# ====================================== 4.pred ======================================
all_features_encoder_train = []
all_labels_t = []
y_hard = []

with torch.no_grad():
    for i, (input_x, y_t) in enumerate(tqdm(loader_pred)):
        input_x = input_x.to(DEVICE)
        y_t = y_t.to(DEVICE)

        class_out, _ = model.predict(input_x)  # (B, D)

        all_labels_t.append(y_t.cpu())
        
        y_out = torch.argmax(class_out, dim=1)
        y_hard.append(y_out.cpu())

all_labels_t = torch.cat(all_labels_t, dim=0).numpy()  # (N,)
y_hard = torch.cat(y_hard, dim=0).numpy()  # (N,)

# ====================================== 5.save & acc ======================================
save_to_npy(y_hard, OUT_PATH)

print(f' ***grid-{i} pred_finished*** save output')
print_metrics(data_y, y_hard)


#清空缓存
del model, dataset_pred, loader_pred, data_x, data_y, all_labels_t, y_hard
gc.collect()
torch.cuda.empty_cache()

