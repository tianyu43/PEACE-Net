import os, re
import json
import random
import numpy as np
import argparse as ag
from sklearn.metrics import accuracy_score, confusion_matrix, confusion_matrix, f1_score, cohen_kappa_score, precision_score, recall_score

import torch



def seed_torch(seed=6):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    

def get_random_sample_indices(pseudo_label, n_samples = 50000):

    # 按照pseudo_label随机选取样本，正负样本平衡
    pos_idx = np.where(pseudo_label == 1)[0]
    neg_idx = np.where(pseudo_label == 0)[0]
    selected_pos_idx = np.random.choice(pos_idx, n_samples, replace=False)
    selected_neg_idx = np.random.choice(neg_idx, n_samples, replace=False)
    
    selected_idx = np.concatenate([selected_pos_idx, selected_neg_idx])
    
    np.random.shuffle(selected_idx)
    
    return selected_idx




def load_npy_data(SITE, YEAR):
    data_x = np.load(f'../0_data/{SITE}/{YEAR}/x_sr_s.npy')
    data_y = np.load(f'../0_data/{SITE}/{YEAR}/label.npy')
    return data_x, data_y


def get_parser_with_args_from_json(config_file='configs.json'):
    parser = ag.ArgumentParser(description='Training network')
    config_name = os.path.basename(config_file).split('.')[0]
    
    with open(config_file, 'r') as fin:
        configs = json.load(fin)
        parser.set_defaults(**configs)
        parser.add_argument('--config_name', default=config_name, type=str, help='configs_name')
        return parser.parse_args()  
    return None



def ensure_dir(path):
    """确保给定文件路径的上级目录存在，如果不存在就自动创建"""
    dir_path = path if os.path.isdir(path) else os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 创建目录: {dir_path}")
    else:
        print(f"✅ 目录已存在: {dir_path}")


def _assert_suffix_match(suffix, path):
    assert re.search(r"\.{}$".format(suffix), path), "suffix mismatch"

def make_parent_dir(filepath):
    parent_path = os.path.dirname(filepath)
    if not os.path.isdir(parent_path):
        try:
            os.mkdir(parent_path)
        except FileNotFoundError:
            make_parent_dir(parent_path)
            os.mkdir(parent_path)
        print("[INFO] Make new directory: '{}'".format(parent_path))
        
def save_to_npy(data, path):
    _assert_suffix_match("npy", path)
    make_parent_dir(path)
    np.save(path, data)
    print("[INFO] Save as npy: '{}'".format(path))
    
    
    

def print_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    
    print(f"Accuracy (OA): {accuracy*100:.2f}")
    #print(f"Kappa: {kappa:.4f}")
    print(f"Recall (PA): {recall*100:.2f}")
    print(f"Precision (UA): {precision*100:.2f}")
    print(f"F1 Score: {f1:.4f}")



from matplotlib import pyplot as plt
def plot_vi_ts(VI_ts, y_train, title):
    
    doy_list = [120 + i * 7 for i in range(22)]
    VI_soy = VI_ts[y_train==1]
    VI_other = VI_ts[y_train==0]

    VI_soy_mean = np.mean(VI_soy, axis=0)
    VI_soy_std = np.std(VI_soy, axis=0)

    VI_other_mean = np.mean(VI_other, axis=0)
    VI_other_std = np.std(VI_other, axis=0)

    #plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(doy_list, VI_soy_mean, label='soy', color='green',marker='o')
    plt.plot(doy_list, VI_other_mean, label='other', color='gray',marker='o')
    plt.fill_between(doy_list, VI_soy_mean-VI_soy_std, VI_soy_mean+VI_soy_std, alpha=0.2, color='green')
    plt.fill_between(doy_list, VI_other_mean-VI_other_std, VI_other_mean+VI_other_std, alpha=0.2, color='gray')


    plt.axvline(x=120, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=150, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=180, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=210, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=240, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=270, color='black', linestyle='--', linewidth=1)

    plt.legend(loc='upper right')