import os
import sys
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
from torch.utils.data import Dataset



class DynamicLabelDataset(Dataset):
    def __init__(self, features, pseudo_h_labels, true_labels, share_memory=False):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.true_labels = torch.tensor(true_labels, dtype=torch.long)
        self.pseudo_h_labels = torch.stack([
            torch.tensor([1.0, 0.0]) if label == 0 else
            torch.tensor([0.0, 1.0]) if label == 1 else
            torch.tensor([1.0, 0.0]) for label in pseudo_h_labels
        ])
        #self.pseudo_h_labels = torch.nn.functional.one_hot(torch.tensor(pseudo_h_labels, dtype=torch.long)).float()
        self.pseudo_s_labels = torch.zeros_like(self.pseudo_h_labels, dtype=torch.float32)
        self._soft_ready = torch.zeros(1, dtype=torch.uint8)

        if share_memory:
            self.pseudo_h_labels.share_memory_()
            self.pseudo_s_labels.share_memory_()
            self._soft_ready.share_memory_()

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y_p = self.pseudo_s_labels[idx] if self._soft_ready.item() > 0 else self.pseudo_h_labels[idx]
        y_t = self.true_labels[idx]
        return x, y_p, y_t
    
    
    def update_soft_labels(self, old_soft_labels, new_soft_labels, alpha):
            old_cpu = old_soft_labels.detach().cpu()
            new_cpu = new_soft_labels.detach().cpu()
            self.pseudo_s_labels.copy_(old_cpu).mul_(1 - alpha).add_(new_cpu, alpha=alpha)
            self._soft_ready[0] = 1
            
            



class MemmapDatasetPred(Dataset):
    def __init__(self, feat_path, label_path):
        self.feat_path = feat_path
        self.label_path = label_path
        tmp = np.load(feat_path, mmap_mode="r")
        self.n = tmp.shape[0]
        del tmp
        self._feat = None
        self._lab = None

    def _open(self):
        if self._feat is None:
            self._feat = np.load(self.feat_path, mmap_mode="r")
            self._lab  = np.load(self.label_path, mmap_mode="r")

    def __len__(self): return self.n

    def __getitem__(self, idx):
        self._open()
        x = self._feat[idx]  # numpy view
        y = self._lab[idx]
        return torch.from_numpy(x), torch.as_tensor(y)
