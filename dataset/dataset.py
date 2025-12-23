import os
import sys
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.utils.data import Dataset



class DynamicLabelDataset(Dataset):
    def __init__(self, features, pseudo_h_labels, true_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.true_labels = torch.tensor(true_labels, dtype=torch.long)
        self.pseudo_h_labels = torch.stack([
            torch.tensor([1.0, 0.0]) if label == 0 else
            torch.tensor([0.0, 1.0]) if label == 1 else
            torch.tensor([1.0, 0.0]) for label in pseudo_h_labels
        ])
        #self.pseudo_h_labels = torch.nn.functional.one_hot(torch.tensor(pseudo_h_labels, dtype=torch.long)).float()
        self.pseudo_s_labels = torch.zeros_like(self.pseudo_h_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y_p = self.pseudo_s_labels[idx] if self.pseudo_s_labels.sum() > 0 else self.pseudo_h_labels[idx]
        y_t = self.true_labels[idx]
        return x, y_p, y_t
    
    
    def update_soft_labels(self, old_soft_labels, new_soft_labels, alpha):
            self.pseudo_s_labels = alpha * new_soft_labels + (1 - alpha) * old_soft_labels
            
            







class DynamicLabelDataset_pred(Dataset):
    def __init__(self, features, true_labels):
        self.features = features
        self.true_labels = true_labels
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y_t = self.true_labels[idx]
        return x, y_t