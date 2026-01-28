import torch
import torch.nn as nn
import torch.nn.functional as F


def sim_loss(features, labels):
    """
    Contrastive similarity loss 
    Args:
        features: (B, d), L2-normalized feature embeddings
        labels: (B,)
    """
    device = features.device
    B = features.shape[0]
    if labels.ndim != 1:
        labels = torch.argmax(labels, dim=1)
    # Cosine similarity matrix (features should be normalized)
    sim_matrix = torch.matmul(features, features.T)  #(B, B)
    # Label mask
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  #(B, B)
    diag_mask = 1 - torch.eye(B, device=device)
    # Similarity Loss
    sim_pos = (1 - sim_matrix) * mask * diag_mask
    sim_neg = (1 + sim_matrix) * (1 - mask) * diag_mask
    
    sim_loss = (sim_pos.sum() + sim_neg.sum()) / (B * (B - 1))
    return sim_loss



def sim_loss_with_margin(features, labels, margin=1):
    """
    Contrastive similarity loss with margin.
    Args:
        features: torch.Tensor of shape (B, d), L2-normalized feature embeddings
        labels: torch.Tensor of shape (B, C) or (B,), class labels or one-hot
        margin: float, margin for inter-class separation
    """
    device = features.device
    B = features.shape[0]
    if labels.ndim != 1:
        labels = torch.argmax(labels, dim=1)
    # Cosine similarity matrix (features should be normalized)
    sim_matrix = torch.matmul(features, features.T)  # (B, B), cosine similarities
    # Label mask
    labels = labels.contiguous().view(-1, 1)  # (B, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # (B, B), 1 for same class, 0 for different
    # Positive pairs: encourage similarity → loss = 1 - cos
    sim_pos = (1 - sim_matrix) * mask
    # Negative pairs: enforce margin → loss = max(0, cos - m)
    sim_neg = F.relu(sim_matrix - margin) * (1 - mask)
    # Exclude diagonal elements (i == j), as they are self-similarity (always 1)
    diag_mask = 1 - torch.eye(B, device=device)
    sim_pos = sim_pos * diag_mask
    sim_neg = sim_neg * diag_mask
    # Normalize by number of valid pairs (excluding diagonal)
    loss = (sim_pos.sum() + sim_neg.sum()) / (B * (B - 1))
    
    return loss




def binary_soft_ce_loss(logits, target_probs, pos_weight=None, reduction='mean'):

    return F.binary_cross_entropy_with_logits(
        logits, target_probs, pos_weight=pos_weight, reduction=reduction
    )