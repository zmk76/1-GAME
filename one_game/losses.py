# coding=utf-8
import torch
import numpy as np
import torch.nn.functional as F

def compute_info_bpr_loss(a_embeddings, b_embeddings, pos_edges, num_negs=300, reduction='mean', hard_negs=None):

    if isinstance(pos_edges, list):
        pos_edges = np.array(pos_edges)

    device = a_embeddings.device

    a_indices = pos_edges[:, 0]
    b_indices = pos_edges[:, 1]

    if isinstance(pos_edges, torch.Tensor):
        num_pos_edges = pos_edges.size(0)
    else:
        num_pos_edges = len(pos_edges)

    if hard_negs is None:
        num_b = b_embeddings.size(0)
        neg_b_indices = torch.randint(0, num_b, [num_pos_edges, num_negs]).to(device)
    else:
        a_hard_negs = hard_negs[b_indices]
        neg_b_columns = torch.randint(0, hard_negs.size(1), [num_pos_edges, num_negs]).to(device)
        neg_b_rows = torch.arange(num_pos_edges).unsqueeze(1).expand(-1, num_negs).to(device)
        neg_b_indices = a_hard_negs[neg_b_rows, neg_b_columns]

    embedded_a = a_embeddings[a_indices]
    embedded_b = b_embeddings[b_indices]
    embedded_neg_b = b_embeddings[neg_b_indices]

    embedded_combined_b = torch.cat([embedded_b.unsqueeze(1), embedded_neg_b], 1)
    logits = (embedded_combined_b @ embedded_a.unsqueeze(-1)).squeeze(-1)

    info_bpr_loss = F.cross_entropy(logits, torch.zeros([num_pos_edges], dtype=torch.int64).to(device), reduction=reduction)

    return info_bpr_loss

def compute_bpr_loss(a_embeddings, b_embeddings, pos_edges, reduction='mean'):
    return compute_info_bpr_loss(a_embeddings, b_embeddings, pos_edges, num_negs=1, reduction=reduction)

def compute_l2_loss(params):
    l2_loss = 0.0
    for param in params:
        l2_loss += param.pow(2).sum() * 0.5
    return l2_loss