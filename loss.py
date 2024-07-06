import torch.nn.functional as F
import torch


def nce_loss(anchor_emb, pos_emb, neg_emb):
    pos_scores = F.cosine_similarity(anchor_emb, pos_emb)
    neg_scores = F.cosine_similarity(anchor_emb, neg_emb)

    pos_scores = torch.log(torch.sigmoid(pos_scores))
    neg_scores = torch.log(1 - torch.sigmoid(neg_scores))

    loss = - torch.mean(pos_scores + neg_scores)

    return loss
