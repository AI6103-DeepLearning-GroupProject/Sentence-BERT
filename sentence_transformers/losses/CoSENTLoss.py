import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F


class CoSENTLoss(nn.Module):
    def __init__(self, model, scale: float = 20.0):
        super(CoSENTLoss, self).__init__()
        self.model = model
        self.scale = float(scale)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        rep_a, rep_b = reps

        scores = F.cosine_similarity(rep_a, rep_b) * self.scale
        scores = scores[:, None] - scores[None, :]

        labels = labels.view(-1)
        label_mask = labels[:, None] < labels[None, :]
        scores = scores - (1 - label_mask.float()) * 1e12
        scores = torch.cat([torch.zeros(1, device=scores.device), scores.view(-1)], dim=0)
        return torch.logsumexp(scores, dim=0)
