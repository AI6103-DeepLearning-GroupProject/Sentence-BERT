import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F


class AoECombinedLoss(nn.Module):
    """
    AoE-style objective:
      L = w_angle * L_angle + w_cl * L_cl

    - L_angle: pair-ranking objective over angle differences
    - L_cl: supervised contrastive objective (InfoNCE style with in-batch negatives)
    """

    def __init__(
        self,
        sentence_embedder,
        angle_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        angle_temperature: float = 0.05,
        contrastive_temperature: float = 0.05,
        contrastive_symmetric: bool = True,
        eps: float = 1e-8,
    ):
        super(AoECombinedLoss, self).__init__()
        self.sentence_embedder = sentence_embedder
        self.angle_weight = float(angle_weight)
        self.contrastive_weight = float(contrastive_weight)
        self.angle_temperature = float(angle_temperature)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_symmetric = bool(contrastive_symmetric)
        self.eps = float(eps)

    def _split_complex(self, embeddings: Tensor):
        if embeddings.size(1) % 2 != 0:
            raise ValueError(
                "AoECombinedLoss requires even embedding dimension, got {}".format(
                    embeddings.size(1)
                )
            )
        half = embeddings.size(1) // 2
        return embeddings[:, :half], embeddings[:, half:]

    def _angle_difference(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        a_re, a_im = self._split_complex(embeddings_a)
        b_re, b_im = self._split_complex(embeddings_b)

        # Complex dot and cross components for angle difference.
        dot = torch.sum(a_re * b_re + a_im * b_im, dim=1)
        cross = torch.sum(a_im * b_re - a_re * b_im, dim=1)
        return torch.atan2(torch.abs(cross), dot + self.eps)

    def _angle_ranking_loss(self, angle_diff: Tensor, labels: Tensor) -> Tensor:
        labels = labels.view(-1).float()
        pair_mask = labels.unsqueeze(1) > labels.unsqueeze(0)

        if torch.sum(pair_mask).item() == 0:
            return angle_diff.new_tensor(0.0)

        logits = (angle_diff.unsqueeze(1) - angle_diff.unsqueeze(0)) / self.angle_temperature
        valid_logits = logits[pair_mask]

        # log(1 + sum(exp(valid_logits))) with logsumexp stability.
        zero = valid_logits.new_zeros(1)
        return torch.logsumexp(torch.cat([zero, valid_logits], dim=0), dim=0)

    def _contrastive_loss(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        logits = torch.matmul(embeddings_a, embeddings_b.t()) / self.contrastive_temperature
        target = torch.arange(logits.size(0), device=logits.device)

        loss_ab = F.cross_entropy(logits, target)
        if not self.contrastive_symmetric:
            return loss_ab

        loss_ba = F.cross_entropy(logits.t(), target)
        return 0.5 * (loss_ab + loss_ba)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [
            self.sentence_embedder(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        if len(reps) != 2:
            raise ValueError(
                "AoECombinedLoss expects exactly 2 inputs per example, got {}".format(len(reps))
            )

        reps_a, reps_b = reps
        angle_diff = self._angle_difference(reps_a, reps_b)
        loss_angle = self._angle_ranking_loss(angle_diff, labels)
        loss_cl = self._contrastive_loss(reps_a, reps_b)
        return self.angle_weight * loss_angle + self.contrastive_weight * loss_cl
