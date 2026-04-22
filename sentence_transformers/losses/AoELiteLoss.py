import torch
from torch import nn, Tensor
from typing import Iterable, Dict


class AoELiteLoss(nn.Module):
    """
    AoE-lite objective for STS pair training:
      L = cosine_weight * L_cos + angle_weight * L_angle

    - L_cos: CoSENT-style ranking loss over cosine similarity scores
    - L_angle: CoSENT-style ranking loss over angle-derived similarity scores
    """

    def __init__(
        self,
        model,
        cosine_weight: float = 1.0,
        angle_weight: float = 0.02,
        cosine_tau: float = 20.0,
        angle_tau: float = 20.0,
        eps: float = 1e-8,
    ):
        super(AoELiteLoss, self).__init__()
        self.model = model
        self.cosine_weight = float(cosine_weight)
        self.angle_weight = float(angle_weight)
        self.cosine_tau = float(cosine_tau)
        self.angle_tau = float(angle_tau)
        self.eps = float(eps)

    def _split_complex(self, embeddings: Tensor):
        if embeddings.size(1) % 2 != 0:
            raise ValueError(
                "AoELiteLoss requires even embedding dimension, got {}".format(
                    embeddings.size(1)
                )
            )

        half = embeddings.size(1) // 2
        return embeddings[:, :half], embeddings[:, half:]

    def _angle_difference(self, embeddings_a: Tensor, embeddings_b: Tensor) -> Tensor:
        a_re, a_im = self._split_complex(embeddings_a)
        b_re, b_im = self._split_complex(embeddings_b)

        # atan2 keeps the angle computation numerically stable near zero.
        dot = torch.sum(a_re * b_re + a_im * b_im, dim=1)
        cross = torch.sum(a_im * b_re - a_re * b_im, dim=1)
        return torch.atan2(torch.abs(cross), dot + self.eps)

    def _ranking_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        labels = labels.view(-1).float()
        pair_mask = labels.unsqueeze(1) < labels.unsqueeze(0)

        if torch.sum(pair_mask).item() == 0:
            return scores.new_tensor(0.0)

        diffs = scores.unsqueeze(1) - scores.unsqueeze(0)
        valid_logits = diffs[pair_mask]

        zero = valid_logits.new_zeros(1)
        return torch.logsumexp(torch.cat([zero, valid_logits], dim=0), dim=0)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        if len(reps) != 2:
            raise ValueError(
                "AoELiteLoss expects exactly 2 inputs per example, got {}".format(len(reps))
            )

        reps_a, reps_b = reps
        cosine_scores = self.cosine_tau * torch.cosine_similarity(reps_a, reps_b)
        # Smaller angle difference means more similar, so negate it before ranking.
        angle_scores = -self.angle_tau * self._angle_difference(reps_a, reps_b)

        loss_cosine = self._ranking_loss(cosine_scores, labels)
        loss_angle = self._ranking_loss(angle_scores, labels)
        return self.cosine_weight * loss_cosine + self.angle_weight * loss_angle
