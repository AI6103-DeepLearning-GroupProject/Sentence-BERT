import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F


class MultipleNegativesRankingUniRegLoss(nn.Module):
    """
    MultipleNegativesRankingLoss with a hyperspherical uniformity regularizer.

    The task loss keeps the original in-batch negative training signal. The
    regularizer is applied to L2-normalized sentence embeddings, matching the
    cosine-similarity geometry used by STS evaluation.
    """

    def __init__(
        self,
        sentence_embedder,
        scale: float = 20.0,
        uniformity_weight: float = 1e-3,
        uniformity_t: float = 2.0,
        eps: float = 1e-8,
    ):
        super(MultipleNegativesRankingUniRegLoss, self).__init__()
        self.sentence_embedder = sentence_embedder
        self.scale = float(scale)
        self.uniformity_weight = float(uniformity_weight)
        self.uniformity_t = float(uniformity_t)
        self.eps = float(eps)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [
            self.sentence_embedder(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        if len(reps) < 2:
            raise ValueError("MultipleNegativesRankingUniRegLoss expects at least anchor and positive embeddings")

        reps_a = reps[0]
        reps_b = reps[1]
        hard_negatives = reps[2:] if len(reps) > 2 else None

        ranking_loss = self._multiple_negatives_ranking_loss(reps_a, reps_b, hard_negatives)
        uniformity_loss = self._uniformity_loss(reps)
        return ranking_loss + self.uniformity_weight * uniformity_loss

    def _multiple_negatives_ranking_loss(self, embeddings_a: Tensor, embeddings_b: Tensor, hard_negative_embeddings=None):
        candidate_embeddings = [embeddings_b]
        if hard_negative_embeddings:
            candidate_embeddings.extend(hard_negative_embeddings)

        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        candidate_embeddings = [
            F.normalize(candidate_embedding, p=2, dim=1)
            for candidate_embedding in candidate_embeddings
        ]

        scores = torch.matmul(embeddings_a, torch.cat(candidate_embeddings, dim=0).t()) * self.scale
        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores, labels)

    def _uniformity_loss(self, embeddings):
        normalized = [F.normalize(embedding, p=2, dim=1) for embedding in embeddings]
        all_embeddings = torch.cat(normalized, dim=0)

        if all_embeddings.size(0) < 2:
            return all_embeddings.new_tensor(0.0)

        cosine_scores = torch.matmul(all_embeddings, all_embeddings.t())
        distances = torch.clamp(2.0 - 2.0 * cosine_scores, min=0.0)
        mask = 1.0 - torch.eye(all_embeddings.size(0), device=all_embeddings.device)
        exp_distances = torch.exp(-self.uniformity_t * distances) * mask
        return torch.log(exp_distances.sum() / mask.sum() + self.eps)
