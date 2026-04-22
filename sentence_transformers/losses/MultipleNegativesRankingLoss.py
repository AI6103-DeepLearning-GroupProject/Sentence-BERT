import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, sentence_embedder, scale: float = 20.0):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.sentence_embedder = sentence_embedder
        self.scale = float(scale)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.sentence_embedder(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        if len(reps) < 2:
            raise ValueError("MultipleNegativesRankingLoss expects at least anchor and positive embeddings")

        reps_a = reps[0]
        reps_b = reps[1]
        hard_negatives = reps[2:] if len(reps) > 2 else None
        return self.multiple_negatives_ranking_loss(reps_a, reps_b, hard_negatives)

    # Multiple Negatives Ranking Loss
    # Paper: https://arxiv.org/pdf/1705.00652.pdf
    #   Efficient Natural Language Response Suggestion for Smart Reply
    #   Section 4.4
    def multiple_negatives_ranking_loss(self, embeddings_a: Tensor, embeddings_b: Tensor, hard_negative_embeddings=None):
        """
        Compute the loss over a batch with two or more embeddings per example.

        Each pair is a positive example. The negative examples are all other embeddings in embeddings_b with each embedding
        in embedding_a. Optional hard-negative embedding blocks are appended to the candidate matrix, while the positive
        label remains the diagonal item in the first candidate block.

        See the paper for more information: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        :param embeddings_a:
            Tensor of shape (batch_size, embedding_dim)
        :param embeddings_b:
            Tensor of shape (batch_size, embedding_dim)
        :return:
            The scalar loss
        """
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
