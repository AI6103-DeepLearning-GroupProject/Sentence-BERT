import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class Pooling(nn.Module):
    """Performs pooling (max, mean, or attention) on token embeddings."""

    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_attention_tokens: bool = False,
                 pooling_mode_attention_fusion: str = 'concat',
                 ):
        super(Pooling, self).__init__()

        self.config_keys = [
            'word_embedding_dimension',
            'pooling_mode_cls_token',
            'pooling_mode_mean_tokens',
            'pooling_mode_max_tokens',
            'pooling_mode_mean_sqrt_len_tokens',
            'pooling_mode_attention_tokens',
            'pooling_mode_attention_fusion',
        ]

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_attention_tokens = pooling_mode_attention_tokens
        self.pooling_mode_attention_fusion = pooling_mode_attention_fusion

        if self.pooling_mode_attention_fusion not in {'concat', 'gate', 'residual'}:
            raise ValueError('Unknown pooling_mode_attention_fusion: {}'.format(self.pooling_mode_attention_fusion))

        if self.pooling_mode_attention_tokens:
            self.attention = nn.Linear(word_embedding_dimension, 1)
            # Start from near-mean behavior for stable early training.
            nn.init.zeros_(self.attention.weight)
            nn.init.zeros_(self.attention.bias)
            if self.pooling_mode_attention_fusion == 'gate':
                self.attention_gate = nn.Linear(2 * word_embedding_dimension, word_embedding_dimension)
                nn.init.zeros_(self.attention_gate.weight)
                nn.init.zeros_(self.attention_gate.bias)
            else:
                self.attention_gate = None

            if self.pooling_mode_attention_fusion == 'residual':
                self.attention_residual_scale = nn.Parameter(torch.zeros(1))
            else:
                self.attention_residual_scale = None
        else:
            self.attention = None
            self.attention_gate = None
            self.attention_residual_scale = None

        pooling_mode_multiplier = int(pooling_mode_cls_token) + int(pooling_mode_max_tokens)
        if pooling_mode_mean_tokens:
            pooling_mode_multiplier += 1
        if pooling_mode_mean_sqrt_len_tokens:
            pooling_mode_multiplier += 1
        if pooling_mode_attention_tokens:
            if not (pooling_mode_mean_tokens and self.pooling_mode_attention_fusion in {'gate', 'residual'}):
                pooling_mode_multiplier += 1
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        input_mask = features['input_mask']

        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)

        if self.pooling_mode_max_tokens:
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings_masked = token_embeddings.masked_fill(input_mask_expanded == 0, -1e9)
            max_over_time = torch.max(token_embeddings_masked, 1)[0]
            output_vectors.append(max_over_time)

        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present.
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        if self.pooling_mode_attention_tokens:
            attn_scores = self.attention(token_embeddings).squeeze(-1)
            # Use a finite negative value for masked positions to keep fp16/fp32 stable.
            attn_scores = attn_scores.masked_fill(input_mask == 0, -1e4)
            attn_probs = torch.softmax(attn_scores, dim=1)
            # Re-normalize on valid tokens only for numerical robustness.
            attn_probs = attn_probs * input_mask.float()
            attn_probs = attn_probs / torch.clamp(attn_probs.sum(dim=1, keepdim=True), min=1e-9)
            attn_probs_expanded = attn_probs.unsqueeze(-1).expand(token_embeddings.size())
            attention_vector = torch.sum(token_embeddings * attn_probs_expanded, dim=1)

            if self.pooling_mode_mean_tokens and self.pooling_mode_attention_fusion in {'gate', 'residual'}:
                mean_vector = output_vectors[-1]
                if self.pooling_mode_attention_fusion == 'gate':
                    gate_input = torch.cat([mean_vector, attention_vector], dim=1)
                    gate = torch.sigmoid(self.attention_gate(gate_input))
                    output_vectors[-1] = gate * attention_vector + (1.0 - gate) * mean_vector
                else:
                    output_vectors[-1] = mean_vector + self.attention_residual_scale * (attention_vector - mean_vector)
            else:
                output_vectors.append(attention_vector)

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        if self.attention is not None:
            torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        if 'pooling_mode_attention_tokens' not in config:
            config['pooling_mode_attention_tokens'] = False
        if 'pooling_mode_attention_fusion' not in config:
            config['pooling_mode_attention_fusion'] = 'concat'

        model = Pooling(**config)
        weights_path = os.path.join(input_path, 'pytorch_model.bin')
        if model.attention is not None and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

        return model
