import torch
from torch import Tensor
from torch import nn
from typing import Dict
import os
import json


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding.
    This layer also allows to use the CLS token if it is returned by the underlying word
    embedding model. You can concatenate multiple poolings together.
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_attention_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = [
            'word_embedding_dimension',
            'pooling_mode_cls_token',
            'pooling_mode_mean_tokens',
            'pooling_mode_max_tokens',
            'pooling_mode_mean_sqrt_len_tokens',
            'pooling_mode_attention_tokens'
        ]

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_attention_tokens = pooling_mode_attention_tokens

        if self.pooling_mode_attention_tokens:
            self.attention = nn.Linear(word_embedding_dimension, 1)
        else:
            self.attention = None

        pooling_mode_multiplier = sum([
            pooling_mode_cls_token,
            pooling_mode_max_tokens,
            pooling_mode_mean_tokens,
            pooling_mode_mean_sqrt_len_tokens,
            pooling_mode_attention_tokens
        ])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        input_mask = features['input_mask']

        # Pooling strategy
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

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
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
            attn_scores = attn_scores.masked_fill(input_mask == 0, -1e9)
            attn_probs = torch.softmax(attn_scores, dim=1)
            attn_probs_expanded = attn_probs.unsqueeze(-1).expand(token_embeddings.size())
            output_vectors.append(torch.sum(token_embeddings * attn_probs_expanded, dim=1))

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

        model = Pooling(**config)
        weights_path = os.path.join(input_path, 'pytorch_model.bin')
        if model.attention is not None and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

        return model
