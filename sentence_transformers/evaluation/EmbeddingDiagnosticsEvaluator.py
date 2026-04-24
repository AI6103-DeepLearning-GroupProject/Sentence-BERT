from . import SentenceEvaluator
from torch.utils.data import DataLoader

import csv
import json
import logging
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from ..util import batch_to_device


class EmbeddingDiagnosticsEvaluator(SentenceEvaluator):
    """
    Compute lightweight embedding-geometry diagnostics to detect collapse-like
    behavior such as vanishing per-dimension variance or extremely low effective rank.

    The evaluator writes append-only records to ``embedding_stat.json``.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", show_progress_bar: bool = None, pairwise_sample_size: int = 1024):
        self.dataloader = dataloader
        self.name = name or "embedding_diagnostics"
        if show_progress_bar is None:
            show_progress_bar = (
                logging.getLogger().getEffectiveLevel() == logging.INFO
                or logging.getLogger().getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pairwise_sample_size = int(pairwise_sample_size)

    @staticmethod
    def _resolve_output_dir(output_path: str) -> str:
        normalized = os.path.normpath(output_path)
        base_name = os.path.basename(normalized)
        if base_name in {"model", "eval_current"}:
            return os.path.dirname(normalized)
        return normalized

    def _compute_stats(self, embeddings: np.ndarray):
        embedding_count, embedding_dim = embeddings.shape

        norms = np.linalg.norm(embeddings, axis=1)
        feature_std = embeddings.std(axis=0)

        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        if embedding_count > 1:
            covariance = np.matmul(centered.T, centered) / float(embedding_count - 1)
            eigvals = np.linalg.eigvalsh(covariance)
            eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
            eigval_sum = float(eigvals.sum())
        else:
            eigvals = np.zeros((embedding_dim,), dtype=np.float64)
            eigval_sum = 0.0

        if eigval_sum > 0.0:
            probs = eigvals / eigval_sum
            nonzero = probs > 0
            entropy = float(-(probs[nonzero] * np.log(probs[nonzero])).sum())
            effective_rank = float(np.exp(entropy))
            top_explained_variance_ratio = float(eigvals.max() / eigval_sum)
        else:
            effective_rank = 0.0
            top_explained_variance_ratio = 0.0

        sample_size = min(self.pairwise_sample_size, embedding_count)
        if sample_size >= 2:
            if embedding_count > sample_size:
                rng = np.random.RandomState(42)
                sample_indices = rng.choice(embedding_count, size=sample_size, replace=False)
                sample = embeddings[sample_indices]
            else:
                sample = embeddings

            sample_norms = np.linalg.norm(sample, axis=1, keepdims=True)
            sample_norms = np.clip(sample_norms, a_min=1e-12, a_max=None)
            sample = sample / sample_norms
            cosine_matrix = np.matmul(sample, sample.T)
            off_diag_mask = ~np.eye(sample.shape[0], dtype=bool)
            pairwise_cosines = cosine_matrix[off_diag_mask]
            pairwise_cosine_mean = float(pairwise_cosines.mean())
            pairwise_cosine_std = float(pairwise_cosines.std())
            pairwise_cosine_max = float(pairwise_cosines.max())
        else:
            pairwise_cosine_mean = 0.0
            pairwise_cosine_std = 0.0
            pairwise_cosine_max = 0.0

        return {
            "embedding_count": int(embedding_count),
            "embedding_dim": int(embedding_dim),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "feature_std_mean": float(feature_std.mean()),
            "feature_std_min": float(feature_std.min()),
            "feature_std_max": float(feature_std.max()),
            "near_zero_var_dims_1e-05": int((feature_std < 1e-5).sum()),
            "near_zero_var_dims_1e-04": int((feature_std < 1e-4).sum()),
            "effective_rank": effective_rank,
            "top_explained_variance_ratio": top_explained_variance_ratio,
            "pairwise_cosine_mean": pairwise_cosine_mean,
            "pairwise_cosine_std": pairwise_cosine_std,
            "pairwise_cosine_max": pairwise_cosine_max,
        }

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        all_embeddings: List[np.ndarray] = []

        self.dataloader.collate_fn = model.smart_batching_collate

        iterator = self.dataloader
        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Embedding Diagnostics")

        for batch in iterator:
            features, _ = batch_to_device(batch, self.device)
            with torch.no_grad():
                batch_embeddings = [
                    model(sent_features)["sentence_embedding"].to("cpu").numpy()
                    for sent_features in features
                ]
            all_embeddings.extend(batch_embeddings)

        if not all_embeddings:
            raise ValueError("No embeddings collected for diagnostics evaluator '{}'".format(self.name))

        merged_embeddings = np.concatenate(all_embeddings, axis=0)
        stats = self._compute_stats(merged_embeddings)
        record = {
            "evaluator_name": self.name,
            "epoch": int(epoch),
            "steps": int(steps),
        }
        record.update(stats)

        logging.info(
            "Embedding diagnostics (%s): n=%d dim=%d norm_mean=%.6f feature_std_mean=%.6f eff_rank=%.2f top_ratio=%.6f",
            self.name,
            record["embedding_count"],
            record["embedding_dim"],
            record["norm_mean"],
            record["feature_std_mean"],
            record["effective_rank"],
            record["top_explained_variance_ratio"],
        )

        if output_path is not None:
            resolved_output_dir = self._resolve_output_dir(output_path)
            os.makedirs(resolved_output_dir, exist_ok=True)
            json_path = os.path.join(resolved_output_dir, "embedding_stat.json")
            csv_path = os.path.join(resolved_output_dir, "embedding_stat.csv")

            if os.path.isfile(json_path):
                with open(json_path, encoding="utf-8") as f_in:
                    payload = json.load(f_in)
            else:
                payload = {"records": []}

            payload.setdefault("records", []).append(record)
            with open(json_path, "w", encoding="utf-8") as f_out:
                json.dump(payload, f_out, indent=2)

            csv_headers = [
                "evaluator_name",
                "epoch",
                "steps",
                "embedding_count",
                "embedding_dim",
                "norm_mean",
                "norm_std",
                "norm_min",
                "norm_max",
                "feature_std_mean",
                "feature_std_min",
                "feature_std_max",
                "near_zero_var_dims_1e-05",
                "near_zero_var_dims_1e-04",
                "effective_rank",
                "top_explained_variance_ratio",
                "pairwise_cosine_mean",
                "pairwise_cosine_std",
                "pairwise_cosine_max",
            ]
            csv_exists = os.path.isfile(csv_path)
            with open(csv_path, "a" if csv_exists else "w", newline="", encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=csv_headers)
                if not csv_exists:
                    writer.writeheader()
                writer.writerow({header: record.get(header) for header in csv_headers})

        return 0.0
