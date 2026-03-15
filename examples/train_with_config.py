"""
Train Sentence-Transformer experiments from a YAML config file.

This script keeps the original v0.2.4 training intentions (same readers, losses,
and evaluators) while adding reproducible seeding and centralized parameter management.
"""
import argparse
import csv
import logging
import math
import os
from datetime import datetime

import yaml
from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, SentenceTransformer, SentencesDataset, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
from sentence_transformers.readers import NLIDataReader, STSDataReader, TripletReader
from sentence_transformers.util import set_seed


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Sentence-Transformer experiment from config.yaml")
    parser.add_argument("--config", type=str, default=os.path.join("examples", "config.yaml"), help="Path to YAML config file")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name inside config.experiments")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    return parser.parse_args()


def _load_config(config_path: str, experiment_name: str):
    with open(config_path, encoding="utf-8") as f_in:
        loaded = yaml.safe_load(f_in) or {}

    global_cfg = loaded.get("global", {})
    experiments = loaded.get("experiments", {})
    if experiment_name not in experiments:
        raise ValueError("Experiment '{}' not found in {}".format(experiment_name, config_path))

    exp_cfg = experiments[experiment_name]
    merged = dict(global_cfg)
    for key, value in exp_cfg.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value

    return merged


def _build_output_path(exp_name: str, config: dict):
    output_cfg = config.get("output", {})
    if "path" in output_cfg:
        return output_cfg["path"]

    root = output_cfg.get("root", "output")
    prefix = output_cfg.get("name_prefix", exp_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(root, "{}-{}".format(prefix, timestamp))


def _build_model(model_cfg: dict):
    pretrained_model_name = model_cfg.get("pretrained_model_name")
    if pretrained_model_name:
        return SentenceTransformer(pretrained_model_name)

    encoder_type = model_cfg.get("encoder_type", "BERT")
    model_name = model_cfg["model_name"]
    max_seq_length = model_cfg.get("max_seq_length", 128)
    do_lower_case = model_cfg.get("do_lower_case", True)
    pooling_cfg = model_cfg.get("pooling", {})

    if not hasattr(models, encoder_type):
        raise ValueError("Unknown encoder_type: {}".format(encoder_type))

    encoder_cls = getattr(models, encoder_type)
    word_embedding_model = encoder_cls(
        model_name,
        max_seq_length=max_seq_length,
        do_lower_case=do_lower_case
    )

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=pooling_cfg.get("cls_token", False),
        pooling_mode_max_tokens=pooling_cfg.get("max_tokens", False),
        pooling_mode_mean_tokens=pooling_cfg.get("mean_tokens", True),
        pooling_mode_mean_sqrt_len_tokens=pooling_cfg.get("mean_sqrt_len_tokens", False),
    )

    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def _parse_csv_quoting(quoting_value):
    if isinstance(quoting_value, int):
        return quoting_value
    if isinstance(quoting_value, str) and hasattr(csv, quoting_value):
        return getattr(csv, quoting_value)
    return csv.QUOTE_NONE


def _build_task_components(config: dict, model: SentenceTransformer):
    task_cfg = config["task"]
    training_cfg = config["training"]
    data_cfg = config["data"]

    batch_size = training_cfg.get("batch_size", 16)
    task_type = task_cfg["type"]

    if task_type == "nli_softmax":
        nli_reader = NLIDataReader(data_cfg["nli_path"])
        sts_reader = STSDataReader(
            data_cfg["sts_path"],
            normalize_scores=data_cfg.get("normalize_scores", True),
        )

        train_data = SentencesDataset(nli_reader.get_examples(data_cfg.get("nli_train_file", "train.gz")), model=model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        train_loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=nli_reader.get_num_labels(),
        )

        dev_data = SentencesDataset(sts_reader.get_examples(data_cfg.get("sts_dev_file", "sts-dev.csv")), model=model)
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
        evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

        test_data = SentencesDataset(sts_reader.get_examples(data_cfg.get("sts_test_file", "sts-test.csv")), model=model)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        test_evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
        return train_dataloader, train_loss, evaluator, test_evaluator

    if task_type == "sts_cosine":
        sts_reader = STSDataReader(
            data_cfg["sts_path"],
            normalize_scores=data_cfg.get("normalize_scores", True),
        )

        train_data = SentencesDataset(sts_reader.get_examples(data_cfg.get("sts_train_file", "sts-train.csv")), model=model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(model=model)

        dev_data = SentencesDataset(sts_reader.get_examples(data_cfg.get("sts_dev_file", "sts-dev.csv")), model=model)
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
        evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

        test_data = SentencesDataset(sts_reader.get_examples(data_cfg.get("sts_test_file", "sts-test.csv")), model=model)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        test_evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
        return train_dataloader, train_loss, evaluator, test_evaluator

    if task_type == "triplet":
        triplet_reader = TripletReader(
            data_cfg["triplet_path"],
            s1_col_idx=data_cfg.get("s1_col_idx", 0),
            s2_col_idx=data_cfg.get("s2_col_idx", 1),
            s3_col_idx=data_cfg.get("s3_col_idx", 2),
            has_header=data_cfg.get("has_header", False),
            delimiter=data_cfg.get("delimiter", "\t"),
            quoting=_parse_csv_quoting(data_cfg.get("quoting", "QUOTE_NONE")),
        )

        train_data = SentencesDataset(triplet_reader.get_examples(data_cfg.get("train_file", "train.csv")), model=model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        train_loss = losses.TripletLoss(model=model)

        dev_data = SentencesDataset(
            triplet_reader.get_examples(data_cfg.get("dev_file", "validation.csv"), data_cfg.get("dev_max_examples", 0)),
            model=model,
        )
        dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
        evaluator = TripletEvaluator(dev_dataloader)

        test_data = SentencesDataset(triplet_reader.get_examples(data_cfg.get("test_file", "test.csv")), model=model)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        test_evaluator = TripletEvaluator(test_dataloader)
        return train_dataloader, train_loss, evaluator, test_evaluator

    raise ValueError("Unknown task.type: {}".format(task_type))


def main():
    args = _parse_args()
    config = _load_config(args.config, args.experiment)

    if args.seed is not None:
        config["seed"] = args.seed

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    seed = config.get("seed", 42)
    deterministic = config.get("deterministic", True)
    set_seed(seed, deterministic=deterministic)
    logging.info("Seed set to %d (deterministic=%s)", seed, deterministic)

    model = _build_model(config["model"])
    train_dataloader, train_loss, evaluator, test_evaluator = _build_task_components(config, model)

    training_cfg = config["training"]
    num_epochs = training_cfg.get("num_epochs", 1)
    warmup_steps = training_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_ratio = training_cfg.get("warmup_ratio", 0.1)
        warmup_steps = int(math.ceil(len(train_dataloader) * num_epochs * warmup_ratio))

    output_path = _build_output_path(args.experiment, config)
    logging.info("Output path: %s", output_path)
    logging.info("Warmup steps: %s", warmup_steps)

    fit_kwargs = {
        "train_objectives": [(train_dataloader, train_loss)],
        "evaluator": evaluator,
        "epochs": num_epochs,
        "scheduler": training_cfg.get("scheduler", "WarmupLinear"),
        "warmup_steps": warmup_steps,
        "weight_decay": training_cfg.get("weight_decay", 0.01),
        "evaluation_steps": training_cfg.get("evaluation_steps", 1000),
        "output_path": output_path,
        "save_best_model": training_cfg.get("save_best_model", True),
        "max_grad_norm": training_cfg.get("max_grad_norm", 1),
        "fp16": training_cfg.get("fp16", False),
        "fp16_opt_level": training_cfg.get("fp16_opt_level", "O1"),
        "seed": seed,
        "deterministic": deterministic,
    }

    optimizer_params = training_cfg.get("optimizer_params")
    if optimizer_params is not None:
        fit_kwargs["optimizer_params"] = optimizer_params

    model.fit(**fit_kwargs)

    if config.get("evaluate_on_test", True):
        eval_model = SentenceTransformer(output_path) if training_cfg.get("save_best_model", True) else model
        eval_model.evaluate(test_evaluator)


if __name__ == "__main__":
    main()
