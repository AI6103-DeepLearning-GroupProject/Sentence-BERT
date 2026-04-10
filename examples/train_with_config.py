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
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import yaml
from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, SentenceTransformer, SentencesDataset, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, TripletEvaluator
from sentence_transformers.readers import InputExample, NLIDataReader, STSDataReader, TripletReader
from sentence_transformers.util import set_seed


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Sentence-Transformer experiment from config.yaml")
    parser.add_argument("--config", type=str, default=os.path.join("examples", "config.yaml"), help="Path to YAML config file")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name inside config.experiments")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    parser.add_argument("--model-variant", type=str, default=None, help="Runtime model variant key from config.runtime.model_variants")
    parser.add_argument("--data-mode", type=str, default=None, help="Runtime data mode key from config.runtime.data_modes. Examples: sts, nli, nli+mnrl, sts+nli")
    parser.add_argument("--epochs", type=int, default=None, help="Runtime epoch override for stages that do not set num_epochs explicitly")
    parser.add_argument("--output-path", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume-from", type=str, default=None, help="Load model from an existing path before training")
    parser.add_argument("--start-stage", type=int, default=1, help="1-based runtime stage index to start from")
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


def _build_output_path(exp_name: str, config: dict, suffix: Optional[str] = None):
    output_cfg = config.get("output", {})
    if "path" in output_cfg:
        return output_cfg["path"]

    root = output_cfg.get("root", "output")
    prefix = output_cfg.get("name_prefix", exp_name)
    if suffix:
        prefix = "{}_{}".format(prefix, suffix)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(root, "{}-{}".format(prefix, timestamp))


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace("+", "_plus_").replace("-", "_").lower()


def _normalize_data_mode(value: str) -> str:
    return value.strip().lower().replace("+", "_").replace("-", "_")


def _resolve_runtime_plan(config: dict, args) -> Optional[dict]:
    runtime_cfg = config.get("runtime")
    if not isinstance(runtime_cfg, dict):
        return None

    model_variants = runtime_cfg.get("model_variants", {})
    data_modes = runtime_cfg.get("data_modes", {})
    if not model_variants or not data_modes:
        return None

    model_variant = args.model_variant or runtime_cfg.get("model_variant", "sbert_base")
    if model_variant not in model_variants:
        raise ValueError("Unknown runtime model_variant '{}'. Available: {}".format(model_variant, sorted(model_variants.keys())))

    requested_data_mode = args.data_mode or runtime_cfg.get("data_mode", "sts")
    normalized_mode = _normalize_data_mode(requested_data_mode)
    normalized_to_key = {_normalize_data_mode(key): key for key in data_modes.keys()}
    if normalized_mode not in normalized_to_key:
        raise ValueError("Unknown runtime data_mode '{}'. Available: {}".format(requested_data_mode, sorted(data_modes.keys())))
    data_mode_key = normalized_to_key[normalized_mode]

    default_epochs = runtime_cfg.get("epochs", 4)
    epochs = args.epochs if args.epochs is not None else default_epochs

    mode_cfg = data_modes[data_mode_key]
    stages_cfg = mode_cfg.get("stages", [])
    if not stages_cfg:
        raise ValueError("Runtime data_mode '{}' has no stages configured".format(data_mode_key))

    base_training_cfg = dict(config.get("training", {}))
    stages = []
    for stage_idx, stage_cfg in enumerate(stages_cfg):
        if "task" not in stage_cfg or "data" not in stage_cfg:
            raise ValueError("runtime.data_modes.{}.stages[{}] must define task and data".format(data_mode_key, stage_idx))

        stage_training_cfg = dict(base_training_cfg)
        stage_training_cfg.update(stage_cfg.get("training", {}))
        if "num_epochs" not in stage_training_cfg:
            stage_training_cfg["num_epochs"] = epochs

        stage_name = stage_cfg.get("name", "stage{}".format(stage_idx + 1))
        stages.append({
            "name": stage_name,
            "task": stage_cfg["task"],
            "data": stage_cfg["data"],
            "training": stage_training_cfg,
        })

    return {
        "model_variant": model_variant,
        "data_mode": data_mode_key,
        "model": model_variants[model_variant],
        "stages": stages,
    }


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


def _load_pair_examples(data_cfg: dict) -> List[InputExample]:
    pairs_path = data_cfg["pairs_path"]
    delimiter = data_cfg.get("delimiter", "\t")
    quoting = _parse_csv_quoting(data_cfg.get("quoting", "QUOTE_MINIMAL"))
    has_header = data_cfg.get("has_header", False)
    text1_col_idx = data_cfg.get("text1_col_idx", 0)
    text2_col_idx = data_cfg.get("text2_col_idx", 1)
    max_examples = data_cfg.get("max_examples", 0)
    drop_mirror_pairs = data_cfg.get("drop_mirror_pairs", True)

    examples = []
    seen_pair_keys = set()
    skipped_duplicate_pairs = 0
    with open(pairs_path, encoding="utf-8", newline="") as f_in:
        reader = csv.reader(f_in, delimiter=delimiter, quoting=quoting)
        if has_header:
            next(reader, None)

        for row_idx, row in enumerate(reader):
            max_col_idx = max(text1_col_idx, text2_col_idx)
            if len(row) <= max_col_idx:
                continue

            text1 = row[text1_col_idx].strip()
            text2 = row[text2_col_idx].strip()
            if not text1 or not text2:
                continue

            if drop_mirror_pairs:
                pair_key = (text1, text2) if text1 <= text2 else (text2, text1)
            else:
                pair_key = (text1, text2)

            if pair_key in seen_pair_keys:
                skipped_duplicate_pairs += 1
                continue
            seen_pair_keys.add(pair_key)

            guid = "{}-{}".format(os.path.basename(pairs_path), row_idx)
            examples.append(InputExample(guid=guid, texts=[text1, text2], label=1.0))

            if 0 < max_examples <= len(examples):
                break

    if not examples:
        raise ValueError("No valid sentence pairs found in {}".format(pairs_path))

    if skipped_duplicate_pairs > 0:
        logging.info(
            "Filtered %d duplicate/mirrored pair rows while loading %s",
            skipped_duplicate_pairs,
            pairs_path,
        )

    return examples


def _build_task_components(config: dict, model: SentenceTransformer):
    task_cfg = config["task"]
    training_cfg = config["training"]
    data_cfg = config["data"]

    batch_size = training_cfg.get("batch_size", 16)
    task_type = task_cfg["type"]

    if task_type == "nli_softmax":
        nli_reader = NLIDataReader(data_cfg["nli_path"])

        train_data = SentencesDataset(nli_reader.get_examples(data_cfg.get("nli_train_file", "train.gz")), model=model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        train_loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=nli_reader.get_num_labels(),
        )

        evaluator = None
        test_evaluator = None
        sts_path = data_cfg.get("sts_path")
        if sts_path:
            sts_reader = STSDataReader(
                sts_path,
                normalize_scores=data_cfg.get("normalize_scores", True),
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

    if task_type == "mnrl_contrastive":
        pair_examples = _load_pair_examples(data_cfg)
        train_data = SentencesDataset(pair_examples, model=model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        evaluator = None
        test_evaluator = None
        sts_path = data_cfg.get("sts_path")
        if sts_path:
            sts_reader = STSDataReader(
                sts_path,
                normalize_scores=data_cfg.get("normalize_scores", True),
            )

            dev_data = SentencesDataset(sts_reader.get_examples(data_cfg.get("sts_dev_file", "sts-dev.csv")), model=model)
            dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
            evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

            test_data = SentencesDataset(sts_reader.get_examples(data_cfg.get("sts_test_file", "sts-test.csv")), model=model)
            test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
            test_evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

        return train_dataloader, train_loss, evaluator, test_evaluator

    raise ValueError("Unknown task.type: {}".format(task_type))


def _build_fit_kwargs(train_dataloader, train_loss, evaluator, training_cfg: dict, output_path: Optional[str], seed: int, deterministic: bool):
    num_epochs = training_cfg.get("num_epochs", 1)
    warmup_steps = training_cfg.get("warmup_steps")
    if warmup_steps is None:
        warmup_ratio = training_cfg.get("warmup_ratio", 0.1)
        warmup_steps = int(math.ceil(len(train_dataloader) * num_epochs * warmup_ratio))

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

    return fit_kwargs


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

    runtime_plan = _resolve_runtime_plan(config, args)
    if runtime_plan is not None:
        runtime_suffix = "{}_{}".format(_safe_name(runtime_plan["model_variant"]), _safe_name(runtime_plan["data_mode"]))
        output_path = args.output_path or _build_output_path(args.experiment, config, suffix=runtime_suffix)
        logging.info("Runtime mode enabled: model_variant=%s, data_mode=%s", runtime_plan["model_variant"], runtime_plan["data_mode"])
        logging.info("Output path: %s", output_path)

        if args.resume_from:
            model = SentenceTransformer(args.resume_from)
            logging.info("Resuming model from: %s", args.resume_from)
        else:
            model = _build_model(runtime_plan["model"])

        final_test_evaluator = None
        final_save_best_model = True

        total_stages = len(runtime_plan["stages"])
        if args.start_stage < 1 or args.start_stage > total_stages:
            raise ValueError("--start-stage must be within [1, {}], got {}".format(total_stages, args.start_stage))
        if args.start_stage > 1:
            logging.info("Skip first %d stage(s), start from stage %d/%d", args.start_stage - 1, args.start_stage, total_stages)

        for stage_idx in range(args.start_stage - 1, total_stages):
            stage_cfg = runtime_plan["stages"][stage_idx]
            stage_name = stage_cfg["name"]
            is_last_stage = (stage_idx == total_stages - 1)

            logging.info("Start stage %d/%d: %s", stage_idx + 1, total_stages, stage_name)
            train_dataloader, train_loss, evaluator, test_evaluator = _build_task_components(stage_cfg, model)

            stage_training_cfg = dict(stage_cfg["training"])
            pending_swap = None
            if not is_last_stage:
                stage_training_cfg["save_best_model"] = False
                stage_output_path = None
            else:
                stage_output_path = output_path
                # fit() refuses non-empty output dirs. For resume-in-place, save to tmp then swap.
                if (
                    args.resume_from
                    and os.path.abspath(args.resume_from) == os.path.abspath(output_path)
                    and os.path.isdir(output_path)
                    and os.listdir(output_path)
                ):
                    tmp_output_path = output_path + ".tmp_save"
                    if os.path.isdir(tmp_output_path):
                        shutil.rmtree(tmp_output_path)
                    stage_output_path = tmp_output_path
                    pending_swap = (tmp_output_path, output_path)

            fit_kwargs = _build_fit_kwargs(
                train_dataloader=train_dataloader,
                train_loss=train_loss,
                evaluator=evaluator,
                training_cfg=stage_training_cfg,
                output_path=stage_output_path,
                seed=seed,
                deterministic=deterministic,
            )
            logging.info("Stage %s warmup steps: %s", stage_name, fit_kwargs["warmup_steps"])
            model.fit(**fit_kwargs)

            if pending_swap is not None:
                tmp_output_path, final_output_path = pending_swap
                if os.path.isdir(final_output_path):
                    shutil.rmtree(final_output_path)
                shutil.move(tmp_output_path, final_output_path)

            final_test_evaluator = test_evaluator
            final_save_best_model = stage_training_cfg.get("save_best_model", True)

        if config.get("evaluate_on_test", True) and final_test_evaluator is not None:
            eval_model = SentenceTransformer(output_path) if final_save_best_model else model
            eval_model.evaluate(final_test_evaluator)
        return

    model = _build_model(config["model"])
    train_dataloader, train_loss, evaluator, test_evaluator = _build_task_components(config, model)
    training_cfg = config["training"]
    output_path = args.output_path or _build_output_path(args.experiment, config)
    logging.info("Output path: %s", output_path)

    fit_kwargs = _build_fit_kwargs(
        train_dataloader=train_dataloader,
        train_loss=train_loss,
        evaluator=evaluator,
        training_cfg=training_cfg,
        output_path=output_path,
        seed=seed,
        deterministic=deterministic,
    )
    logging.info("Warmup steps: %s", fit_kwargs["warmup_steps"])
    model.fit(**fit_kwargs)

    if config.get("evaluate_on_test", True):
        eval_model = SentenceTransformer(output_path) if training_cfg.get("save_best_model", True) else model
        eval_model.evaluate(test_evaluator)


if __name__ == "__main__":
    main()
