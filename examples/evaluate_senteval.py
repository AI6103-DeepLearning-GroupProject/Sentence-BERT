"""
Evaluate a sentence-transformer model with SentEval and write outputs to a folder.

This script expects a local SentEval toolkit checkout, e.g.:
  --senteval-toolkit-path datasets/SentEval
  --senteval-data-path datasets/SentEval/data
"""
import argparse
import csv
import json
import logging
import os
import statistics
import sys
import inspect
import collections
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.util import set_seed


def _ensure_inspect_getargspec_compat() -> None:
    """
    SentEval calls inspect.getargspec(), removed in Python 3.11+.
    Provide a backward-compatible shim.
    """
    if hasattr(inspect, "getargspec"):
        return

    ArgSpec = collections.namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _compat_getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)

    inspect.getargspec = _compat_getargspec  # type: ignore[attr-defined]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SentenceTransformer model with SentEval")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--senteval-toolkit-path", type=str, default=os.path.join("datasets", "SentEval"))
    parser.add_argument("--senteval-data-path", type=str, default=os.path.join("datasets", "SentEval", "data"))
    parser.add_argument(
        "--tasks",
        type=str,
        default="MR,CR,SUBJ,MPQA,SST2,TREC,MRPC",
        help="Comma separated SentEval tasks",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="SentenceTransformer encode batch size")
    parser.add_argument("--kfold", type=int, default=2, help="SentEval kfold setting (must be >=2 for CV tasks)")
    parser.add_argument("--main-metric", type=str, default="auto", help="Metric key used as score, or auto")
    parser.add_argument("--classifier-nhid", type=int, default=0)
    parser.add_argument("--classifier-optim", type=str, default="rmsprop")
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--classifier-tenacity", type=int, default=3)
    parser.add_argument("--classifier-epoch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="")
    return parser.parse_args()


def _flatten_numeric(prefix: str, value, out: Dict[str, float]):
    if isinstance(value, (int, float, np.floating, np.integer)):
        out[prefix] = float(value)
        return

    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _flatten_numeric("{}_{}".format(prefix, idx), item, out)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            safe_key = str(key).replace(" ", "_").replace("/", "_")
            _flatten_numeric("{}_{}".format(prefix, safe_key), item, out)


def _task_primary_metric(task: str, metrics: Dict[str, float]) -> Tuple[str, float]:
    preferred_suffixes = [
        "_acc",
        "_test_acc",
        "_devacc",
        "_all_spearman_all",
        "_spearman",
    ]

    for suffix in preferred_suffixes:
        for key in sorted(metrics.keys()):
            if key.startswith(task + "_") and key.endswith(suffix):
                return key, metrics[key]

    task_keys = [k for k in sorted(metrics.keys()) if k.startswith(task + "_")]
    if not task_keys:
        return "", float("nan")

    first_key = task_keys[0]
    return first_key, metrics[first_key]


def _load_senteval_engine(toolkit_path: str):
    if not os.path.isdir(toolkit_path):
        raise ValueError("SentEval toolkit path not found: {}".format(toolkit_path))

    # Patch before import.
    _ensure_inspect_getargspec_compat()

    sys.path.insert(0, toolkit_path)
    try:
        import senteval  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Failed to import senteval from {}. Ensure toolkit is available.".format(toolkit_path)
        ) from exc

    # Patch again after import in case SentEval internal modules cached inspect early.
    _ensure_inspect_getargspec_compat()
    try:
        import senteval.utils as senteval_utils  # type: ignore
        if hasattr(senteval_utils, "inspect") and not hasattr(senteval_utils.inspect, "getargspec"):
            senteval_utils.inspect.getargspec = inspect.getargspec  # type: ignore[attr-defined]
    except Exception:
        pass

    logging.info("SentEval inspect compatibility patch active: getargspec=%s", hasattr(inspect, "getargspec"))
    return senteval


def _evaluate_once(args, run_seed: int):
    set_seed(run_seed, deterministic=True)
    senteval = _load_senteval_engine(args.senteval_toolkit_path)
    model = SentenceTransformer(args.model_name)

    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = []
        for tokens in batch:
            cur_tokens = []
            for token in tokens:
                if isinstance(token, bytes):
                    cur_tokens.append(token.decode("utf-8", errors="ignore"))
                else:
                    cur_tokens.append(str(token))
            sentences.append(" ".join(cur_tokens))

        embeddings = model.encode(
            sentences,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)

    effective_kfold = max(2, int(args.kfold))
    if effective_kfold != args.kfold:
        logging.warning("SentEval kfold=%d is invalid for CV tasks; using kfold=%d", args.kfold, effective_kfold)

    params_senteval = {
        "task_path": args.senteval_data_path,
        "usepytorch": True,
        "kfold": effective_kfold,
        "classifier": {
            "nhid": args.classifier_nhid,
            "optim": args.classifier_optim,
            "batch_size": args.classifier_batch_size,
            "tenacity": args.classifier_tenacity,
            "epoch_size": args.classifier_epoch_size,
        },
    }

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("No SentEval tasks specified")

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results = se.eval(tasks)

    metrics: Dict[str, float] = {}
    primary_values: List[float] = []
    primary_keys: List[str] = []

    for task in tasks:
        task_result = results.get(task, {})
        local_metrics: Dict[str, float] = {}
        _flatten_numeric(task, task_result, local_metrics)
        metrics.update(local_metrics)

        primary_key, primary_value = _task_primary_metric(task, local_metrics)
        if primary_key and np.isfinite(primary_value):
            primary_keys.append(primary_key)
            primary_values.append(primary_value)

    if not metrics:
        raise ValueError("SentEval returned no numeric metrics for tasks: {}".format(tasks))

    if args.main_metric != "auto":
        if args.main_metric not in metrics:
            raise ValueError(
                "--main-metric '{}' not found. Available metrics include: {}".format(
                    args.main_metric, sorted(metrics.keys())[:20]
                )
            )
        score = float(metrics[args.main_metric])
        main_metric_used = args.main_metric
    elif primary_values:
        score = float(statistics.mean(primary_values))
        main_metric_used = "mean({})".format(",".join(primary_keys))
    else:
        score = float(statistics.mean(metrics.values()))
        main_metric_used = "mean(all_metrics)"

    return score, metrics, main_metric_used, tasks, effective_kfold


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")

    if args.output_path:
        output_path = args.output_path
    else:
        safe_model_name = args.model_name.replace("/", "_")
        output_path = os.path.join(
            "output",
            "eval_senteval_{}_{}".format(safe_model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
        )

    os.makedirs(output_path, exist_ok=True)

    seeds = [args.seed + idx * args.seed_step for idx in range(args.num_runs)]
    run_results = []
    main_metric_used = "auto"
    tasks = []
    effective_kfold = args.kfold

    for run_idx, run_seed in enumerate(seeds, start=1):
        score, metrics, main_metric_used, tasks, effective_kfold = _evaluate_once(args, run_seed=run_seed)
        run_results.append(
            {
                "run_index": run_idx,
                "seed": run_seed,
                "score": float(score),
                "metrics": metrics,
            }
        )
        logging.info("Run %d/%d finished. seed=%d score=%.6f", run_idx, args.num_runs, run_seed, score)

    score_values = [row["score"] for row in run_results]
    aggregate = {
        "score_mean": float(statistics.mean(score_values)),
        "score_std": float(statistics.stdev(score_values)) if len(score_values) > 1 else 0.0,
    }

    metric_keys = sorted(run_results[0]["metrics"].keys())
    for key in metric_keys:
        values = [row["metrics"][key] for row in run_results]
        aggregate["{}_mean".format(key)] = float(statistics.mean(values))
        aggregate["{}_std".format(key)] = float(statistics.stdev(values)) if len(values) > 1 else 0.0

    summary = {
        "model_name": args.model_name,
        "evaluation_task": "senteval",
        "num_runs": args.num_runs,
        "seed_start": args.seed,
        "seed_step": args.seed_step,
        "seeds": seeds,
        "score_metric": main_metric_used,
        "senteval_toolkit_path": args.senteval_toolkit_path,
        "senteval_data_path": args.senteval_data_path,
        "senteval_tasks": tasks,
        "senteval_kfold": effective_kfold,
        "classifier": {
            "nhid": args.classifier_nhid,
            "optim": args.classifier_optim,
            "batch_size": args.classifier_batch_size,
            "tenacity": args.classifier_tenacity,
            "epoch_size": args.classifier_epoch_size,
        },
        "aggregate": aggregate,
        "runs": run_results,
        "output_path": output_path,
    }

    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, indent=2)

    with open(os.path.join(output_path, "summary_table.csv"), "w", newline="", encoding="utf-8") as f_out:
        fieldnames = ["run_index", "seed", "score"] + metric_keys
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_results:
            out_row = {"run_index": row["run_index"], "seed": row["seed"], "score": row["score"]}
            out_row.update(row["metrics"])
            writer.writerow(out_row)

    logging.info(
        "SentEval finished. %d runs, score mean=%.6f, score std=%.6f",
        args.num_runs,
        aggregate["score_mean"],
        aggregate["score_std"],
    )
    logging.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()
