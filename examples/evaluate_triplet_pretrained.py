"""
Evaluate a sentence-transformer model on a triplet dataset and aggregate results over seeds.
"""
import argparse
import csv
import json
import logging
import os
import statistics
from datetime import datetime

from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, SentenceTransformer, SentencesDataset
from sentence_transformers.evaluation import (
    EmbeddingDiagnosticsEvaluator,
    SequentialEvaluator,
    SimilarityFunction,
    TripletEvaluator,
)
from sentence_transformers.readers import TripletReader
from sentence_transformers.util import set_seed


def _parse_csv_quoting(quoting_value):
    if isinstance(quoting_value, int):
        return quoting_value
    if isinstance(quoting_value, str) and hasattr(csv, quoting_value):
        return getattr(csv, quoting_value)
    return csv.QUOTE_MINIMAL


def _resolve_main_distance(name: str):
    if name == "cosine":
        return SimilarityFunction.COSINE
    if name == "euclidean":
        return SimilarityFunction.EUCLIDEAN
    if name == "manhattan":
        return SimilarityFunction.MANHATTAN
    return None


def _read_eval_metrics(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))
    if not rows:
        raise ValueError("No rows found in {}".format(csv_path))

    last_row = rows[-1]
    metrics = {}
    for key, value in last_row.items():
        if key in {"epoch", "steps"}:
            continue
        metrics[key] = float(value)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SentenceTransformer model on triplet dataset")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--triplet-path", type=str, default="datasets/wikipedia-sections-triplets")
    parser.add_argument("--triplet-test-file", type=str, default="test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument(
        "--main-distance",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean", "manhattan", "max"],
        help="Distance metric used as run score",
    )
    parser.add_argument("--s1-col-idx", type=int, default=1)
    parser.add_argument("--s2-col-idx", type=int, default=2)
    parser.add_argument("--s3-col-idx", type=int, default=3)
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--quoting", type=str, default="QUOTE_MINIMAL")
    parser.add_argument("--has-header", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="")
    return parser.parse_args()


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
        output_path = os.path.join("output", "eval_triplet_{}_{}".format(safe_model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(output_path, exist_ok=True)

    quoting = _parse_csv_quoting(args.quoting)
    has_header = bool(args.has_header)
    main_distance = _resolve_main_distance(args.main_distance)

    seeds = [args.seed + idx * args.seed_step for idx in range(args.num_runs)]
    run_results = []

    for run_idx, run_seed in enumerate(seeds, start=1):
        set_seed(run_seed, deterministic=True)

        run_output_path = output_path if args.num_runs == 1 else os.path.join(output_path, "run_{:02d}_seed_{}".format(run_idx, run_seed))
        os.makedirs(run_output_path, exist_ok=True)

        model = SentenceTransformer(args.model_name)
        reader = TripletReader(
            args.triplet_path,
            s1_col_idx=args.s1_col_idx,
            s2_col_idx=args.s2_col_idx,
            s3_col_idx=args.s3_col_idx,
            has_header=has_header,
            delimiter=args.delimiter,
            quoting=quoting,
        )

        test_data = SentencesDataset(examples=reader.get_examples(args.triplet_test_file), model=model)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
        evaluator = SequentialEvaluator(
            [
                EmbeddingDiagnosticsEvaluator(test_dataloader, name="triplet_test"),
                TripletEvaluator(test_dataloader, main_distance_function=main_distance, name="triplet_test"),
            ]
        )

        score = model.evaluate(evaluator, output_path=run_output_path)
        metrics = _read_eval_metrics(os.path.join(run_output_path, "triplet_evaluation_triplet_test_results.csv"))
        run_results.append(
            {
                "run_index": run_idx,
                "seed": run_seed,
                "score": float(score),
                "metrics": metrics,
                "output_path": run_output_path,
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
        "triplet_path": args.triplet_path,
        "triplet_test_file": args.triplet_test_file,
        "batch_size": args.batch_size,
        "main_distance": args.main_distance,
        "num_runs": args.num_runs,
        "seed_start": args.seed,
        "seed_step": args.seed_step,
        "seeds": seeds,
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
        "Evaluation finished. %d runs, score mean=%.6f, score std=%.6f",
        args.num_runs,
        aggregate["score_mean"],
        aggregate["score_std"],
    )
    logging.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()
