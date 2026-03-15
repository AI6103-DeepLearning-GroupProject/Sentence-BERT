"""
Evaluate a pretrained sentence-transformer model on STS benchmark and write outputs to a folder.
"""
import argparse
import json
import logging
import os
from datetime import datetime

from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler, SentenceTransformer, SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
from sentence_transformers.util import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained SentenceTransformer model on STS benchmark")
    parser.add_argument("--model-name", type=str, default="bert-base-nli-mean-tokens")
    parser.add_argument("--sts-path", type=str, default="datasets/stsbenchmark")
    parser.add_argument("--sts-test-file", type=str, default="sts-test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
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

    set_seed(args.seed, deterministic=True)

    if args.output_path:
        output_path = args.output_path
    else:
        safe_model_name = args.model_name.replace("/", "_")
        output_path = os.path.join("output", "eval_pretrained_{}_{}".format(safe_model_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    os.makedirs(output_path, exist_ok=True)

    model = SentenceTransformer(args.model_name)
    sts_reader = STSDataReader(args.sts_path)
    test_data = SentencesDataset(examples=sts_reader.get_examples(args.sts_test_file), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader, name="sts_test")

    score = model.evaluate(evaluator, output_path=output_path)

    summary = {
        "model_name": args.model_name,
        "sts_path": args.sts_path,
        "sts_test_file": args.sts_test_file,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "score": float(score),
        "output_path": output_path,
    }

    with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, indent=2)

    logging.info("Evaluation finished. Main score: %.6f", score)
    logging.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()
