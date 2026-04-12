import argparse
import csv
import os
from pathlib import Path

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.readers import STSDataReader, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate baseline vs prompt SBERT on STS test.')
    parser.add_argument('--baseline_model', required=True, help='Path to baseline model folder')
    parser.add_argument('--prompt_model', required=True, help='Path to prompt model folder')
    parser.add_argument('--sts_path', default='datasets/stsbenchmark', help='Path to STS folder')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--prompt_sts',
                        default='[Task: Semantic Similarity] Determine how similar this sentence is: ',
                        help='Prompt prefix used for prompt model input')
    parser.add_argument('--output_csv', default='output/summary/prompt_vs_baseline_sts_test.csv',
                        help='Output csv file')
    return parser.parse_args()


def apply_prompt(examples, prompt_prefix, use_prompt):
    if not use_prompt:
        return examples
    updated = []
    for ex in examples:
        updated.append(InputExample(guid=ex.guid,
                                    texts=[prompt_prefix + ex.texts[0], prompt_prefix + ex.texts[1]],
                                    label=ex.label))
    return updated


def eval_model(model_path, examples, batch_size):
    model = SentenceTransformer(model_path)
    ds = SentencesDataset(examples=examples, model=model)
    dl = DataLoader(ds, shuffle=False, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dl, name='sts_test')
    return float(model.evaluate(evaluator))


def main():
    args = parse_args()

    reader = STSDataReader(args.sts_path, normalize_scores=True)
    base_examples = reader.get_examples('sts-test.csv')
    prompt_examples = apply_prompt(base_examples, args.prompt_sts, use_prompt=True)

    baseline_score = eval_model(args.baseline_model, base_examples, args.batch_size)
    prompt_score = eval_model(args.prompt_model, prompt_examples, args.batch_size)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['model_type', 'model_path', 'sts_test_cosine_spearman'])
        writer.writeheader()
        writer.writerow({'model_type': 'baseline', 'model_path': args.baseline_model,
                         'sts_test_cosine_spearman': '{:.10f}'.format(baseline_score)})
        writer.writerow({'model_type': 'prompt', 'model_path': args.prompt_model,
                         'sts_test_cosine_spearman': '{:.10f}'.format(prompt_score)})

    print('baseline\t{:.10f}\t{}'.format(baseline_score, args.baseline_model))
    print('prompt\t{:.10f}\t{}'.format(prompt_score, args.prompt_model))
    print('Saved:', out_path)


if __name__ == '__main__':
    main()
