import argparse
import csv
import os
from pathlib import Path

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.readers import STSDataReader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate three trained models on STS test split.')
    parser.add_argument('--dataset_path', default='datasets/stsbenchmark', help='Path to STS benchmark folder')
    parser.add_argument('--batch_size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--output_csv', default='output/summary/three_model_sts_test_results.csv',
                        help='Path to save test metrics csv')
    parser.add_argument('--model_dirs', nargs='+', default=[
        'output/training_stsbenchmark_bert-mean-seed42-2026-04-04_06-47-47',
        'output/training_stsbenchmark_bert-attention-seed42-2026-04-04_06-50-15',
        'output/training_simcse_unsup-seed42-2026-04-04_06-53-48',
    ], help='Model directories to evaluate')
    return parser.parse_args()


def infer_method_name(model_dir: str) -> str:
    name = os.path.basename(model_dir)
    if 'attention' in name:
        return 'Attention Pooling'
    if 'simcse' in name or 'unsup' in name:
        return 'Unsupervised (SimCSE-style)'
    if 'mean' in name:
        return 'Mean Pooling'
    return name


def main():
    args = parse_args()
    reader = STSDataReader(args.dataset_path, normalize_scores=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_dir in args.model_dirs:
        model = SentenceTransformer(model_dir)
        test_data = SentencesDataset(examples=reader.get_examples('sts-test.csv'), model=model)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
        evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

        # model.evaluate returns main score (cosine spearman)
        main_score = float(model.evaluate(evaluator))

        tmp_csv = Path(model_dir) / 'similarity_evaluation_results.csv'
        cosine_pearson = ''
        cosine_spearman = main_score
        if tmp_csv.exists():
            with tmp_csv.open(encoding='utf8') as f:
                entries = list(csv.DictReader(f))
            if entries:
                cosine_pearson = entries[-1].get('cosine_pearson', '')

        rows.append({
            'method': infer_method_name(model_dir),
            'model_dir': model_dir,
            'test_cosine_spearman': f'{cosine_spearman:.10f}',
            'reference_cosine_pearson': cosine_pearson,
        })

    with out_path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'model_dir', 'test_cosine_spearman', 'reference_cosine_pearson'])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['method']}\t{row['test_cosine_spearman']}\t{row['model_dir']}")
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
