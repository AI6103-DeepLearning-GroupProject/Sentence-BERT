import argparse
import csv
import glob
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize direction-2 contrastive experiments.')
    parser.add_argument('--simcse_pattern', default='output/training_simcse_unsup-*',
                        help='Glob pattern for SimCSE run folders')
    parser.add_argument('--test_summary_csv', default='output/summary/three_model_sts_test_results.csv',
                        help='Unified STS-test summary csv')
    parser.add_argument('--output_csv', default='output/summary/direction2_simcse_run_summary.csv',
                        help='Output csv path for run-level summary')
    parser.add_argument('--output_md', default='docs/direction2_analysis_2026-04-04.md',
                        help='Output markdown path for discussion')
    return parser.parse_args()


def read_last_row(csv_path):
    with open(csv_path, encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def load_test_scores(test_csv):
    scores = {}
    if not os.path.exists(test_csv):
        return scores
    with open(test_csv, encoding='utf8') as f:
        for row in csv.DictReader(f):
            scores[row['model_dir']] = float(row['test_cosine_spearman'])
    return scores


def main():
    args = parse_args()
    run_dirs = sorted(glob.glob(args.simcse_pattern))
    test_scores = load_test_scores(args.test_summary_csv)

    rows = []
    for run_dir in run_dirs:
        eval_csv = os.path.join(run_dir, 'similarity_evaluation_results.csv')
        if not os.path.exists(eval_csv):
            continue

        last = read_last_row(eval_csv)
        if last is None:
            continue

        rows.append({
            'run_dir': run_dir,
            'dev_cosine_spearman': float(last.get('cosine_spearman', 0.0)),
            'dev_cosine_pearson': float(last.get('cosine_pearson', 0.0)),
            'test_cosine_spearman': test_scores.get(run_dir, ''),
        })

    rows.sort(key=lambda x: float(x['test_cosine_spearman']) if x['test_cosine_spearman'] != '' else x['dev_cosine_spearman'],
              reverse=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['run_dir', 'dev_cosine_spearman', 'dev_cosine_pearson', 'test_cosine_spearman'])
        writer.writeheader()
        writer.writerows(rows)

    best_line = 'No valid SimCSE run found.'
    if rows:
        best = rows[0]
        best_line = 'Best run: {} (dev={:.4f}, test={})'.format(
            best['run_dir'],
            best['dev_cosine_spearman'],
            '{:.4f}'.format(best['test_cosine_spearman']) if best['test_cosine_spearman'] != '' else 'N/A'
        )

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open('w', encoding='utf8') as f:
        f.write('# Direction-2 Contrastive Learning Analysis\n\n')
        f.write('## Summary\n\n')
        f.write('- {}\n'.format(best_line))
        f.write('- SimCSE run count: {}\n'.format(len(rows)))
        f.write('- Run summary csv: `{}`\n\n'.format(args.output_csv))

        f.write('## Run Table\n\n')
        f.write('| Run | Dev Spearman | Dev Pearson | Test Spearman |\n')
        f.write('|---|---:|---:|---:|\n')
        for row in rows:
            test_val = row['test_cosine_spearman']
            test_txt = '{:.4f}'.format(test_val) if test_val != '' else 'N/A'
            f.write('| {} | {:.4f} | {:.4f} | {} |\n'.format(
                os.path.basename(row['run_dir']),
                row['dev_cosine_spearman'],
                row['dev_cosine_pearson'],
                test_txt
            ))

        f.write('\n## Recommended Next Experiments\n\n')
        f.write('1. Increase unsupervised epochs to 3-5 with large batch size.\n')
        f.write('2. Compare pair_mode=`dropout` vs `aug_only` vs `dropout_and_aug`.\n')
        f.write('3. For augmentation, tune `random_delete_swap` with delete_ratio/swap_ratio in [0.05, 0.15].\n')
        f.write('4. Keep same seeds and evaluate by mean±std on STS-test.\n')

    print('Saved:', args.output_csv)
    print('Saved:', args.output_md)
    print(best_line)


if __name__ == '__main__':
    main()
