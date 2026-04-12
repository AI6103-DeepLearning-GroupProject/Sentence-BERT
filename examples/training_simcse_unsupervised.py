"""
Unsupervised SimCSE-style training with Sentence-BERT.

Baseline mode:
- Positive pair: (text, text), relying on dropout noise for two stochastic views.

Advanced mode (augmentation):
- Positive pair: (text, augmented_text), where augmented_text is created with
  random deletion / random swap.
- Batch-internal samples are treated as negatives via MultipleNegativesRankingLoss.
"""
from torch.utils.data import DataLoader
import argparse
import math
import os
import random
import torch
import logging
from datetime import datetime

from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader, InputExample


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def parse_args():
    parser = argparse.ArgumentParser(description='Train SBERT with unsupervised SimCSE-style objective.')
    parser.add_argument('--model_name', default='bert-base-uncased', help='Backbone model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--unsup_text_path', required=True,
                        help='Path to plain text corpus (one sentence per line)')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Limit training samples (0 means use all)')
    parser.add_argument('--sts_path', default='datasets/stsbenchmark', help='Path to STSbenchmark data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Direction-2 related options
    parser.add_argument('--pair_mode', default='dropout',
                        choices=['dropout', 'aug_only', 'dropout_and_aug'],
                        help='How to build positive pairs for contrastive learning')
    parser.add_argument('--augmentation', default='none',
                        choices=['none', 'random_delete', 'random_swap', 'random_delete_swap'],
                        help='Augmentation strategy for positive pair second view')
    parser.add_argument('--delete_ratio', type=float, default=0.1,
                        help='Token deletion ratio for random_delete augmentation')
    parser.add_argument('--swap_ratio', type=float, default=0.1,
                        help='Token swap ratio for random_swap augmentation')
    parser.add_argument('--min_tokens_for_aug', type=int, default=4,
                        help='Minimum tokens required before applying augmentation')

    # Training controls
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio over total train steps')
    parser.add_argument('--evaluation_steps', type=int, default=1000, help='Evaluate every N train steps')
    parser.add_argument('--scheduler', default='warmuplinear',
                        choices=['constantlr', 'warmupconstant', 'warmuplinear', 'warmupcosine',
                                 'warmupcosinewithhardrestarts'],
                        help='Learning rate scheduler')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training if supported')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _random_delete(words, delete_ratio):
    if len(words) <= 2:
        return words
    kept = [w for w in words if random.random() > delete_ratio]
    if len(kept) < 2:
        return words
    return kept


def _random_swap(words, swap_ratio):
    if len(words) <= 2:
        return words
    swapped = list(words)
    swap_count = max(1, int(len(swapped) * swap_ratio))
    for _ in range(swap_count):
        i = random.randrange(len(swapped))
        j = random.randrange(len(swapped))
        swapped[i], swapped[j] = swapped[j], swapped[i]
    return swapped


def augment_sentence(sentence, method, delete_ratio, swap_ratio, min_tokens_for_aug):
    words = sentence.split()
    if method == 'none' or len(words) < min_tokens_for_aug:
        return sentence

    if method == 'random_delete':
        out_words = _random_delete(words, delete_ratio)
    elif method == 'random_swap':
        out_words = _random_swap(words, swap_ratio)
    elif method == 'random_delete_swap':
        out_words = _random_delete(words, delete_ratio)
        out_words = _random_swap(out_words, swap_ratio)
    else:
        out_words = words

    augmented = ' '.join(out_words).strip()
    return augmented if augmented else sentence


def build_positive_pair(sentence, args):
    if args.pair_mode == 'dropout':
        return sentence, sentence

    augmented = augment_sentence(
        sentence,
        method=args.augmentation,
        delete_ratio=args.delete_ratio,
        swap_ratio=args.swap_ratio,
        min_tokens_for_aug=args.min_tokens_for_aug
    )

    if args.pair_mode == 'aug_only':
        return sentence, augmented

    # dropout_and_aug: keep both dropout noise and text augmentation effects
    # If augmentation is none, this naturally falls back to dropout-only pair.
    return sentence, augmented


def load_unsupervised_examples(path, args):
    if not os.path.exists(path):
        raise ValueError('File not found: {}'.format(path))

    examples = []
    changed_pairs = 0

    with open(path, 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            sentence = line.strip()
            if not sentence:
                continue

            s1, s2 = build_positive_pair(sentence, args)
            if s1 != s2:
                changed_pairs += 1

            examples.append(InputExample(guid='unsup-{}'.format(idx), texts=[s1, s2], label=1.0))

            if args.max_samples > 0 and len(examples) >= args.max_samples:
                break

    if not examples:
        raise ValueError('No valid sentences found in {}'.format(path))

    return examples, changed_pairs


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = 'training_simcse_unsup-{}-{}-seed{}-{}'.format(
        args.pair_mode,
        args.augmentation,
        args.seed,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    model_save_path = os.path.join('output', run_name)

    # Use BERT + mean pooling as SimCSE baseline encoder
    word_embedding_model = models.BERT(args.model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False,
                                   pooling_mode_attention_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: {}'.format(device_name))
    logging.info('Read unsupervised corpus: {}'.format(args.unsup_text_path))

    train_examples, changed_pairs = load_unsupervised_examples(args.unsup_text_path, args)
    logging.info('Loaded {} unsupervised examples'.format(len(train_examples)))
    logging.info('Positive pairs with text-level change: {} ({:.2f}%)'.format(
        changed_pairs,
        100.0 * changed_pairs / max(1, len(train_examples))
    ))

    train_data = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    sts_reader = STSDataReader(args.sts_path, normalize_scores=True)
    logging.info('Read STSbenchmark dev dataset')
    dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    warmup_steps = math.ceil(len(train_data) * args.epochs / args.batch_size * args.warmup_ratio)
    logging.info('Warmup-steps: {}'.format(warmup_steps))

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=args.epochs,
              evaluation_steps=args.evaluation_steps,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              scheduler=args.scheduler,
              weight_decay=args.weight_decay,
              optimizer_params={'lr': args.learning_rate, 'eps': 1e-6, 'correct_bias': False},
              fp16=args.fp16,
              seed=args.seed)

    # Final STS test evaluation (for quick visibility)
    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=sts_reader.get_examples('sts-test.csv'), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
    final_score = model.evaluate(evaluator)
    logging.info('Final STS test cosine Spearman: {:.6f}'.format(final_score))


if __name__ == '__main__':
    main()
