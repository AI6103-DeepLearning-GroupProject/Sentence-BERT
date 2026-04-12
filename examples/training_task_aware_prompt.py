"""
Task-aware / prompt-based SBERT training.

This script demonstrates instruction-style prompting for different tasks:
- STS (similarity regression)
- NLI (classification)
- Unsupervised contrastive retrieval-style objective

It supports both baseline mode (no prompt) and prompt mode with task-specific prefixes.
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
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator, SequentialEvaluator
from sentence_transformers.readers import STSDataReader, NLIDataReader, InputExample


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def parse_args():
    parser = argparse.ArgumentParser(description='Task-aware prompt training for SBERT.')

    parser.add_argument('--model_name', default='bert-base-uncased', help='Backbone model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for all tasks')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--sts_path', default='datasets/stsbenchmark', help='Path to STS benchmark folder')
    parser.add_argument('--nli_path', default='datasets/AllNLI', help='Path to AllNLI folder')
    parser.add_argument('--unsup_text_path', default='datasets/unsup_sts_sentences.txt',
                        help='Plain text file (one sentence per line) for unsupervised contrastive data')

    parser.add_argument('--max_sts_samples', type=int, default=0,
                        help='Limit STS train/dev/test samples for quick experiments (0 means all)')
    parser.add_argument('--max_nli_samples', type=int, default=0,
                        help='Limit NLI train/dev samples for quick experiments (0 means all)')
    parser.add_argument('--max_unsup_samples', type=int, default=0,
                        help='Limit unsupervised samples (0 means use all)')

    parser.add_argument('--use_prompt', action='store_true',
                        help='Enable task-specific instruction prefix')
    parser.add_argument('--tasks', default='sts,nli,unsup',
                        help='Comma-separated tasks from: sts,nli,unsup')

    parser.add_argument('--prompt_sts',
                        default='[Task: Semantic Similarity] Determine how similar this sentence is: ',
                        help='Prompt prefix for STS task')
    parser.add_argument('--prompt_nli',
                        default='[Task: Natural Language Inference] Determine entailment relation for this sentence: ',
                        help='Prompt prefix for NLI task')
    parser.add_argument('--prompt_unsup',
                        default='[Task: Semantic Retrieval] Encode this sentence for retrieval: ',
                        help='Prompt prefix for unsupervised contrastive task')

    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Optimizer learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio over train steps')
    parser.add_argument('--evaluation_steps', type=int, default=1000, help='Evaluate every N train steps')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_task_set(tasks_csv):
    tasks = set([t.strip().lower() for t in tasks_csv.split(',') if t.strip()])
    supported = {'sts', 'nli', 'unsup'}
    unknown = tasks.difference(supported)
    if unknown:
        raise ValueError('Unknown tasks: {}'.format(','.join(sorted(list(unknown)))))
    if not tasks:
        raise ValueError('At least one task is required')
    return tasks


def maybe_prompt_text(text, prompt_prefix, use_prompt):
    if not use_prompt:
        return text
    return prompt_prefix + text


def apply_prompt_to_examples(examples, prompt_prefix, use_prompt):
    if not use_prompt:
        return examples

    updated = []
    for ex in examples:
        new_texts = [maybe_prompt_text(t, prompt_prefix, use_prompt=True) for t in ex.texts]
        updated.append(InputExample(guid=ex.guid, texts=new_texts, label=ex.label))
    return updated


def load_unsup_examples(path, prompt_prefix, use_prompt, max_samples):
    if not os.path.exists(path):
        raise ValueError('Unsupervised text file not found: {}'.format(path))

    examples = []
    with open(path, 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            sentence = line.strip()
            if not sentence:
                continue
            s = maybe_prompt_text(sentence, prompt_prefix, use_prompt)
            examples.append(InputExample(guid='unsup-{}'.format(idx), texts=[s, s], label=1.0))
            if max_samples > 0 and len(examples) >= max_samples:
                break

    if not examples:
        raise ValueError('No usable unsupervised samples in {}'.format(path))
    return examples


def resolve_dataset_path(path, fallback_path):
    if os.path.exists(path):
        return path
    if os.path.exists(fallback_path):
        return fallback_path
    return path


def build_model(model_name):
    word_embedding_model = models.BERT(model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False,
                                   pooling_mode_attention_tokens=False)
    return SentenceTransformer(modules=[word_embedding_model, pooling_model])


def main():
    args = parse_args()
    set_seed(args.seed)
    task_set = parse_task_set(args.tasks)

    args.nli_path = resolve_dataset_path(args.nli_path, 'examples/datasets/AllNLI')
    args.sts_path = resolve_dataset_path(args.sts_path, 'examples/datasets')

    run_name = 'training_task_aware_prompt-{}-{}-seed{}-{}'.format(
        'prompt' if args.use_prompt else 'baseline',
        '-'.join(sorted(list(task_set))),
        args.seed,
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    output_path = os.path.join('output', run_name)

    model = build_model(args.model_name)
    train_objectives = []
    eval_list = []

    # STS objective + evaluator
    if 'sts' in task_set:
        sts_reader = STSDataReader(args.sts_path, normalize_scores=True)

        sts_train = sts_reader.get_examples('sts-train.csv', max_examples=args.max_sts_samples)
        sts_train = apply_prompt_to_examples(sts_train, args.prompt_sts, args.use_prompt)
        sts_train_data = SentencesDataset(sts_train, model)
        sts_train_loader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size)
        sts_loss = losses.CosineSimilarityLoss(model=model)
        train_objectives.append((sts_train_loader, sts_loss))

        sts_dev = sts_reader.get_examples('sts-dev.csv', max_examples=args.max_sts_samples)
        sts_dev = apply_prompt_to_examples(sts_dev, args.prompt_sts, args.use_prompt)
        sts_dev_data = SentencesDataset(sts_dev, model)
        sts_dev_loader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size)
        eval_list.append(EmbeddingSimilarityEvaluator(sts_dev_loader, name='sts_dev'))

        logging.info('STS train/dev loaded with prompt={}'.format(args.use_prompt))

    # NLI objective + evaluator
    nli_softmax_loss = None
    if 'nli' in task_set:
        nli_reader = NLIDataReader(args.nli_path)
        nli_num_labels = nli_reader.get_num_labels()

        nli_train = nli_reader.get_examples('train.gz', max_examples=args.max_nli_samples)
        nli_train = apply_prompt_to_examples(nli_train, args.prompt_nli, args.use_prompt)
        nli_train_data = SentencesDataset(nli_train, model)
        nli_train_loader = DataLoader(nli_train_data, shuffle=True, batch_size=args.batch_size)

        nli_softmax_loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=nli_num_labels
        )
        train_objectives.append((nli_train_loader, nli_softmax_loss))

        nli_dev = nli_reader.get_examples('dev.gz', max_examples=args.max_nli_samples)
        nli_dev = apply_prompt_to_examples(nli_dev, args.prompt_nli, args.use_prompt)
        nli_dev_data = SentencesDataset(nli_dev, model)
        nli_dev_loader = DataLoader(nli_dev_data, shuffle=False, batch_size=args.batch_size)
        eval_list.append(LabelAccuracyEvaluator(nli_dev_loader, name='nli_dev', softmax_model=nli_softmax_loss))

        logging.info('NLI train/dev loaded with prompt={}'.format(args.use_prompt))

    # Unsupervised contrastive objective
    if 'unsup' in task_set:
        unsup_train = load_unsup_examples(
            args.unsup_text_path,
            prompt_prefix=args.prompt_unsup,
            use_prompt=args.use_prompt,
            max_samples=args.max_unsup_samples
        )
        unsup_train_data = SentencesDataset(unsup_train, model)
        unsup_train_loader = DataLoader(unsup_train_data, shuffle=True, batch_size=args.batch_size, drop_last=True)
        unsup_loss = losses.MultipleNegativesRankingLoss(model)
        train_objectives.append((unsup_train_loader, unsup_loss))

        logging.info('Unsupervised train loaded with prompt={}'.format(args.use_prompt))

    if not train_objectives:
        raise ValueError('No train objectives were built. Check --tasks and dataset paths.')

    evaluator = None
    if len(eval_list) == 1:
        evaluator = eval_list[0]
    elif len(eval_list) > 1:
        evaluator = SequentialEvaluator(eval_list)

    min_loader_len = min([len(loader) for loader, _ in train_objectives])
    warmup_steps = math.ceil(min_loader_len * args.epochs * args.warmup_ratio)
    logging.info('Warmup-steps: {}'.format(warmup_steps))

    model.fit(train_objectives=train_objectives,
              evaluator=evaluator,
              epochs=args.epochs,
              evaluation_steps=args.evaluation_steps,
              warmup_steps=warmup_steps,
              output_path=output_path,
              optimizer_params={'lr': args.learning_rate, 'eps': 1e-6, 'correct_bias': False},
              seed=args.seed)

    # Final test-time checks across available tasks
    eval_model = SentenceTransformer(output_path)

    if 'sts' in task_set:
        sts_reader = STSDataReader(args.sts_path, normalize_scores=True)
        sts_test = sts_reader.get_examples('sts-test.csv', max_examples=args.max_sts_samples)
        sts_test = apply_prompt_to_examples(sts_test, args.prompt_sts, args.use_prompt)
        sts_test_data = SentencesDataset(sts_test, eval_model)
        sts_test_loader = DataLoader(sts_test_data, shuffle=False, batch_size=args.batch_size)
        sts_eval = EmbeddingSimilarityEvaluator(sts_test_loader, name='sts_test')
        sts_score = eval_model.evaluate(sts_eval, output_path=output_path)
        logging.info('Final STS test cosine Spearman: {:.6f}'.format(sts_score))

    if 'nli' in task_set and nli_softmax_loss is not None:
        logging.info('NLI dev accuracy is logged during training via LabelAccuracyEvaluator CSV in output path.')


if __name__ == '__main__':
    main()
