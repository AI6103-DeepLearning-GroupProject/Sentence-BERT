"""
This example trains BERT for STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure similarity.

New option:
- --pooling mean|attention
"""
from torch.utils.data import DataLoader
import argparse
import math
import random
import torch
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
import logging
from datetime import datetime


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


def parse_args():
    parser = argparse.ArgumentParser(description='Train SBERT on STSBenchmark with configurable pooling.')
    parser.add_argument('--model_name', default='bert-base-uncased', help='Backbone model name')
    parser.add_argument('--pooling', default='mean', choices=['mean', 'attention'],
                        help='Sentence pooling strategy')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--dataset_path', default='datasets/stsbenchmark', help='Path to STSbenchmark data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_pooling_model(word_embedding_dimension, pooling):
    if pooling == 'mean':
        return models.Pooling(
            word_embedding_dimension,
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
            pooling_mode_attention_tokens=False
        )

    return models.Pooling(
        word_embedding_dimension,
        pooling_mode_mean_tokens=False,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
        pooling_mode_attention_tokens=True
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    model_save_path = 'output/training_stsbenchmark_bert-{}-seed{}-{}'.format(
        args.pooling,
        args.seed,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    sts_reader = STSDataReader(args.dataset_path, normalize_scores=True)

    # Use BERT for mapping tokens to embeddings
    word_embedding_model = models.BERT(args.model_name)

    # Apply selected pooling to get one fixed sized sentence vector
    pooling_model = build_pooling_model(
        word_embedding_model.get_word_embedding_dimension(),
        args.pooling
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read STSbenchmark train dataset")
    train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    logging.info("Read STSbenchmark dev dataset")
    dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    warmup_steps = math.ceil(len(train_data) * args.epochs / args.batch_size * 0.1)  # 10% warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=args.epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)

    # Evaluate on test split
    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
    model.evaluate(evaluator)


if __name__ == '__main__':
    main()
