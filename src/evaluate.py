import argparse
import regex
import string
import pandas as pd
from collections import Counter
from src.config import get_config
import src.normalize_text

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for generating LLM predictions.")
    parser.add_argument('--model', type=str, help='Name of the model', default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument('--dataset', type=str, help='Name of the dataset: nq_open', required=True)
    parser.add_argument('--prefix_name', type=str, default='bm25',
                        help='Initial part of the name of the saved index')
    parser.add_argument('--k', type=int, default=5,  help='Number of documents in the prompt')
    parser.add_argument('--N', type=int, default=10, help='Number of documents retrieved')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--split', type=str, help='Split to run LLM inference on', default="test")
    parser.add_argument('--max_new-tokens', type=int, default=15, help='Max new tokens')
    args = parser.parse_args()
    return args

# Normalization adapted from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def remove_articles(text: str) -> str:
    """
    Removes articles ('a', 'an', 'the') from the text.
    """
    return regex.sub(r'\b(a|an|the)\b', ' ', text)

def remove_punc(text: str) -> str:
    """
    Removes punctuation from the text and replaces it with a space.
    """
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    return text

def white_space_fix(text: str) -> str:
    """
    Fixes extra whitespace in the text by collapsing multiple spaces into one.
    """
    return ' '.join(text.split())

def normalize_answer(s, lowercase=True):
    if lowercase:
        s = str(s).lower()
    s = src.normalize_text.normalize(s)
    return white_space_fix(remove_articles(remove_punc(s)))

def are_answers_matching(prediction, ground_truths):
    normalized_prediction = normalize_answer(prediction)
    ground_truths = eval(ground_truths)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth in normalized_prediction:
            return True
    return False

def compute_f1(prediction, ground_truth):
    """Computes the token-level F1 score between a single prediction and a single ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    # If either list is empty, F1 is 1.0 only if both are empty.
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    # Precision: fraction of predicted tokens that are correct
    precision = 1.0 * num_same / len(pred_tokens)

    # Recall: fraction of ground truth tokens that are retrieved
    recall = 1.0 * num_same / len(gold_tokens)

    # F1 score is the harmonic mean
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_metrics(prediction, ground_truths, which):
    """
    Computes Exact Match (EM) and F1 score for a prediction against multiple ground truths.
    The final score is the maximum achieved against any ground truth answer.
    """

    max_f1 = 0.0
    max_acc = 0.0
    max_em = 0.0

    # Iterate over all possible ground truth answers
    for ground_truth in eval(ground_truths):
        em = int(normalize_answer(prediction) == normalize_answer(ground_truth))
        max_em = max(max_em, em)

        # 1. Exact Match (EM)
        acc = int(normalize_answer(ground_truth) in normalize_answer(prediction))
        max_acc = max(max_acc, acc)

        # 2. F1 Score
        f1 = compute_f1(prediction, ground_truth)
        max_f1 = max(max_f1, f1)

    if which == 'em':
        return max_em
    elif which == 'accuracy':
        return max_acc
    elif which == 'f1':
        return max_f1

def main():
    args = parse_arguments()
    configs = get_config(args.dataset)
    results_df = pd.read_csv(f'processed/inferred_{args.model.split("/")[1]}_{args.dataset}_{args.prefix_name}_{args.N}_{args.k}_{args.split}.tsv', sep='\t')
    results_df['ans_match_after_norm'] = results_df.apply(lambda x: are_answers_matching(x['generated_answers'], x['answers']), axis=1)
    print(results_df.head())
    accuracy = round(results_df['ans_match_after_norm'].sum() / len(results_df), 4)
    print("ACCURACY: ", accuracy)
    results_df['em'] = results_df.apply(
        lambda x: compute_metrics(x['generated_answers'], x['answers'], 'em'), axis=1)
    results_df['accuracy'] = results_df.apply(
        lambda x: compute_metrics(x['generated_answers'], x['answers'], 'accuracy'), axis=1)
    results_df['f1'] = results_df.apply(
        lambda x: compute_metrics(x['generated_answers'], x['answers'], 'f1'), axis=1)
    em = round(results_df['em'].sum() / len(results_df), 4)
    print("EM: ", em)
    accuracy = round(results_df['accuracy'].sum() / len(results_df), 4)
    print("ACCURACY: ", accuracy)
    f1 = round(results_df['f1'].sum() / len(results_df), 4)
    print("F1: ", f1)

if __name__ == '__main__':
    main()
