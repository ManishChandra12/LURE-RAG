import argparse
import time
import pyterrier as pt
from src.utils import seed_everything, str2bool, read_json
from src.config import get_config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for indexing a corpus.")
    parser.add_argument('--dataset', type=str, help='Name of the dataset: nq_open', required=True)
    parser.add_argument('--lower_case', type=str2bool, default=False, help='Whether to lower case the corpus text')
    parser.add_argument('--do_normalize_text', type=str2bool, default=True, help='Whether to normalize the corpus text')
    parser.add_argument('--output_dir', type=str, default='index/',
                        help='Output directory for saving index')
    parser.add_argument('--prefix_name', type=str, default='bm25',
                        help='Initial part of the name of the saved index')
    parser.add_argument('--threads', type=int, help='Number of threads to use for indexing', default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    configs = get_config(args.dataset)
    corpus_path = configs[0]

    print("Loading corpus...")
    start_time = time.perf_counter()
    corpus = read_json(corpus_path)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Corpus loaded in {elapsed_time:.6f} seconds")

    # Convert to format PyTerrier expects: list of dicts with docno + text
    docs = [{"docno": str(i), "text": t["text"]} for i, t in enumerate(corpus)]

    print("Building index...")
    start_time = time.perf_counter()
    # using default porter stemmer, English stopwords and English tokenizer
    indexer = pt.IterDictIndexer(f"./{args.output_dir}{args.dataset}_{args.prefix_name}_wiki_index", threads=args.threads, meta={'docno': 20, 'text': 4096}, overwrite=True)
    indexref = indexer.index(docs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Index created in {elapsed_time:.6f} seconds")


if __name__ == '__main__':
    seed_everything(10)
    main()
