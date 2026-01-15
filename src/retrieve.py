import argparse
import pyterrier as pt
import pandas as pd
from datasets import load_dataset
from src.utils import seed_everything, sanitize_query
from src.config import get_config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for retrieving from document collection.")
    parser.add_argument('--dataset', type=str, help='Name of the dataset: nq_open', required=True)
    parser.add_argument('--index_dir', type=str, default='index/',
                        help='Input directory for saved index')
    parser.add_argument('--prefix_name', type=str, default='bm25',
                        help='Initial part of the name of the saved index')
    parser.add_argument('--N', type=int, help='Number of documents to retrieve', required=True)
    args = parser.parse_args()
    return args

def preprocess_nq_open(split):
    ds = load_dataset("florin-hf/nq_open_gold", split=split, cache_dir="./data/nq_open/")
    questions = [r["question"] for r in ds]
    answers = []
    for r in ds:
        if "answers" in r:
            if isinstance(r["answers"], (list, tuple)):
                answers.append(r["answers"])
            elif isinstance(r["answers"], dict) and "text" in r["answers"]:
                answers.append(r["answers"]["text"])
            else:
                answers.append([str(r["answers"])])
        else:
            answers.append([str(r.get("answer", ""))])
    return questions, answers

def preprocess_triviaqa(split):
    ds = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split=split, cache_dir="./data/triviaqa/")
    questions = [r["question"] for r in ds]
    answers = []
    for r in ds:
        if isinstance(r["answers"], (list, tuple)):
            answers.append(r["answers"])
        elif isinstance(r["answers"], dict) and "aliases" in r["answers"]:
            answers.append(r["answers"]["aliases"])
        else:
            answers.append([str(r["answers"])])
    return questions, answers

def main():
    args = parse_arguments()
    configs = get_config(args.dataset)
    dataset_name = configs[4]
    index = pt.IndexFactory.of(f"./{args.index_dir}{args.dataset}_{args.prefix_name}_wiki_index/data.properties")
    #index.getProperties().setProperty("index.meta.index-source", "fileinmem")

    #Create a BM25 retrieval model
    bm25_topN = pt.terrier.Retriever(index, wmodel="BM25", metadata=['docno', 'text'], threads=10, num_results=args.N, verbose=True)
    #bm25_topN = (bm25 % args.N) #>> pt.text.get_text(index, metadata=['text'])

    for split in ["test", "validation", "train"]:
        print(f"Loading {split} split of the dataset")
        if args.dataset == 'nq_open':
            questions, answers = preprocess_nq_open(dataset_name, split)
        elif args.dataset == 'triviaqa':
            questions, answers = preprocess_triviaqa(dataset_name, split)

        query_df = pd.DataFrame({"qid": range(len(questions)), "query": questions, "answers": answers})
        query_df["query"] = query_df["query"].fillna("").astype(str).apply(sanitize_query)
        print(f"Retrieving top-{args.N} documents for {len(query_df)} queries")
        res = bm25_topN.transform(query_df)
        res.to_csv(f'processed/retrieved_{args.dataset}_{args.prefix_name}_{args.N}_{split}.tsv', sep='\t', index=False)



if __name__ == '__main__':
    seed_everything(10)
    main()
