import argparse
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.config import get_config
from src.utils import seed_everything, str2bool, load_model_tokenizer, PromptDataset

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
    parser.add_argument('--for_utilities', type=str2bool, default=False, help='Whether to perform inference to compute utilities downstream')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    configs = get_config(args.dataset)
    prompt_text = configs[3]
    model, tokenizer = load_model_tokenizer(args.model, '/scratch/manish/hf_cache/', 8)
    dataset = PromptDataset(f'processed/retrieved_{args.dataset}_{args.prefix_name}_{args.N}_{args.split}.tsv', tokenizer, args.k, prompt_text, args.for_utilities)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True)
   
    all_queries = list()
    all_answers = list()
    all_generated_answers = list()
    for idx, prompt_batch in enumerate(tqdm(dataloader)):
        prompts = prompt_batch['prompt']
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        generated_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_answers = []
        for output in generated_output:
            start = output.find(prompt_text['suffix']) + len(prompt_text['suffix'])
            response = output[start:].strip()
            generated_answers.append(response)
        all_queries.extend(prompt_batch['query'])
        all_answers.extend(prompt_batch['answers'])
        all_generated_answers.extend(generated_answers)

    if not args.for_utilities:        
        pd.DataFrame({"query": all_queries, "answers": all_answers, "generated_answers": all_generated_answers}).to_csv(f'processed/inferred_{args.model.split("/")[1]}_{args.dataset}_{args.prefix_name}_{args.N}_{args.k}_{args.split}.tsv', sep='\t', index=False)
    else:
        pd.DataFrame({"query": all_queries, "answers": all_answers, "generated_answers": all_generated_answers}).to_csv(f'processed/inferred_forutilities_{args.model.split("/")[1]}_{args.dataset}_{args.prefix_name}_{args.N}_{args.k}_{args.split}.tsv', sep='\t', index=False)

if __name__ == '__main__':
    seed_everything(10)
    main()
