import os
import argparse
import ast
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from src.config import get_config
from src.utils import seed_everything, str2bool, load_model_tokenizer, PromptDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def compute_utilities(model, tokenizer, prompts, answers_batch):
    """
    Computes the log-likelihood utility for a batch of prompts and their multiple ground-truth answers.
    Utility = max_{y in answers} log P(y | prompt)
    """
    batch_utilities = []
    
    for prompt, answers in zip(prompts, answers_batch):
        if isinstance(answers, str):
            try:
                answers = ast.literal_eval(answers)
            except:
                answers = [answers]
       
        # 1. Format the input context as a User message in Chat Format
        messages = [
            {"role": "user", "content": prompt}
        ]
        # apply_chat_template with add_generation_prompt=True formats the user input
        # and appends the assistant header (e.g., "<|start_header_id|>assistant<|end_header_id|>\n\n")
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
 
        # Encode prompt strictly WITH special tokens (e.g., BOS)
        prompt_enc = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True).to(model.device)
        prompt_ids = prompt_enc.input_ids
        prompt_mask = prompt_enc.attention_mask
        prompt_len = prompt_ids.size(1)
 
        max_log_prob = float('-inf')
        
        # Compute log-prob for each ground-truth answer
        for answer in answers:
            # 2. Encode answer WITHOUT special tokens (no extra BOS in the middle)
            #formatted_answer = answer.strip() #if answer.startswith(" ") else " " + answer
            clean_answer = answer.strip()
            answer_enc = tokenizer(clean_answer, return_tensors="pt", add_special_tokens=False).to(model.device)
            answer_ids = answer_enc.input_ids
            answer_mask = answer_enc.attention_mask

            # 3. Concatenate IDs and Attention Masks
            input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, answer_mask], dim=1)
            input_ids = input_ids.cpu()

            # 4. Forward Pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
   
            # 5. Shift logits and labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Compute token-level log probabilities
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            # Gather the log prob of the actual tokens that were present
            target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Slice out only the tokens corresponding to the ground-truth answer
            # prompt_len - 1 due to the left-shift
            answer_log_probs = target_log_probs[0, (prompt_len - 1):]
            
            # Utility: Sum of log-probabilities of target tokens
            sequence_log_prob = answer_log_probs.sum().item()
            
            if sequence_log_prob > max_log_prob:
                max_log_prob = sequence_log_prob
        batch_utilities.append(max_log_prob)
        
    return batch_utilities

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
    all_docs = list()
    all_qids = list()
    all_docids = list()
    all_utilities = list()
    for idx, prompt_batch in enumerate(tqdm(dataloader)):
        prompts = prompt_batch['prompt']
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        generated_ids = generated_ids.cpu()
        generated_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_answers = []
        for output in generated_output:
            start = output.find(prompt_text['suffix']) + len(prompt_text['suffix'])
            response = output[start:].strip()
            generated_answers.append(response)

        all_queries.extend(prompt_batch['query'])
        all_answers.extend(prompt_batch['answers'])
        all_generated_answers.extend(generated_answers)
        all_docs.extend(prompt_batch['docs'])
        if args.for_utilities:
            batch_utilities = compute_utilities(model, tokenizer, prompts, prompt_batch['answers'])
            all_qids.extend(prompt_batch['qid'].tolist())
            all_docids.extend(prompt_batch['docid'].tolist())
            all_utilities.extend(batch_utilities)

        del inputs, generated_ids
        if idx % 10 == 0: # Optional: Clear cache every 10 batches to prevent fragmentation
            torch.cuda.empty_cache()

    if not args.for_utilities:        
        pd.DataFrame({"query": all_queries, "answers": all_answers, "generated_answers": all_generated_answers}).to_csv(f'processed/inferred_{args.model.split("/")[1]}_{args.dataset}_{args.prefix_name}_{args.N}_{args.k}_{args.split}.tsv', sep='\t', index=False)
    else:
        pd.DataFrame({"qid": all_qids, "query": all_queries, "docid": all_docids, "doc": all_docs, "answers": all_answers, "generated_answers": all_generated_answers, "utility": all_utilities}).to_csv(f'processed/inferred_forutilities_{args.model.split("/")[1]}_{args.dataset}_{args.prefix_name}_{args.N}_{args.k}_{args.split}.tsv', sep='\t', index=False)

if __name__ == '__main__':
    seed_everything(10)
    main()
