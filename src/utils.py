import argparse
import json
import random
import numpy as np
import pandas as pd
import os
import re
import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria


# import pandas as pd
# from sklearn.metrics import recall_score, confusion_matrix
# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, set_seed
# from tqdm import tqdm
# from torch.nn import functional as F
# from src.dataset_processors import DataProcessor

def arg_parser():
    parser = argparse.ArgumentParser(description='Light-weight Reranker in RAG')
    parser.add_argument('-d', '--dataset', type=str,  help='Dataset', required=True)
    parser.add_argument('-m', '--model_name', type=str, help='Model name', required=True)
    parser.add_argument('--single_precision', action='store_true')
    parser.add_argument('--no_single_precision', dest='single_precision', action='store_false')
    parser.set_defaults(single_precision=True)
    parser.add_argument('-k', '--K_max', type=int, help='Max value of k', required=True)
    parser.add_argument('-g', '--gpu_id', type=int, help='GPU Id', default=0, required=False)
    parser.add_argument('-c', '--cache_dir', type=str, help='Cache dir for storing transformers models',
                        default="/scratch/manish/hf_cache/", required=False)
    parser.add_argument('-a', '--approach', type=str, help='kshot', default='kshot',
                        required=False)
    args = vars(parser.parse_args())
    return args

# def set_seed_device(seed_value, gpu_id):
#     random.seed(seed_value)
#     torch.cuda.manual_seed(seed_value)
#     torch.cuda.manual_seed_all(seed_value)
#     set_seed(seed_value)
#     device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
#     if device != torch.device('cpu'):
#         torch.cuda.set_device(gpu_id)
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
#     return device

def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_json(file_path: str):
    with open(file_path, "rb") as reader:
        data = json.load(reader)
    return data

def sanitize_query(query):
    query = query.lower()
    query = re.sub(r'["\'(){}<>\\|/:*?&#=-]', ' ', query)
    query = re.sub(r"\s+", " ", query).strip()
    return query

def set_quantization(quantization_bits):
    if quantization_bits in [4, 8]:
        bnb_config = BitsAndBytesConfig()
        if quantization_bits == 4:
            bnb_config.load_in_4bit = True
            bnb_config.bnb_4bit_quant_type = 'nf4'
            bnb_config.bnb_4bit_use_double_quant = True
            bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
        elif quantization_bits == 8:
            bnb_config.load_in_8bit = True
        return bnb_config
    return None

def load_model_tokenizer(model_name, cache_dir, quantization_bits):
    hf_access_token = '<Paste_your_hf_access_token_here>'
    os.environ['HF_ACCESS_TOKEN'] = hf_access_token
    os.environ['HF_TOKEN'] = hf_access_token
    bnb_config = set_quantization(quantization_bits)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config,
                                                 torch_dtype=torch.bfloat16,
                                                 cache_dir=cache_dir, token=hf_access_token, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir=cache_dir)
    #if tokenizer.pad_token_id is None:
    #    tokenizer.pad_token_id = tokenizer.bos_token_id
    #if model.config.pad_token_id is None:
    #    model.config.pad_token_id = model.config.bos_token_id
    # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    #tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    #model.resize_token_embeddings(len(tokenizer))
    #model.config.pad_token_id = tokenizer.pad_token_id
    # Most LLMs don't have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token  
    #assert model.config.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"
    return model, tokenizer

class PromptDataset(Dataset):
    def __init__(self, data_path, tokenizer, k, prompt_text, for_utilities=False):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.k = k
        self.prompt_text = prompt_text
        self.for_utilities = for_utilities
        self._load_data()

    def _load_data(self):
        data = pd.read_csv(self.data_path, sep='\t')
        # data_dict = data.to_dict(orient='records')
        self.queries = list()
        self.prompts = list()
        self.prompt_tokens_lengths = list()
        self.answers = list()
        if not self.for_utilities:
            for qid, group in data.groupby("qid"):
                answers = group.iloc[0]["answers"]
                #try:
                #    answers = eval(gt) if isinstance(gt, str) else gt
                #except Exception:
                #    answers = [str(gt)]
                sorted_g = group.sort_values("score", ascending=False).head(self.k)
                docs = '\n'.join(sorted_g['text'].astype(str).tolist()[::-1])
                query = group.iloc[0]['query']
                query = sanitize_query(query)

                prompt = f"{self.prompt_text['instruction']}\nDocuments:\n{docs}\n{self.prompt_text['prefix']} {query}\n{self.prompt_text['suffix']}"
                # Check if the prompt exceeds 'max_tokenized_length'
                tokens = self.tokenizer.tokenize(prompt)
                tokens_len = len(tokens)
                if tokens_len >= self.tokenizer.model_max_length:
                    print("Skipping example with qid {} due to prompt length.".format(qid))
                    continue

                # If prompt is within limit, add to preprocessed data
                self.queries.append(query)
                self.prompts.append(prompt)
                self.prompt_tokens_lengths.append(tokens_len)
                self.answers.append(answers)
        else:
            self.k = 1 #1-shot utilities
            for index, row in data.iterrows():
                answers = row["answers"]
                docs = row['text']
                query = row['query']
                query = sanitize_query(query)

                prompt = f"{self.prompt_text['instruction']}\nDocuments:\n{docs}\n{self.prompt_text['prefix']} {query}\n{self.prompt_text['suffix']}"
                # Check if the prompt exceeds 'max_tokenized_length'
                tokens = self.tokenizer.tokenize(prompt)
                tokens_len = len(tokens)
                if tokens_len >= self.tokenizer.model_max_length:
                    print("Skipping example with qid {} due to prompt length.".format(row['qid']))
                    continue

                # If prompt is within limit, add to preprocessed data
                self.queries.append(query)
                self.prompts.append(prompt)
                self.prompt_tokens_lengths.append(tokens_len)
                self.answers.append(answers)


    def __getitem__(self, idx: int):
        return {
            "query": self.queries[idx],
            "prompt": self.prompts[idx],
            "prompt_tokens_len": self.prompt_tokens_lengths[idx],
            "answers": self.answers[idx]
        }

    def __len__(self):
        return len(self.prompts)

# def load_dataset(datapath):
#     dataset_dict = dict()
#     data_processor = DataProcessor(classes_in_data)
#     dataset_dict['train'], distinct_attributes = data_processor.get_examples(datapath,
#                                                                          ('train_subset_supervised_'  + str(lambd)) if corruption_prct == 0 else (f'train_{corruption_prct * 100}_subset_supervised_' + str(lambd)))
#     dataset_dict['test'], _ = data_processor.get_examples(datapath, 'test')
#     dataset_dict['dev'], _ = data_processor.get_examples(datapath, 'dev')
#     print(type(dataset_dict['train']))
#     print("Length of train set: ", len(dataset_dict['train']))
#     print("Length of test set", len(dataset_dict['test']))
#     print("Length of dev set", len(dataset_dict['dev']))
#     print("Train example at 0th  index: ", dataset_dict['train'][0])
#     return dataset_dict

# def get_kdtree(dataset):
#     sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Load SentenceTransformer model
#     train_sentences = [ex.text_a if ex.text_b == "" else ex.text_a + " " + ex.text_b for ex in dataset['train']]  # Get sentence embeddings
#     train_embeddings = sbert_model.encode(train_sentences)
#     tree = KDTree(train_embeddings)
#     return sbert_model, tree
#
# def get_predictions(dataset_dict, k, balance_demonstrations, classes_in_data, distinct_attribute_values, class_idx,
#                     classes, device, prompt_text, tokenizer, model, tree, sbert_model, batch_size, chat_format, splt='test'):
#     prompts_end = tokenizer.batch_encode_plus([prompt_text['suffix'] for _ in range(batch_size)], return_tensors='pt',
#                                               padding='longest', add_special_tokens=False)
#     max_rem_len = model.config.max_position_embeddings - prompts_end['input_ids'].shape[1]
#     all_preds = list()
#     all_probs = list()
#     all_labels = list()
#     all_attributes = list()
#     seq_lens = list()
#
#     all_trees = dict()
#     all_dsets = dict()
#     if balance_demonstrations:
#         for clas in range(len(classes_in_data)):
#             for atr in distinct_attribute_values:
#                 temp_dataset = dict()
#                 temp_dataset['train'] = [ex for ex in dataset_dict['train'] if ex.label == clas and atr in ex.attribute.values()]
#                 _, tree_subgroup = get_kdtree(temp_dataset)
#                 all_trees[atr + str(clas)] = tree_subgroup
#                 all_dsets[atr + str(clas)] = temp_dataset
#
#     for start_idx in tqdm(range(0, len(dataset_dict[splt]), batch_size)):
#         indexes = slice(start_idx, start_idx + batch_size)
#         seq_len, labels, preds, confidence, attributes = pred_batch(dataset_dict, indexes, k, balance_demonstrations,
#                                                                     classes_in_data, distinct_attribute_values, all_trees,
#                                                                     all_dsets, max_rem_len, class_idx, classes, device,
#                                                                     prompt_text, prompts_end, tokenizer, model, tree,
#                                                                     sbert_model, splt, chat_format)
#         seq_lens.append(seq_len)
#         all_preds.extend(preds)
#         all_probs.append(confidence)
#         all_labels.extend(labels)
#         all_attributes.extend(attributes)
#     return seq_lens, all_preds, all_labels, all_attributes, all_probs
#
# def pred_batch(dataset_dict, indexes, k, balance_demonstrations, classes_in_data, distinct_attribute_values, all_trees,
#                all_dsets, max_rem_len, class_idx, classes, device, prompt_text, prompts_end, tokenizer, model, tree,
#                sbert_model, splt='test', chat_format=True):
#     prompts = list()
#     if k == 0:
#         if not chat_format:
#             prompts = [
#                 f'{prompt_text["instruction"]}{prompt_text["prefix"]}{test_example.text_a}\n{prompt_text["suffix"]}' for
#                 test_example in dataset_dict[splt][indexes]]
#         else:
#             prompts = [[{'role': 'system', 'content': f'{prompt_text["instruction"]}'},
#                         {'role': 'user',
#                          'content': f'{prompt_text["prefix"]}{test_example.text_a}\n{prompt_text["suffix"]}'}] for
#                        test_example in dataset_dict[splt][indexes]]
#     else:
#         test_sentences = [
#             f'{test_example.text_a}' if test_example.text_b == "" else f'{test_example.text_a + " " + test_example.text_b}'
#             for test_example in dataset_dict[splt][indexes]]
#         test_embeddings = sbert_model.encode(test_sentences)  # Get the embedding of the test_sentence
#         K_examples_all = list()
#         if not balance_demonstrations:
#             _, top_k_indices = tree.query(test_embeddings, k=k)  # find the top k most similar train sentences
#             for i in top_k_indices:
#                 K_examples_all.append([dataset_dict['train'][j] for j in i])
#         else:
#             examples_subgroups = dict()
#             subgroup_len = len(classes_in_data) * len(distinct_attribute_values)
#             for clas in range(len(classes_in_data)):
#                 for atr in distinct_attribute_values:
#                     _, top_k_indices = all_trees[atr+str(clas)].query(test_embeddings, k=max(1, int(k/subgroup_len)))
#                     examples_pergroup = list()
#                     for i in top_k_indices:
#                         examples_pergroup.append([all_dsets[atr + str(clas)]['train'][j] for j in i])
#                     examples_subgroups[atr + str(clas)] = examples_pergroup
#             fnal = None
#             for key, val in examples_subgroups.items():
#                 if fnal == None:
#                     fnal = val
#                 else:
#                     fnal = np.column_stack((fnal, val)).tolist()
#             fnal = [random.sample(f, k) for f in fnal]
#             K_examples_all = fnal
#         demonstrations = list()
#         for K_examples in K_examples_all:
#             demonstrations.append(
#                 ''.join(
#                     [f'{prompt_text["prefix"]}{example.text_a}\n{prompt_text["suffix"]} {classes[example.label]}\n' for
#                      example in K_examples]))
#         #print([f'{prompt_text["instruction"]}{demonstrations[indx]}{prompt_text["prefix"]}{test_example.text_a}\n' for indx, test_example in
#         #     enumerate(dataset_dict['test'][indexes])])
#         #exit()
#         if not chat_format:
#             prompts = [f'{prompt_text["instruction"]}{demonstrations[indx]}{prompt_text["prefix"]}{test_example.text_a}\n' for
#              indx, test_example in
#              enumerate(dataset_dict[splt][indexes])]
#         else:
#             prompts = [[{'role': 'system', 'content': f'{prompt_text["instruction"]}{demonstrations[indx]}'},
#                         {'role': 'user',
#                          'content': f'{prompt_text["prefix"]}{test_example.text_a}\n{prompt_text["suffix"]}'}] for
#                        indx, test_example in
#                        enumerate(dataset_dict[splt][indexes])]
#     if chat_format:
#         prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True, padding='longest')
#     #print(prompts)
#     #exit()
#     enc = tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding='longest')
#     # for key, enc_value in list(enc.items()):
#     #     # prompts[key] = prompts[key][:, 1:]
#     #     enc_value = enc_value[:, :max_rem_len]
#     #     enc[key] = torch.cat([enc_value, prompts_end[key][:enc_value.shape[0]]], dim=1)
#     seq_len = enc['input_ids'].shape[1]
#
#     enc = {ky: v.to(device) for ky, v in enc.items()}
#     with torch.no_grad():
#         #result = model(**enc)
#         #print(result.keys())
#         #tokenizer.batch_decode(result)
#         #exit()
#         result = model(**enc).logits
#     result = result[:, -1, class_idx]
#     result = F.softmax(result, dim=1)
#     preds = torch.argmax(result, dim=-1)
#     labels = [test_example.label for test_example in dataset_dict[splt][indexes]]
#     attributes = [test_example.attribute for test_example in dataset_dict[splt][indexes]]
#     confidence = result[0][labels[0]].item()
#     return seq_len, labels, preds, confidence, attributes
#
# #def macro_averaged_tnr(y_true, y_pred):
# #    cm = confusion_matrix(y_true, y_pred)
# #    num_classes = cm.shape[0]
# #    tnr_per_class = []
# #    for i in range(num_classes):
# #        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
# #        fp = np.sum(cm[:, i]) - cm[i, i]
# #        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
# #        tnr_per_class.append(tnr)
# #    return np.mean(tnr_per_class)
#
# def macro_averaged_tnr(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     fp_class = np.sum(cm, axis=0) - np.diag(cm)
#     fn_class = np.sum(cm, axis=1) - np.diag(cm)
#     tp_class = np.diag(cm)
#     tn_class = np.sum(cm) - (fp_class + fn_class + tp_class)
#     tnr_class = tn_class / (tn_class + fp_class)
#     tpr_class = tp_class / (tp_class + fn_class)
#     return tnr_class, tpr_class
#
# def micro_averaged_tnr(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     fp_total = np.sum(np.sum(cm, axis=0) - np.diag(cm))
#     fn_total = np.sum(np.sum(cm, axis=1) - np.diag(cm))
#     tp_total = np.sum(np.diag(cm))
#     tn_total = np.sum(cm) - (fp_total + fn_total + tp_total)
#     print(cm)
#     print(fp_total)
#     print(fn_total)
#     print(tp_total)
#     print(tn_total)
#     return tn_total / (tn_total + fp_total)
#
# def fpr_fnr(all_labels, all_preds, all_attributes, distinct_attributes, num_classes):
#     fpr, fnr, num_instances = dict(), dict(), dict()
#     if num_classes == 2:
#         tpr = recall_score(all_labels, all_preds)
#         tnr = recall_score(all_labels, all_preds, pos_label=0)
#     else:
#         tnr, tpr = macro_averaged_tnr(all_labels, all_preds)
#     fpr['overall'] = 1 - tnr
#     fnr['overall'] = 1 - tpr
#     num_instances['overall'] = len(all_labels)
#     print(f'Overall fpr: {fpr["overall"]}, fnr: {fnr["overall"]}, num_instances: {num_instances["overall"]}')
#
#     for attribute, attribute_values in distinct_attributes.items():
#         all_labels_attribute = [l for (l, a) in zip(all_labels, all_attributes) if a[attribute] in attribute_values]
#         all_preds_attribute = [p for (p, a) in zip(all_preds, all_attributes) if a[attribute] in attribute_values]
#         if num_classes == 2:
#             tpr = recall_score(all_labels_attribute, all_preds_attribute)
#             tnr = recall_score(all_labels_attribute, all_preds_attribute, pos_label=0)
#         else:
#             tnr, tpr = macro_averaged_tnr(all_labels_attribute, all_preds_attribute)
#         fpr[attribute] = 1 - tnr
#         fnr[attribute] = 1 - tpr
#         num_instances[attribute] = len(all_labels_attribute)
#         print(f'{attribute} fpr: {fpr[attribute]}, fnr: {fnr[attribute]}, num_instances: {num_instances[attribute]}')
#         for attribute_val in attribute_values:
#             all_labels_attribute_val = [l for (l, a) in zip(all_labels, all_attributes) if a[attribute] == attribute_val]
#             all_preds_attribute_val = [p for (p, a) in zip(all_preds, all_attributes) if a[attribute] == attribute_val]
#             if num_classes == 2:
#                 tpr = recall_score(all_labels_attribute_val, all_preds_attribute_val)
#                 tnr = recall_score(all_labels_attribute_val, all_preds_attribute_val, pos_label=0)
#             else:
#                 tnr, tpr = macro_averaged_tnr(all_labels_attribute_val, all_preds_attribute_val)
#             fpr[attribute_val] = 1 - tnr
#             fnr[attribute_val] = 1 - tpr
#             num_instances[attribute_val] = len(all_labels_attribute_val)
#             print(f'{attribute_val} fpr: {fpr[attribute_val]}, fnr: {fnr[attribute_val]}, num_instances: {num_instances[attribute_val]}')
#     fneds, fpeds = dict(), dict()
#     for attribute, attribute_values in distinct_attributes.items():
#         fned, fped = 0, 0
#         #for attribute_val in attribute_values:
#         #    fned += abs(fnr[attribute] - fnr[attribute_val])
#         #    fped += abs(fpr[attribute] - fpr[attribute_val])
#         for comb in combinations(attribute_values, 2):
#             if num_classes == 2:
#                 fned += abs(fnr[comb[0]] - fnr[comb[1]])
#                 fped += abs(fpr[comb[0]] - fpr[comb[1]])
#             else:
#                 fned += np.mean(np.abs(fnr[comb[0]] - fnr[comb[1]]))
#                 fped += np.mean(np.abs(fpr[comb[0]] - fpr[comb[1]]))
#         fneds[attribute] = fned / len(list(combinations(attribute_values, 2)))
#         fpeds[attribute] = fped / len(list(combinations(attribute_values, 2)))
#     print(f'fneds: {fneds}, fpeds: {fpeds}')
#     return fned, fped

