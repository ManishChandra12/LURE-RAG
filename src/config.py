def get_config(dataset):
    corpus_path = 'data/corpus.json'
    if dataset == 'nq_open':
        datapath = '/scratch/manish/hf_cache/dataset/nq_open_gold/train_dataset.json'
        prompt_instruction = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."
        prompt_prefix = 'Question: '
        prompt_suffix = 'Answer:'
        batch_size = 1
        prompt_text = {'instruction': prompt_instruction, 'prefix': prompt_prefix, 'suffix': prompt_suffix}
    if dataset == 'triviaqa':
        datapath = '/scratch/manish/hf_cache/dataset/nq_open_gold/train_dataset.json'
        prompt_instruction = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."
        prompt_prefix = 'Question: '
        prompt_suffix = 'Answer:'
        batch_size = 1
        prompt_text = {'instruction': prompt_instruction, 'prefix': prompt_prefix, 'suffix': prompt_suffix}
    return corpus_path, datapath, batch_size, prompt_text
