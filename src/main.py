import numpy as np
from sklearn.metrics import classification_report
from src.config import get_config
from src.utils import arg_parser, set_seed_device, load_model_tokenizer, load_dataset, get_kdtree, get_predictions

def main():
    args = arg_parser()
    print(args)

    device = set_seed_device(42, args['gpu_id'])
    print(device)

    chat_format = 'instruct' in args['model_name'].lower()

    datapath, classes, classes_in_data, prompt_text, batch_size = get_config(args['dataset'], args['K_max'], args['fairness_instruction'])
    if args['approach'] == 'subset':
        if args['supervised']:
            f1s, fneds, fpeds, scores = subset_selection_supervised(args['gpu_id'], args['dataset'], datapath,
                                                                    args['subset_exists'],
                                                                    args['cache_dir'], classes, batch_size=64, epochs=15,
                                                                    learning_rate=1e-5, num_batch=args['num_batch'],
                                                                    lambd=args['lambd'],
                                                                    corruption_prct=args['corruption_prct'])
        else:
            f1s, fneds, fpeds, scores = subset_selection(chat_format, load_dataset, load_model_tokenizer, get_kdtree,
                                                         get_predictions, datapath, classes_in_data, args, classes,
                                                         prompt_text, args['gpu_id'], args['dataset'], datapath,
                                                         args['subset_exists'],
                                                         args['cache_dir'], batch_size=batch_size, epochs=15,
                                                         learning_rate=1e-5, num_batch=args['num_batch'],
                                                         lambd=args['lambd'], corruption_prct=args['corruption_prct'], update_freq=args['update_freq'])
    dataset_dict, distinct_attributes = load_dataset(datapath, classes_in_data, args['approach'], args['lambd'],
                                                     args['model_name'], args['corruption_prct'], args['supervised'], args['label_flip'])

    distinct_attribute_values = []
    for v in distinct_attributes.values():
        distinct_attribute_values.extend(list(v))

    model, tokenizer = load_model_tokenizer(args['model_name'], args['single_precision'], args['cache_dir'])
    model.to(device)
    model.eval()

    print(f'Class tokens: {tuple([tokenizer.encode(clas, add_special_tokens=False) for clas in classes])}')
    class_idx = tuple([tokenizer.encode(clas, add_special_tokens=False)[0] for clas in classes])

    sbert_model, tree = get_kdtree(dataset_dict)

    seq_lens, all_preds, all_labels, all_attributes, _ = get_predictions(dataset_dict, args['K_max'],
                                                                         args['balance_demonstrations'],
                                                                         classes_in_data, distinct_attribute_values,
                                                                         class_idx, classes, device, prompt_text,
                                                                         tokenizer, model, tree, sbert_model,
                                                                         batch_size, chat_format)
    seq_lens = np.array(seq_lens)
    if args['approach'] == 'subset':
        print(f'f1s: {f1s}')
        print(f'fneds: {fneds}')
        print(f'fpeds: {fpeds}')
        print(f'scores: {scores}')
    print("Mean Sequence length: ", seq_lens.mean())
    print("Min Sequence length: ", seq_lens.min())
    print("Max Sequence length: ", seq_lens.max())
    print("95th percentile: ", np.percentile(seq_lens, 95))
    print("99th percentile: ", np.percentile(seq_lens, 99))
    print("99.9th percentile: ", np.percentile(seq_lens, 99.9))
    all_preds = [pred.item() for pred in all_preds]
    report = classification_report(all_labels, all_preds, digits=4)
    print('Classification Report:')
    print(report)
    fpr_fnr(all_labels, all_preds, all_attributes, distinct_attributes, len(classes))


if __name__ == "__main__":
    main()


