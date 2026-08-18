import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

train_corpus = pd.read_csv("./data/top-10-dataset/train.tsv", sep="\t", names=["candidate", "candidate_label"])

train_df_full = pd.read_csv("./data/utilities/train_dataset_lambdamart_N10.csv")
val_df_full = pd.read_csv("./data/utilities/validation_dataset_lambdamart_N10.csv")

def get_top10_by_score(df):
    return df.sort_values(by=["query", "score"], ascending=[True, False]) \
             .groupby("query").head(10).reset_index(drop=True)

train_top10 = get_top10_by_score(train_df_full)
val_top10 = get_top10_by_score(val_df_full)

corpus_texts = train_corpus["candidate"].tolist()
idf_dict = get_idf_dict(corpus_texts)

print("Training LDA model on full train corpus...")
lda_model, lda_dict = build_lda_model(corpus_texts)

handcrafted_features = [
    'num_q_terms', 'num_q_unique', 'num_d_terms', 'num_d_unique',
    'min_q_idf', 'max_q_idf', 'sum_q_idf',
    'min_d_idf', 'max_d_idf', 'sum_d_idf',
    'overlap', 'bm25_score'
]

lda_features = [
    'q_avg_phi_norm', 'q_phi_norm_min', 'q_phi_norm_max',
    'q_phi_pair_min_sim', 'q_phi_pair_max_sim', 'q_phi_pair_avg_sim',
    'd_avg_phi_norm', 'd_phi_norm_min', 'd_phi_norm_max',
    'd_phi_pair_min_sim', 'd_phi_pair_max_sim', 'd_phi_pair_avg_sim',
    'qd_phi_single_link_sim', 'qd_phi_complete_link_sim', 'qd_phi_avg_link_sim',
    'qd_theta_cos_sim'
]

feature_names = handcrafted_features + lda_features

def extract_all_features(row):
    handcrafted = extract_features(row['query'], row['candidate'], idf_dict)
    lda_feats = extract_lda_features(row['query'], row['candidate'], lda_model, lda_dict)
    return handcrafted + lda_feats

print("Extracting features for TRAIN...")
train_feats = train_top10.apply(extract_all_features, axis=1, result_type='expand')
train_feats.columns = feature_names
train_top10 = pd.concat([train_top10, train_feats], axis=1)

print("Extracting features for VALIDATION...")
val_feats = val_top10.apply(extract_all_features, axis=1, result_type='expand')
val_feats.columns = feature_names
val_top10 = pd.concat([val_top10, val_feats], axis=1)

def assign_relevance(df, ascending=True):
    df = df.sort_values(by=['query', 'score'], ascending=[True, ascending])
    df['relevance'] = df.groupby('query').cumcount(ascending=False)
    return df

train_asc = assign_relevance(train_top10.copy(), ascending=True)
val_asc = assign_relevance(val_top10.copy(), ascending=True)

train_desc = assign_relevance(train_top10.copy(), ascending=False)
val_desc = assign_relevance(val_top10.copy(), ascending=False)

def train_and_save_model(train_df, val_df, model_name):
    X_train = train_df[feature_names]
    y_train = train_df['relevance']
    group_train = train_df.groupby('query').size().to_list()

    X_val = val_df[feature_names]
    y_val = val_df['relevance']
    group_val = val_df.groupby('query').size().to_list()

    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.01,
        'num_leaves': 31,
        'verbosity': -1,
        'label_gain': list(range(max(y_train.max(), y_val.max()) + 1))
    }

    evals_result = {}

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            early_stopping(stopping_rounds=20),
            log_evaluation(period=10),
            lgb.record_evaluation(evals_result)
        ],
    )

    model.save_model(model_name)
    print(f"Saved model: {model_name}")

train_and_save_model(train_asc, val_asc, "lambdamart_with_LDA_features_asc.txt")