import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup
)

from sensorimotor_representation.sensory_tag import sensory_tag_sentence


# Import SENSE-LM Step 1 code artifacts
from step_1.data_loader import get_data_loader       # for SENSE-LM data loading :contentReference[oaicite:0]{index=0}
from step_1.classifier import SenseLM_BinaryClassifier  # the fused BERT+sensorimotor classifier :contentReference[oaicite:1]{index=1}
from step_1.train import train as train_sense_lm     # training loop for SENSE-LM :contentReference[oaicite:2]{index=2}

# Sensorimotor norms loader
from sensorimotor_representation.load_sensorimotor_norms import load_sensorimotor_norms

def train_bert(
    train_texts, train_labels,
    val_texts,   val_labels,
    model_name='bert-base-uncased',
    epochs=3,
    batch_size=16,
    lr=2e-5,
    device='cuda'
):
    """Fine-tune a vanilla BERT on the binary task."""
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    # Tokenize
    def encode(texts):
        enc = tokenizer.batch_encode_plus(
            texts, add_special_tokens=True, padding='max_length',
            truncation=True, max_length=64, return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask']

    train_ids, train_masks = encode(train_texts)
    val_ids,   val_masks   = encode(val_texts)
    train_lbls = torch.tensor(train_labels, dtype=torch.long)
    val_lbls   = torch.tensor(val_labels, dtype=torch.long)

    train_ds = TensorDataset(train_ids, train_masks, train_lbls)
    val_ds   = TensorDataset(val_ids,   val_masks,   val_lbls)

    train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=batch_size)
    val_loader   = DataLoader(val_ds,   sampler=SequentialSampler(val_ds), batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type="single_label_classification")
    model.to(device)

    # Optimizer & scheduler
    total_steps = len(train_loader) * epochs
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            b_ids, b_mask, b_lbl = [t.to(device) for t in batch]
            outputs = model(
                input_ids=b_ids,
                attention_mask=b_mask,
                labels=b_lbl
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); model.zero_grad()

    # Evaluation
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            b_ids, b_mask, b_lbl = [t.to(device) for t in batch]
            logits = model(b_ids, attention_mask=b_mask).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(b_lbl.cpu().numpy())

    return trues, preds

def train_logistic(sensor_feats, labels):
    """Train a logistic regression on sensorimotor features."""
    scaler = StandardScaler()
    X = scaler.fit_transform(sensor_feats)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)
    return clf, scaler

def eval_clf(clf, scaler, sensor_feats, labels):
    X = scaler.transform(sensor_feats)
    preds = clf.predict(X)
    return labels, preds

def print_metrics(name, true, pred):
    print(f"--- {name} ---")
    print("Accuracy :", accuracy_score(true, pred))
    print("Precision:", precision_score(true, pred))
    print("Recall   :", recall_score(true, pred))
    print("F1       :", f1_score(true, pred))
    print()

def main():
    parser = argparse.ArgumentParser("Compare SENSE-LM, BERT, and LR")

    parser.add_argument("--step1_model", default="emanjavacas/MacBERTh",
                        help="Pretrained BERT for Step 1")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    args = parser.parse_args()

    # 1) Load and split
    df = pd.read_csv('../data/preprocessed/data_olf_gpt4_preprocessed.csv')
    df = df.dropna(subset=["normalized_text","contains_ref"])
    # df["label"] = df["label"].astype(int)

    def deciles(n):
        return [i * n // 5 for i in range(6)]
    max_sizes = [50, 100, 200, 500, 1000, 2000]

    for max_size in max_sizes:
        print(f"===============================")
        print(f"MAX SIZE : {max_size}")
        print(f"===============================")

        sizes = deciles(max_size)

        for size in sizes:
            try:
                if size <= max_size:
                    print(f"===============================")
                    print(f"STEP 1 - SIZE : {size} (OVER {max_size})")
                    print(f"===============================")



                    train_df, test_df = pd.read_csv(f'data/preprocessed/train_fewshot_mixed_{str(size)}_total_{str(max_size)}_aug_samples_original.csv'), pd.read_csv(
                        '../data/preprocessed/test_odeuropa_s1.csv')
                    train_df['normalized_text'] = train_df['normalized_text'].apply(lambda x: str(x))
                    test_df['normalized_text'] = test_df['normalized_text'].apply(lambda x: str(x))

                    # 2) Prepare sensorimotor norms
                    norms = load_sensorimotor_norms("code/sensorimotor_representation")

                    # 3) SENSE-LM Step 1
                    #    Uses exactly the same train/test split as above,
                    #    but wraps them in the repository’s DataLoader & train()
                    print("Training SENSE-LM Step 1…")
                    # Note: get_data_loader expects DataFrame columns 'text' and 'contains_ref', so we rename:
                    from sklearn.model_selection import StratifiedKFold

                    print("Running 5-fold Cross-Validation on training set...")

                    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
                    sense_scores = []
                    bert_scores = []
                    lr_scores = []

                    nlp = spacy.load("en_core_web_md")

                    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df["contains_ref"])):
                        print(f"\nFold {fold + 1}")
                        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
                        val_fold = test_df

                        # =======================
                        # 1. SENSE-LM Step 1
                        # =======================
                        print("Training SENSE-LM...")
                        train_loader = get_data_loader(train_fold, BertTokenizer.from_pretrained(args.step1_model),
                                                       norms)
                        val_loader = get_data_loader(val_fold, BertTokenizer.from_pretrained(args.step1_model), norms)

                        config = type("C", (), {})()
                        config.device = args.device
                        config.random_seed = 42
                        config.epochs_step1 = 10  # reduced for CV
                        config.learning_rate_step1 = 2e-5
                        config.learning_rate_step2 = 1e-8

                        model = SenseLM_BinaryClassifier.from_pretrained(
                            args.step1_model, num_labels=2, output_attentions=False, output_hidden_states=True
                        )
                        model.to(args.device)
                        optimizer = AdamW(model.parameters(), lr=config.learning_rate_step1,
                                          eps=config.learning_rate_step2)
                        total_steps = len(train_loader) * config.epochs_step1
                        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                                    num_training_steps=total_steps)

                        model, preds_sense, true_sense = train_sense_lm(
                            model, optimizer, scheduler, train_loader, val_loader, config
                        )
                        preds_sense = np.concatenate([np.argmax(p, 1) for p in preds_sense])
                        true_sense = np.concatenate(true_sense)
                        f1_sense = f1_score(true_sense, preds_sense)
                        sense_scores.append(f1_sense)
                        print(f"SENSE-LM F1 Score (Fold {fold + 1}): {f1_sense:.4f}")

                        # =======================
                        # 2. BERT
                        # =======================
                        print("Training BERT on fold...")
                        true_b, pred_b = train_bert(
                            train_fold.normalized_text.tolist(), train_fold.contains_ref.tolist(),
                            val_fold.normalized_text.tolist(), val_fold.contains_ref.tolist(),
                            model_name="bert-base-uncased",
                            epochs=10,
                            batch_size=16,
                            lr=2e-5,
                            device=args.device
                        )
                        f1_b = f1_score(true_b, pred_b)
                        bert_scores.append(f1_b)
                        print(f"BERT F1 Score (Fold {fold + 1}): {f1_b:.4f}")

                        def extract_feats(df_, norms):
                            """
                            Apply sensory_tag_sentence to each sentence in df_ to get
                            a sensorimotor feature vector.
                            """
                            feats = []
                            labels = df_["contains_ref"].tolist()
                            for txt in df_["normalized_text"].tolist():
                                # sensory_tag_sentence should return a 1-D numpy array or list
                                v = np.array(list(sensory_tag_sentence(txt, norms, nlp).values()))
                                feats.append(v)
                            X = np.vstack(feats)
                            y = np.array(labels)
                            return X, y

                        # =======================
                        # 3. Logistic Regression
                        # =======================
                        print("Extracting sensorimotor features for LR on fold...")
                        X_tr, y_tr = extract_feats(train_fold, norms)
                        X_val, y_val = extract_feats(val_fold, norms)
                        clf, scaler = train_logistic(X_tr, y_tr)
                        true_l, pred_l = eval_clf(clf, scaler, X_val, y_val)
                        f1_l = f1_score(true_l, pred_l)
                        lr_scores.append(f1_l)
                        print(f"LR F1 Score (Fold {fold + 1}): {f1_l:.4f}")

                    # =======================
                    # CV Summary
                    # =======================
                    print("\n====== 5-Fold Cross-Validation Summary ======")
                    print(f"SENSE-LM      F1: {np.mean(sense_scores):.4f} ± {np.std(sense_scores):.4f}")
                    print(f"BERT          F1: {np.mean(bert_scores):.4f} ± {np.std(bert_scores):.4f}")
                    print(f"LogReg (SM)   F1: {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")
                    print("=============================================\n")

            except Exception as e:
                print("ERROR {}".format(e))
                continue

if __name__ == "__main__":
    main()