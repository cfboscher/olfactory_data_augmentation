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

from utils.prepare_data import prepare_data_step_2

from sensorimotor_representation.load_sensorimotor_norms import load_sensorimotor_norms

from step_2.tokenizer import load_tokenizer, tokenize_train_data, tokenize_test_data

from step_2.train import train
from step_2.test import test, eval_roberta

from step_2.lexifield import apply_lexifield, get_lexifield_terms, eval_lexifield
from step_2.heuristic import apply_heuristic, load_glove_models, eval_heuristic

from step_2.metrics import get_prec, get_rec

from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser("Compare SENSE-LM, BERT, and LR")

    parser.add_argument("--step1_model", default="emanjavacas/MacBERTh",
                        help="Pretrained BERT for Step 1")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    args = parser.parse_args()


    sizes = [20, 50, 100,  200, 300, 360 ]

    # sizes = [1750]
    class Config:
        def __init__(self):
            self.device = "cuda"
            self.data_path = "../dataset"
            self.dataset = "gpt4_olf"
            self.random_seed = 42
            self.steps = "2"
            self.pretrained_parameters_step1 = "bert-base-uncased"
            self.epochs_step1 = 20
            self.learning_rate_step1 = 2e-5
            self.epsilon_step1 = 1e-8
            self.epochs_step2 = 15
            self.learning_rate_step2 = 2e-5
            self.tokenizer_max_len_step2 = 303
            self.epsilon_step2 = 1e-8
            self.T = 4.5
            self.U = 0.75

    config = Config()

    print("Loading Glove Models")
    glove_models = load_glove_models()
    for size in sizes:
            print(f"===============================")
            print(f"SIZE : {size}")
            print(f"===============================")
            print('Running Step 2')


            print('Splitting data...')
            train_data, test_data = pd.read_csv(f'data/preprocessed/train_fewshot_{str(size)}_s2.csv'), pd.read_csv(
                '../dataset/preprocessed/test_odeuropa_s2.csv')
            train_data = train_data.reset_index()
            test_data = test_data.reset_index()

            print('Loading Sensorimotor Norms...')
            sensorimotor_norms = load_sensorimotor_norms("code/sensorimotor_representation")

            # STEP 2.1

            # Load the RoBERTa tokenizer.
            print('Loading Tokenizer...')
            tokenizer, sensoriality_id = load_tokenizer()

            print('Tokenizing Train Data')
            input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = tokenize_train_data(tokenizer,
                                                                                                      train_data,
                                                                                                      sensoriality_id,
                                                                                                      config)

            print('Tokenizing Test Data')
            input_ids_t, attention_mask_t, token_type_ids_t = tokenize_test_data(tokenizer, test_data, sensoriality_id,
                                                                                 config)

            print("STEP 2.1 ")
            print('Train Model for Step 2.1')
            model = train(tokenizer, train_data,
                          input_ids, input_ids_t,
                          attention_mask, attention_mask_t,
                          token_type_ids, token_type_ids_t, start_tokens, end_tokens,
                          config)

            for i in range(10):
                print(f"EVAL for fold {i}")
                model.load_weights('%s-roberta-%i.h5' % ("v0", i))
                print('Test Models for Step 2.1')
                test_data = test(tokenizer, model, i, test_data, test_data, input_ids_t, attention_mask_t,
                                 token_type_ids_t, config)

                test_data = eval_roberta(test_data)

                test_data.to_csv(f'{config.dataset}_s1.csv')

                # STEP 2.2
                print("STEP 2.2")
                print("Expanding predictions with Lexifield")
                lexifield_terms = get_lexifield_terms(config)
                test_data['pred_token'] = test_data.apply(
                    lambda x: apply_lexifield(x.clean_sentence, x.pred_token, lexifield_terms), axis=1)

                test_data = eval_lexifield(test_data)

                test_data.to_csv(f'{config.dataset}_s2.csv')

                # STEP 2.3
                print("STEP 2.3")
                print("Expanding predictions with Heuristic")
                test_data['pred_token'] = test_data.apply(
                    lambda x: apply_heuristic(x.clean_sentence, x.pred_token, sensorimotor_norms, glove_models, config),
                    axis=1)

                test_data = eval_heuristic(test_data)

                test_data.to_csv(f'{config.dataset}_s3.csv')
if __name__ == "__main__":
    # print(torch.cuda.is_available())
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    main()