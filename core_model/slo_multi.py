#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import torch  
torch.manual_seed(3407)
import random
random.seed(3407)
import numpy as np
np.random.seed(3407)

from transformers import XLMRobertaTokenizerFast              
from transformers import XLMRobertaForTokenClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import json
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import timeit

def convert_type(df):
    for i in range(len(df)):
        df['word'].iloc[i] = eval(df['word'].iloc[i])
        df['labels'].iloc[i] = eval(df['labels'].iloc[i])
    return df

def label_mapping(df):
    dct = {'O':'n','B':'B-T','I':'T'}
    for i in range(len(df)):
        df.labels.iloc[i] = [dct[k] for k in df.labels.iloc[i]]
    df = list(zip(*map(df.get, df)))
    return df

def get_data(trainings_data, val_data, test_data):
    #train
    train_tags=[tup[1] for tup in trainings_data]
    train_texts=[tup[0] for tup in trainings_data]

    #val
    val_tags=[tup[1] for tup in val_data]
    val_texts=[tup[0] for tup in val_data]

    #test
    test_tags=[tup[1] for tup in test_data]
    test_texts=[tup[0] for tup in test_data]
    return train_tags, train_texts, val_tags, val_texts, test_tags, test_texts

def tokenize_and_align_labels(texts, tags):
    # lowercase
    texts = [[x.lower() for x in l] for l in texts]
    tokenized_inputs = tokenizer(
      texts,
      padding=True,
      truncation=True,
      # We use this argument because the texts in our dataset are lists of words (with a label for each word).
      is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs  



# create dataset that can be used for training with the huggingface trainer
class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# return the extracted terms given the token level prediction and the original texts
def extract_terms(token_predictions, val_texts):
    extracted_terms = set()
    # go over all predictions
    for i in range(len(token_predictions)):
        pred = token_predictions[i]
        txt  = val_texts[i]
        # print(len(pred), len(txt))
        for j in range(len(pred)):
          # if right tag build term and add it to the set otherwise just continue
          # print(pred[j], txt[j])
            if pred[j]=="B-T":
                term=txt[j]
                for k in range(j+1,len(pred)):
                    if pred[k]=="T": term+=" "+txt[k]
                    else: break
                extracted_terms.add(term)
    return extracted_terms

#compute the metrics TermEval style for Trainer
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    extracted_terms = extract_terms(true_predictions, val) # ??????
    extracted_terms = set([item.lower() for item in extracted_terms])
    gold_set=gold_validation      # ??????

    # print(extracted_terms)
    true_pos=extracted_terms.intersection(gold_set)
    # print("True pos", true_pos)
    recall=len(true_pos)/len(gold_set)
    precision=len(true_pos)/len(extracted_terms)
    f1 = 2*(precision*recall)/(precision+recall) if precision + recall != 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def computeTermEvalMetrics(extracted_terms, gold_df):
    #make lower case cause gold standard is lower case
    extracted_terms = set([item.lower() for item in extracted_terms])
    gold_set=set(gold_df)
    true_pos=extracted_terms.intersection(gold_set)
    recall=round(len(true_pos)*100/len(gold_set),2)
    precision=round(len(true_pos)*100/len(extracted_terms),2)
    fscore = round(2*(precision*recall)/(precision+recall),2)

    print("Extracted",len(extracted_terms))
    print("Gold",len(gold_set))
    print("Intersection",len(true_pos))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", fscore)

    print(str(len(extracted_terms))+ ' | ' + str(len(gold_set)) +' | ' + str(len(true_pos)) +' | ' + str(precision)+' & ' +  str(recall)+' & ' +  str(fscore))
    return len(extracted_terms), len(gold_set), len(true_pos), precision, recall, fscore

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-train1", "--train1_path",type=str,  help="train path 1")
    parser.add_argument("-train2", "--train2_path",type=str, help="train path 2")
    parser.add_argument("-val", "--val_path", type=str, help="validation data path")
    parser.add_argument("-test", "--test_path",type=str, help="test data path")
    parser.add_argument("-gold_val", "--gold_val_path", type=str, help="validation data path")
    parser.add_argument("-gold_test", "--gold_test_path",type=str, help="test data path")
    parser.add_argument("-store", "--store", type=str,help="store the model")
    parser.add_argument("-preds", "--pred_path",type=str, help="save predicted candidate list")
    parser.add_argument("-log", "--log_path",type=str, help="save logs")
    parser.add_argument("-multi", "--multi",type=bool, default=True, help="mono or multi training option")
    parser.add_argument("-version", "--ver", type=str, default='ann', help="ann or nes")
    args = parser.parse_args()
    
    start = timeit.default_timer()
    if not os.path.exists(args.store):
        os.makedirs(args.store) 
    
    path = './'
    
    train1_data = label_mapping(convert_type(pd.read_csv(args.train1_path)[['word','labels']]))     
    train2_data = label_mapping(convert_type(pd.read_csv(args.train2_path)[['word','labels']])) 
    
    if args.multi == True:
        # EN
        with open(path+'processed_data/en'+ args.ver + 'corp.pkl', 'rb') as f:
            train_data_corp_en = pickle.load(f)

        with open(path+'processed_data/en'+ args.ver + 'wind.pkl', 'rb') as f:
            train_data_wind_en = pickle.load(f)

        with open(path+'processed_data/en'+ args.ver + 'equi.pkl', 'rb') as f:
            train_data_equi_en = pickle.load(f)

        with open(path+'processed_data/en'+ args.ver + 'htfl.pkl', 'rb') as f:
            train_data_htfl_en = pickle.load(f)

        # FR
        with open(path+'processed_data/fr'+ args.ver + 'corp.pkl', 'rb') as f:
            train_data_corp_fr = pickle.load(f)

        with open(path+'processed_data/fr'+ args.ver + 'wind.pkl', 'rb') as f:
            train_data_wind_fr = pickle.load(f)

        with open(path+'processed_data/fr'+ args.ver + 'equi.pkl', 'rb') as f:
            train_data_equi_fr = pickle.load(f)

        with open(path+'processed_data/fr'+ args.ver + 'htfl.pkl', 'rb') as f:
            train_data_htfl_fr = pickle.load(f)

        # NL
        with open(path+'processed_data/nl'+ args.ver + 'corp.pkl', 'rb') as f:
            train_data_corp_nl = pickle.load(f)
        with open(path+'processed_data/nl'+ args.ver + 'wind.pkl', 'rb') as f:
            train_data_wind_nl = pickle.load(f)

        with open(path+'processed_data/nl'+ args.ver + 'equi.pkl', 'rb') as f:
            train_data_equi_nl = pickle.load(f)

        with open(path+'processed_data/nl'+ args.ver + 'htfl.pkl', 'rb') as f:
            train_data_htfl_nl = pickle.load(f)
     
        trainings_data = train_data_corp_en + train_data_wind_en + train_data_equi_en + train_data_htfl_en + train_data_corp_fr + train_data_wind_fr + train_data_equi_fr + train_data_htfl_fr + train_data_corp_nl + train_data_wind_nl + train_data_equi_nl + train_data_htfl_nl +  train1_data  +  train2_data
    else:
        trainings_data =  train1_data  +  train2_data
        
    
    val_data = label_mapping(convert_type(pd.read_csv(args.val_path)[['word','labels']]))        
    test_data = label_mapping(convert_type(pd.read_csv(args.test_path)[['word','labels']])) 
    
    gold_set_for_validation= set(pd.read_csv(args.gold_val_path, header=None, delimiter="\t", names=["Term","Label"])["Term"])
    gold_set_for_test = set(pd.read_csv(args.gold_test_path, header=None, delimiter="\t", names=["Term","Label"])["Term"])

    train_tags, train_texts, val_tags, val_texts, test_tags, test_texts = get_data(trainings_data, val_data, test_data)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    #align labels with tokenization from XLM-R

    label_list=["n", "B-T", "T"]
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels=len(label_list)

    train_input_and_labels = tokenize_and_align_labels(train_texts, train_tags)
    val_input_and_labels = tokenize_and_align_labels(val_texts, val_tags)
    test_input_and_labels = tokenize_and_align_labels(test_texts, test_tags)

    train_dataset = OurDataset(train_input_and_labels, train_input_and_labels["labels"])
    val_dataset = OurDataset(val_input_and_labels, val_input_and_labels["labels"])
    test_dataset = OurDataset(test_input_and_labels, test_input_and_labels["labels"])

    training_args = TrainingArguments(
        output_dir= args.store,          # output directory
        num_train_epochs=20,              # total # of training epochs
        per_device_train_batch_size=32,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        # warmup_steps=0,                  # number of warmup steps for learning rate scheduler
        # weight_decay=0,                  # strength of weight decay
        # weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='./logs',            # directory for storing logs
        evaluation_strategy= 'steps', # or use epoch here
        # save_total_limit = 5,
        # save_steps = 1000,
        eval_steps = 500,
        load_best_model_at_end=True,   #loads the model with the best evaluation score
        metric_for_best_model="f1",
        greater_is_better=True
    )

    # initialize model
    model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)

    val = val_texts
    gold_validation =  gold_set_for_validation

    # initialize huggingface trainer
    trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        )

    # train
    trainer.train()

    val = test_texts
    gold_validation =  gold_set_for_test

    #test
    test_predictions, test_labels, test_metrics = trainer.predict(test_dataset)
    test_predictions = np.argmax(test_predictions, axis=2)
    # Remove ignored index (special tokens)
    true_test_predictions = [
        [label_list[p] for (p, l) in zip(test_prediction, test_label) if l != -100]
        for test_prediction, test_label in zip(test_predictions, test_labels)
    ]

    test_extracted_terms = extract_terms(true_test_predictions, test_texts)
    extracted, gold, true_pos, precision, recall, fscore = computeTermEvalMetrics(test_extracted_terms, set(gold_set_for_test))

        
    with open(args.log_path, 'w') as f:
        f. write(json.dumps([extracted, gold, true_pos, precision, recall, fscore]))
        
    with open(args.pred_path, 'w') as f:
        f. write(json.dumps(list(test_extracted_terms)))
        
    stop = timeit.default_timer()
    print('Time: ', stop - start) 