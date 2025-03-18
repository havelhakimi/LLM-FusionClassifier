# -*- coding: utf-8 -*-
import os
import torch
import json
from transformers import RobertaTokenizer, RobertaModel
from tqdm import trange
#from datasets import load_dataset
import argparse
import pandas as pd

def rep_extract(task, mode, device, sents, labels):
    #model_path = '/home/linux/hf_model/bert-large-uncased'
    model_path = 'roberta-large' # roberta-base 
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path).to(device)
    model.eval()

    max_len =200 # 512
    sents_reps = []
    step = 50
    
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)        
        sents_batch = sents[idx: idx_end]

        sents_batch_encoding = tokenizer(sents_batch, padding='longest', truncation= True, max_length=max_len, return_tensors='pt') #
        sents_batch_encoding = sents_batch_encoding.to(device)
        
        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding)
            reps_batch =  batch_outputs.last_hidden_state[:, 0, :]  
        sents_reps.append(reps_batch.cpu())
    sents_reps = torch.cat(sents_reps)

    
    labels = torch.tensor(labels.tolist(), dtype=torch.long)
    print(sents_reps.shape)
    print(labels.shape)

    path = f'./data/{task}/dataset_tensor/roberta/'
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(sents_reps.to('cpu'), path + f'{mode}_sents.pt')
    torch.save(labels, path + f'{mode}_labels.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str)
    parser.add_argument('-task', type=str)   # sst2, mr, agnews, r8, r52
    parser.add_argument('-mode', type=str) # train,test,val  
    args = parser.parse_args()
    device = args.device
    task = args.task
    mode=args.mode
    
    print(args)
    assert mode in ['train', 'valid', 'test']
    df = pd.read_csv(f'./data/{task}/{mode}.csv')
    sents=df.text
    sents = df.text.astype(str).tolist() # Extract sentences and ensure they are strings
    labels=df.label.values
    #print(labels)
    rep_extract(task, mode, device, sents, labels)

 