# -*- coding: utf-8 -*-
import os
import torch
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import trange
#from datasets import load_dataset
import argparse
import pandas as pd

def rep_extract(task, mode, device, sents, labels):
   
    model_id = "meta-llama/Llama-2-7b-chat-hf" #Choices--> meta-llama/Llama-2-7b-chat-hf;meta-llama/Llama-2-7b-hf
    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, )#use_auth_token=True)

    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "right"

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "revision": 'main',
        "use_auth_token": None,
        "output_hidden_states": True
    }
    model_config = AutoConfig.from_pretrained(model_id, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        device_map=device,
        torch_dtype=torch.float16)

    #model = model.to(torch.float32)  # Force model computation to FP32

    model.eval()

    max_len =200 # 512
    sents_reps = []
    step = 2
    
    # for idx in trange(0, 20, step):
    for idx in trange(0, len(sents), step):
        idx_end = idx + step
        if idx_end > len(sents):
            idx_end = len(sents)        
        sents_batch = sents[idx: idx_end]

        sents_batch_encoding = tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)

        # Remove token_type_ids as they're unnecessary for causal LM models
        sents_batch_encoding = {k: v for k, v in sents_batch_encoding.items() if k != "token_type_ids"}
        # Move tensors to the desired device
        sents_batch_encoding = {k: v.to(device) for k, v in sents_batch_encoding.items()}
        #print(sents_batch_encoding.keys())
        with torch.no_grad():
            batch_outputs = model(**sents_batch_encoding)

            #print((batch_outputs))
            #print(len(batch_outputs.hidden_states))
            reps_batch_5L = []
            #print((batch_outputs.hidden_states[-1]).shape)
            for layer in range(-1, -6, -1):
                reps_batch_5L.append(torch.mean(batch_outputs.hidden_states[layer], axis=1))    
            reps_batch_5L = torch.stack(reps_batch_5L, axis=1)

        sents_reps.append(reps_batch_5L.cpu())
    sents_reps = torch.cat(sents_reps)

    
    labels = torch.tensor(labels.tolist(), dtype=torch.long)
    print(sents_reps.shape)
    print(labels.shape)

    path = f'./data/{task}/dataset_tensor/llama2_7b_chatt/'
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
    #sents=sents[:5]
    #labels=labels[:5]
    rep_extract(task, mode, device, sents, labels)

 