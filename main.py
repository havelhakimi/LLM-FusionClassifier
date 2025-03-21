from DownstreamModel import DownstreamModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model_op_multi import Train_multi, Test_multi, Valid_multi
import argparse
import os
import torch
from MyDataset import MyDataset
import json
import datetime
import logging
from datetime import datetime



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda', type=str)
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-epochs', type=int, default=200, help='maximum number of epochs to train for')
    parser.add_argument('-SIGMA', type=float,default=1)
    parser.add_argument('-batch_size', type=int, nargs='?', default=10)
    parser.add_argument('-early-stop', type=int, default=10, help='Epoch before early stop.')
    parser.add_argument('-lr', type=float, nargs='?', default=1e-4)
    args = parser.parse_args()
    
    #current_time = datetime.now().strftime("%Y%m%d_%H%M%S") 
    #print(args)
    print(f'./exp/{args.dataset}_batch{args.batch_size}_SIGMA{args.SIGMA}_LR{args.lr}')
    #logging.basicConfig(filename=f'./exp/{args.current_time}_{args.dataset}_batch{args.batch_size}_SIGMA{args.SIGMA}.log', level=logging.INFO)
    #logging.getLogger().addHandler(logging.StreamHandler())

    device = args.device
    dataset = args.dataset
    epochs = args.epochs
    SIGMA = args.SIGMA
    batch_size = args.batch_size
    lr = args.lr

    class_num = {'ED':32, 'go_emotion':27,} # 'mr':2, 'agnews':4, 'r8':8, 'r52':52}
    class_num = class_num[dataset]
    
    #l_dataset_path = f'../data/{dataset}/dataset_tensor/llama2_7b/'  #f'/datallama2_embedding/{dataset}/dataset_tensor/'
    #b_dataset_path = f'../data/{dataset}/dataset_tensor/bert/'  #f'bert_embedding/{dataset}/dataset_tensor/'
    #r_dataset_path = f'../data/{dataset}/dataset_tensor/roberta/' #f'roberta_embedding/{dataset}/dataset_tensor/'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    l_dataset_path = os.path.join(script_dir, "data", dataset, "dataset_tensor/llama2_7b_chat") # llama2_7b
    b_dataset_path = os.path.join(script_dir, "data", dataset, "dataset_tensor/bert")
    r_dataset_path = os.path.join(script_dir, "data", dataset, "dataset_tensor/roberta")

    
    #l_dataset_path = f'/scratch/gaurav_k.iitr/ashish/LLMembed_fec/data/{dataset}/dataset_tensor/llama2_7b/'  #f'/datallama2_embedding/{dataset}/dataset_tensor/'
    #b_dataset_path = f'/scratch/gaurav_k.iitr/ashish/LLMembed_fec//data/{dataset}/dataset_tensor/bert/'  #f'bert_embedding/{dataset}/dataset_tensor/'
    #r_dataset_path = f'/scratch/gaurav_k.iitr/ashish/LLMembed_fec//data/{dataset}/dataset_tensor/roberta/' #f'roberta_embedding/{dataset}/dataset_tensor/'
    mode = 'train'
    train_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)  
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    mode='valid'
    valid_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)   
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    mode = 'test'
    test_data = MyDataset(mode, l_dataset_path, b_dataset_path, r_dataset_path)   
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DownstreamModel(class_num, SIGMA).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr)

    best_valid_wf1,best_valid_acc, best_test_wf1,best_test_acc =-1, -1, -1,-1
    best_valid_loss = 1e5
    early_stop_count =0
    print('training ...')
    for epoch in range(epochs):
        if early_stop_count >= args.early_stop:
            print("Early stop!")
            break

        print(f'--------------------------- epoch {epoch} ---------------------------')
        train_loss, train_acc, train_wf1 = Train_multi(train_loader, device, model, loss_fn, optimizer)
        print(f'Train loss: {train_loss} | Acc: {train_acc} | F1: {train_wf1}')

        valid_loss, valid_acc, valid_wf1 = Valid_multi(valid_loader, device, model, loss_fn)
        print(f'Valid loss: {valid_loss} | Acc: {valid_acc} | F1: {valid_wf1}')

        test_loss, test_acc, test_wf1 = Test_multi(test_loader, device, model, loss_fn)
        print(f'Test loss: {test_loss} | Acc: {test_acc} | F1: {test_wf1}')

        early_stop_count += 1

        # Store test metrics only when best validation loss OR best validation F1 occurs
        if valid_loss < best_valid_loss:
            early_stop_count = 0
            best_valid_loss = valid_loss
            best_test_acc = test_acc  # Save test accuracy from this epoch
            best_test_wf1 = test_wf1  # Save test F1-score from this epoch

        if valid_wf1 > best_valid_wf1:
            early_stop_count = 0
            best_valid_wf1 = valid_wf1
            best_valid_acc = valid_acc  # Track best valid acc (optional)
            best_test_acc = test_acc  #  Save test accuracy from this epoch
            best_test_wf1 = test_wf1  # Save test F1-score from this epoch

                
    print()
    print('Final Test Results ...')
    #test_loss,test_acc,test_wf1=Test_multi(test_loader, device, model, loss_fn)
    print(f'| Acc : {best_test_acc}| F1: { best_test_wf1}')
