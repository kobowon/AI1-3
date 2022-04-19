import logging
import os
import json
import random
import time
import datetime
import argparse
from attrdict import AttrDict
from fastprogress.fastprogress import master_bar, progress_bar

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from transformers import (
    BertTokenizer, #wordpiece
    get_linear_schedule_with_warmup,
    RobertaForSequenceClassification,
    AdamW,
    RobertaConfig,
)
from dataset import AcallDataset
from src import set_seed, init_logger, make_acall_data
from tokenizer import KobortTokenizer

logger = logging.getLogger(__name__)


def train(train_file_path, page_num):
    with open(os.path.join('config','acall_train_config.json')) as f:
        args = AttrDict(json.load(f))
    args.page_num = page_num
    logger.info(f"Training parameters {args}")
    
    init_logger()
    set_seed(args)
    
    start_time = time.time()
    #Set GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Load tokenizer
    tokenizer = KobortTokenizer("wp-mecab").tokenizer
    #read data
    data = make_acall_data(file_path=train_file_path)
    #make data object
    kwargs = (
        {"num_workers":torch.cuda.device_count(), "pin_memory":True} if torch.cuda.is_available()
        else {}
    )
    data_obj = AcallDataset(tokenizer=tokenizer,
                            dataset=data,
                            page_num=page_num,
                            max_seq_len=args.max_seq_len,
                            batch_size=args.train_batch_size,
                            shuffle=False,
                            **kwargs)
    train_dataloader = data_obj.loader
    total_steps = len(train_dataloader) * args.num_train_epochs
    #Load model
    logger.info("load model...")
    labels = ['페이지 : '+str(i) for i in range(1,page_num+1)]
    model = RobertaForSequenceClassification.from_pretrained(args.model_path,
                                                            num_labels = page_num,
                                                            id2label = {str(i):label for i, label in enumerate(labels)},
                                                            label2id = {label:i for i, label in enumerate(labels)})
    
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                int(total_steps * args.warmup_proportion),
                                                num_training_steps=total_steps)
    
    train_start(args, train_dataloader, model, optimizer, scheduler)
    train_end(args, model)
    
    logger.info("total training time : {:.4f}sec".format(time.time() - start_time))
    logger.info("training complete")
    
def train_start(args, train_dataloader, model, optimizer, scheduler):
    model.zero_grad()
    
    epoch_iterator = master_bar(range(args.num_train_epochs))
    for epoch in epoch_iterator:
        batch_iterator = progress_bar(train_dataloader, parent=epoch_iterator)
        step_num = len(train_dataloader)
        
        train_epoch(args, batch_iterator, model, optimizer, scheduler, epoch, step_num)


def train_end(args, model):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    #checkpoint file & argument 저장
    model.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    #inference를 위한 config 저장
    infer_config =  {'page_num' : args.page_num,
                     'max_seq_len' : args.max_seq_len, 
                     'model_path' : args.output_dir,
                     'seed' : args.seed}
    
    infer_config_path = os.path.join('config','acall_infer_config.json')
    with open(infer_config_path, 'w', encoding='utf-8') as f:
        json.dump(infer_config,f)
    
    
    logger.info(f"save final model checkpoint file at {args.output_dir}...")
    logger.info(f"save final model arguments file at {args.output_dir}...")
    logger.info(f"save inference config file at {infer_config_path}...")
        
    
def train_epoch(args, train_dataloader, model, optimizer, scheduler, epoch, step_num):
    train_loss = 0.0
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        loss = train_step(args, model, optimizer, scheduler, step, batch)
        train_loss += loss.item()
   
    average_train_loss = train_loss / step_num
    logger.info(f"averate train loss : {average_train_loss}")
        
    #To Do
    if args.do_eval:
        logger.info("start validation...")
        pass

def train_step(args, model, optimizer, scheduler, step, batch):
    model.zero_grad()
    batch = tuple(tensor.to(args.device) for tensor in batch)
    inputs = {"input_ids": batch[0],
              "attention_mask": batch[1],
              "labels": batch[2],
              "token_type_ids": None}
    outputs = model(**inputs)
    loss = outputs["loss"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
    optimizer.step()
    scheduler.step()
    
    return loss
