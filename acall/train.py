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


def train(args):
    start_time = time.time()
    #Set GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Load tokenizer
    tokenizer = KobortTokenizer("wp-mecab").tokenizer
    
    #read data
    data = make_acall_data(file_path=os.path.join(args.data_dir, args.train_file))
    
    #make data object
    kwargs = (
        {"num_workers":torch.cuda.device_count(), "pin_memory":True} if torch.cuda.is_available()
        else {}
    )
    data_obj = AcallDataset(tokenizer=tokenizer,
                            dataset=data,
                            page_num=args.page_num,
                            max_seq_len=args.max_seq_len,
                            batch_size=args.train_batch_size,
                            shuffle=False,
                            **kwargs)
    train_dataloader = data_obj.loader
    total_steps = len(train_dataloader) * args.num_train_epochs
    
    #Load model
    logger.info("load model...")
    labels = [str(i) for i in range(1,args.page_num+1)]
    model = RobertaForSequenceClassification.from_pretrained(args.model_path,
                                                            num_labels = args.page_num,
                                                            id2label = {str(i):label for i, label in enumerate(labels)},
                                                            label2id = {label:i for i, label in enumerate(labels)})
    
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                int(total_steps * args.warmup_proportion),
                                                num_training_steps=total_steps)
    
    model.zero_grad()
    
    epoch_iterator = master_bar(range(args.num_train_epochs))
    for epoch in epoch_iterator:
        train_loss = 0.0
        
        model.train()
        batch_iterator = progress_bar(train_dataloader, parent=epoch_iterator)
        for step, batch in enumerate(batch_iterator):
            # print(len(batch), batch)
            # print('@'*50)
            # return
            model.zero_grad()
            #collate_fn result : t_input_ids, t_input_attention_mask, t_input_label_ids
            batch = tuple(tensor.to(args.device) for tensor in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[2],
                      "token_type_ids": None}
            outputs = model(**inputs)
            
            loss = outputs["loss"]
            loss.backward()
            train_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
        
        step_per_epoch = len(train_dataloader)
        average_train_loss = train_loss / step_per_epoch
        logger.info(f"averate train loss : {average_train_loss}")
        
        #To Do
        if args.do_eval:
            logger.info("start validation...")
            pass
        
    logger.info(f"save final model checkpoint file at {args.output_dir}...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.save_pretrained(args.output_dir)
    logger.info(f"save final model arguments file at {args.output_dir}...")
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    total_train_loss = train_loss / total_steps
    logger.info("total training time : {:.4f}sec".format(time.time() - start_time))
    
    logger.info("training complete")
    
    
if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default='/data/bowon_ko/TBERT_Distil/220110/final') 
    parser.add_argument("--output_dir", type=str, default='/data/bowon_ko/TBERT_Distil/220110/finetune/acall/version_1') 
    parser.add_argument("--page_num", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default='/home/ubuntu/bowon_ko/acall/data')
    parser.add_argument("--train_file", type=str, default='acall_data.json')
    parser.add_argument("--dev_file",type=str, default=None)
    parser.add_argument("--test_file",type=str, default=None)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_proportion", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default = 1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    init_logger()
    set_seed(args)
    
    if args.do_train:
        train(args)
    if args.do_eval:
        logger.info("future work, evaluation metric not fixed")
                            
                            
    