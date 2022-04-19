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
from tqdm import tqdm
logger = logging.getLogger(__name__)


def infer(text_list, batch_size):#inference_text가 List일 수도 있음(추후 확장 고려)
    init_logger()
    
    infer_config_path = os.path.join('config','acall_infer_config.json')
    with open(infer_config_path) as f:
        args = AttrDict(json.load(f))
    set_seed(args)
    
    #Set infer batch size
    args.infer_batch_size = batch_size
    
    #Set GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Inference config {args}")
    
    #Load tokenizer
    tokenizer = KobortTokenizer("wp-mecab").tokenizer
    
    #read data
    data = make_acall_data(file_path=None, inference_text=text_list)
    
    #make data object
    kwargs = (
        {"num_workers":torch.cuda.device_count(), "pin_memory":True} if torch.cuda.is_available()
        else {}
    )
    data_obj = AcallDataset(tokenizer=tokenizer,
                            dataset=data,
                            page_num=args.page_num,
                            max_seq_len=args.max_seq_len,
                            batch_size=args.infer_batch_size,
                            shuffle=False,
                            **kwargs)
    dataloader = data_obj.loader
    
    #Load model
    logger.info("load model...")
    labels = [str(i) for i in range(1,args.page_num+1)]
    model = RobertaForSequenceClassification.from_pretrained(args.model_path,
                                                            num_labels = args.page_num,
                                                            id2label = {str(i):label for i, label in enumerate(labels)},
                                                            label2id = {label:i for i, label in enumerate(labels)})
    model.to(args.device)
    model.eval()
    
    preds = None
    label_ids = None
    
    start_time = time.time()
    
    for batch in tqdm(dataloader):
        #collate_fn result : t_input_ids, t_input_attention_mask, t_input_label_ids
        batch = tuple(tensor.to(args.device) for tensor in batch)
        inputs = {"input_ids": batch[0],
                  "attention_mask": batch[1],
                  "labels": None,
                  "token_type_ids": None}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs['logits']
        logits = logits.detach().cpu().numpy()

        if preds is None:
            preds = logits
        else:
            preds = np.append(preds, logits, axis=0)#(b,page_num)

    #check logit
    print(preds)
    preds = list(np.argmax(preds, axis=1)) #(b)
    #print(preds)
    
    label_list = [str(i) for i in range(1, args.page_num+1)]
    id2label = {str(i):label for i, label in enumerate(label_list)}
    
    preds_label = [id2label[str(p)] for p in preds]
    print(preds_label)
    
    print(f'경과 시간 : {time.time() - start_time}')
    
    return preds_label

    