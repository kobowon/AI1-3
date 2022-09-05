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


def infer(args, text):#inference_text가 List일 수도 있음(추후 확장 고려)
    #Set GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Load tokenizer
    tokenizer = KobortTokenizer("wp-mecab").tokenizer
    
    #read data
    data = make_acall_data(file_path=None, inference_text=text)
    
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

def infer_simple(args, text):#inference_text : 하나의 text
    #Set GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Load tokenizer
    tokenizer = KobortTokenizer("wp-mecab").tokenizer
    
    #read data
    data = make_acall_data(file_path=None, inference_text=[text])
    
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
    labels = ['페이지 : '+str(i) for i in range(1,args.page_num+1)]
    id2label = {str(i):label for i, label in enumerate(labels)}
    model = RobertaForSequenceClassification.from_pretrained(args.model_path,
                                                            num_labels = args.page_num,
                                                            id2label = {str(i):label for i, label in enumerate(labels)},
                                                            label2id = id2label)
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
        logits = logits.detach().cpu().numpy()[0]
        #check logit
        print(logits)
        preds = np.argmax(logits, axis=0)
        preds_label = id2label[str(preds)]
        
        return preds_label
    
    print(f'경과 시간 : {time.time() - start_time}')
  
  
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default='/data/bowon_ko/TBERT_Distil/220110/finetune/acall/version_2') 
    parser.add_argument("--page_num", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_batch_size", type=int, default=2)
    
    args = parser.parse_args()
    
    
    init_logger()
    set_seed(args)
    infer(args, ["정수나 실수를 저장하는 데이터 타입은 무엇인가?","티베로에서는 어떤 숫자 타입을 지원하는가?", "특정 컬럼을 빠르게 검색 할 수 있도록 해주는 데이터 구조는 무엇인가?", "모든 테이블의 기본 키 컬럼에 자동으로 생성되는 것은 무엇인가?", "티베로의 함수는 무엇으로 구분되는가?", "대부분의 함수는 하나 이상의 파라미터를 입력으로 받고 무엇을 반환하는가?","서로 다른 두 테이블의 컬럼을 비교하는 조인 조건은 어느 절에서 이루어지는가?",  "세 개 이상의 테이블을 조인하는 방법은?", "CONNECT BY와 WHERE가 하나의 SELECT에 동시에 있을 경우 WHERE가 먼저 수행 되는 경우는?", "상하 관계를 유지한 상태에서 정렬하는 절은?"])
    #, "변환값이 NUMBER 타입인 경우 컬럼의 무엇과 무엇의 범위 내여야 하는가?", "티베로에서는 테이블 간의 조인 순서는 무엇이 정하는가?","CONNECT BY와 WHERE가 하나의 SELECT에 동시에 있을 경우 WHERE가 먼저 수행 되는 경우는?"
    answer = infer_simple(args,"상하 관계를 유지한 상태에서 정렬하는 절은?")
    print(answer)
                            
    