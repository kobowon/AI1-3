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
                            shuffle=True,
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

    preds = list(np.argmax(preds, axis=1)) #(b)
    #print(preds)
    
    label_list = [str(i) for i in range(1, args.page_num+1)]
    id2label = {str(i):label for i, label in enumerate(label_list)}
    
    preds_label = [id2label[str(p)] for p in preds]
    print(preds_label)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, default='/data/bowon_ko/TBERT_Distil/220110/finetune/acall/version_1') 
    parser.add_argument("--page_num", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_batch_size", type=int, default=1)
    
    args = parser.parse_args()
    
    
    init_logger()
    set_seed(args)
    infer(args, ["가변 길이의 문자열을 저장하는 데이터 타입은 무엇인가?","세 개 이상의 테이블을 조인하는 방법은?","동의어를 공용으로 정의하여 모든 사용자가 사용 가능한 동의어를 무엇이라 부르는가?"])
    #2,17,13
        
        
        
     #'사용자 정의형은 어디에 저장 되는가','티베로에서 제공하는 가장 큰 데이터 타입은 무엇인가','형식 문자열은 숫자형과 날짜형을 문자열로 변환하는 형식을 정의한 것이다. 또한, 변환 뿐만 아니라 반대로 변환된 숫자형과 날짜형을 다시 복구하는데 필요하다. 정해진 시간 형식이 아니거나 숫자와 문자를 동시에 포함하는 문자열은 날짜형과 숫자형으로 변환할 수 없다.'   
                        
                            
    