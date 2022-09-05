import random 
import torch
import numpy as np
import logging
import json


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def make_acall_data(file_path=None, inference_text=None):
    #dict로 데이터를 넣으면 좋은 이유 : 코드 가독성이 좋아짐, 어떤 데이터를 넣는지 볼 수 있음
    '''해야하는 일 
    파일이 있으면 data에 (description, keyword, question) 을 key로 갖는 item을 담기
    파일이 없으면 text를 입력
    '''
    final_data = []
    if file_path: #for train
        with open(file_path) as f:
            print(file_path)
            data = json.load(f)['data']
        #data 수 == label 수
        #lable_list = ['페이지 : '+str(i) for i in range(len(data))]
        
        for item in data:
            description = item['description']
            keywords = item['keywords']
            questions = item['questions']
            label = int(item['page_num'])

            assert description != "" and keywords != "" and questions != "" and label != ""
            assert len(keywords) == len(questions)
            pair_num = len(keywords)
            
            for p in range(pair_num):
                keyword = keywords[p]
                question = questions[p]
                final_data.append({"description":description,
                             "keyword":keyword,
                             "question":question,
                             "label":label})    
    elif inference_text:
        for t in inference_text:
            final_data.append({"inference_text" : t})
            
    return final_data 
