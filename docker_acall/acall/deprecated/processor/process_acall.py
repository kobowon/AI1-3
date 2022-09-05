'''
작성일 : 2022.03.21
작성자 : 고보원
목적 : Acall 내부 테스트용
'''
import os
import sys
import logging
import json

from konlpy.tag import Mecab
import torch
from torch.utils.data import TensorDataset
from keras.preprocessing.sequence import pad_sequences
logger = logging.getLogger(__name__)


class InputExample:
    """
    A single example for sequence classification
    """
    #description에 대응되는 내용을 물어봐도
    #질문에 대응되는 내용을 물어봐도
    #keyword로만 물어봐도
    #모두 대응되는 superapp 페이지로 가야함
    def __init__(self, description, keyword, question, label):
        self.description = description
        self.keyword = keyword
        self.question = question
        self.label = label #superapp page num

class InputFeatures:
    """
    A single feature 
    """
    def __init__(self, input_ids, attention_mask, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label = label
        
#data input 구조를 <s>description</s>keyword</s>question</s> 이렇게 해볼까
#데이터 형식 : [0] : description, [1] : keyword, [2] question
#먼저 도전

#아니면 <s>description</s> , <s>keyword</s> , <s>question</s> 이런 식으로 3개의 데이터로 만들어볼까
#데이터 형식 : description 파일 따로, (keyword, question) 파일따로

class AcallProcessor:
    def __init__(self, args):
        self.args = args
    
    #@classmethod #클래스 메소드는 인스턴스가 공유하는 클래스 데이터를 사용할 수 있음, 왜 써야하는가?
    def _read_file(self, file_path): #파일을 어떤 식으로 저장해야할까?
        with open(file_path) as f:
            data = json.load(f)['1']
            return data
#         with open(input_file, "r", encoding="utf-8") as f:
#             lines = [line.strip() for line in f]
#             return lines
    
    def get_labels(self, args):
        label_list = [str(p) for p in range(1, args.page_num+1)]
        return label_list
    
    #List[str] → List[InputExample]
    def _create_examples(self, data): #미구현
        examples = []
        for item in data:
            description = item['description']
            keywords = item['keywords']
            questions = item['questions']
            label = int(item['page_num'])
        
            assert description != "" and keywords != "" and questions != "" and label != ""
            assert len(keywords) == len(questions)
            # if not len(keywords) == len(questions):
            #     continue
            
            pair_num = len(keywords) #한 개의 description에 몇 개의 (keyword, question) 쌍이 들어있는지
            pair_examples = []
            for p in range(pair_num):
                keyword = keywords[p]
                question = questions[p]
                example = InputExample(description, keyword, question, label)
                pair_examples.append(example)
            examples.extend(pair_examples)
        return examples
    
    #File → List[InputExample]
    def get_examples(self, mode):
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file
        
        file_path = os.path.join(self.args.data_dir, file_to_read)
        data = self._read_file(file_path)
        examples = self._create_examples(data)
        
        return examples


def convert_examples_to_features(args, examples, tokenizer):
    processor = AcallProcessor(args)
    max_len = args.max_seq_len
    
    #tokenize return : tokens_with_unk, tokens_without_unk 
    descriptions = [tokenizer.tokenize(example.description)[0] for example in examples]
    keywords = [tokenizer.tokenize(example.keyword)[0] for example in examples]
    questions = [tokenizer.tokenize(example.question)[0] for example in examples]
    
    description_ids = [tokenizer.convert_tokens_to_ids(d) for d in descriptions]
    keyword_ids = [tokenizer.convert_tokens_to_ids(k) for k in keywords] 
    question_ids = [tokenizer.convert_tokens_to_ids(q) for q in questions] 
    
    label_list = processor.get_labels(args)
    label_map = {label: i for i, label in enumerate(label_list)} #"page 1" : 1
    labels = [label_map[str(example.label)] for example in examples]
    
    def build_input_with_special_tokens(description_ids, keyword_ids, question_ids):
        cls_token_id = [tokenizer.cls_token_id]
        sep_token_id = [tokenizer.sep_token_id]
        
        inputs_ids = []
        for d, k, q in zip(description_ids, keyword_ids, question_ids):
            input_ids = cls_token_id + d + sep_token_id + k + sep_token_id + q + sep_token_id
            inputs_ids.append(input_ids)
        return inputs_ids
    
    #List[List[int]]
    inputs_ids = build_input_with_special_tokens(description_ids, keyword_ids, question_ids)
    inputs_ids = pad_sequences(inputs_ids, maxlen=args.max_seq_len, dtype='long', truncating="post", padding="post", value=tokenizer.pad_token_id) 

    
    attention_masks = []
    for input_ids in inputs_ids:
        attention_mask = []
        for i in input_ids:
            if i== tokenizer.pad_token_id:
                attention_mask.append(0.0)
            else:
                attention_mask.append(1.0)
        attention_masks.append(attention_mask)
    
    #create feature
    features = []
    for i in range(len(examples)):
        inputs = {"input_ids" : inputs_ids[i],
                  "attention_mask" : attention_masks[i],
                  "label" : labels[i]}
    
        feature = InputFeatures(**inputs)
        features.append(feature)
    
    show_num = 1
    for i, example in enumerate(examples[:show_num]):
        logger.info("[Example 예시]")
        logger.info(f"Description : {example.description}")
        logger.info(f"Keyword : {example.keyword}")
        logger.info(f"Question : {example.question}")
        logger.info(f"input_ids : {features[i].input_ids}")
        logger.info(f"attention_mask : {features[i].attention_mask}")
        logger.info(f"label : {features[i].label}")
    return features

def cache_and_load_tensors(args, tokenizer, mode):
    processor = AcallProcessor(args)
    
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(args.model_path.split('/')[-1], str(args.max_seq_len), args.train_batch_size, mode))
    if os.path.exists(cached_features_file):
        logger.info("load cache file...")
        features = torch.load(cached_features_file)
    else:
        logger.info("can't find cache file and create feature file")
        assert mode == "train" or mode == "dev" or mode == "test"
        examples = processor.get_examples(mode)
        features = convert_examples_to_features(args, examples ,tokenizer)
        
        logger.info(f"save features file at {cached_features_file}")
        torch.save(features, cached_features_file)
    
    tensor_inputs_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    tensor_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.float)
    tensor_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    assert len(tensor_inputs_ids) == len(tensor_attention_masks) == len(tensor_labels)
    dataset = TensorDataset(tensor_inputs_ids, tensor_attention_masks, tensor_labels)
    return dataset