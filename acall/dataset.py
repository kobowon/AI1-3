from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

class AcallDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 dataset: List[Dict[str,str]],
                 page_num: int,
                 max_seq_len: int,
                 batch_size: int,
                 shuffle: bool = False,
                 sampler = None,
                 **kwargs):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_list = [str(i) for i in range(1,page_num+1)]
        self.label2id = {label:i for i, label in enumerate(self.label_list)} #for collate
        self.collate_fn = CollateAcallSimple(tokenizer, self.label2id, max_seq_len) 
        self.sampler = None
        self.loader = DataLoader(dataset=self,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 sampler=sampler,
                                 collate_fn=self.collate_fn,
                                 **kwargs)
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
            return self.dataset[i]
        
        
class CollateAcall:
    def __init__(self, tokenizer, label2id, max_seq_len):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_len = max_seq_len
    
    def __call__(self, batches):
        b_input_ids = []
        b_input_attention_mask = []
        b_input_label_ids = []
        
        if 'label' in batches[0]:#train, eval
            for i, b in enumerate(batches): #per data
                description = b['description']
                keyword = b['keyword']
                question = b['question']
                label = b['label']

                description_ids = self.tokenizer.encode(description)
                keyword_ids = self.tokenizer.encode(keyword)
                question_ids = self.tokenizer.encode(question)
                label_id = self.label2id[str(label)]

                #truncate description(description이 길이가 가장 길 것으로 예상) 
                SPECIAL_TOKENS_NUM = 4 #<s>description</s>keyword</s>question</s>
                limit = self.max_seq_len - SPECIAL_TOKENS_NUM
                input_len = len(description_ids) + len(keyword_ids) + len(question_ids)
                if input_len > limit:
                    gap = input_len - limit
                    possible_len = len(description_ids) - gap
                    description_ids = description_ids[:possible_len]
                    
                #make input
                input_ids = [self.tokenizer.cls_token_id]
                input_ids = input_ids + description_ids + [self.tokenizer.sep_token_id]
                input_ids = input_ids + keyword_ids + [self.tokenizer.sep_token_id]
                input_ids = input_ids + question_ids + [self.tokenizer.sep_token_id]
                
                #make attention mask
                input_attention_mask = [1] * len(input_ids)
                assert len(input_ids) == len(input_attention_mask)
                
                b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
                b_input_label_ids.append(torch.tensor(label_id, dtype=torch.long))
                
            t_input_ids = torch.nn.utils.rnn.pad_sequence(b_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            t_input_attention_mask = torch.nn.utils.rnn.pad_sequence(b_input_attention_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            t_input_label_ids = torch.stack(b_input_label_ids)
            return t_input_ids, t_input_attention_mask, t_input_label_ids
                
        else: #inference
            for b in batches: #per data
                inference_text = b['inference_text']
                text_ids = self.tokenizer.encode(inference_text)
                
                SPECIAL_TOKENS_NUM = 2 #<s>text_ids</s>
                limit = self.max_seq_len - SPECIAL_TOKENS_NUM
                input_len = len(text_ids)
                if input_len > limit:
                    text_ids = text_ids[:limit]
                    
                input_ids = [self.tokenizer.cls_token_id] + text_ids + [self.tokenizer.sep_token_id]
                input_attention_mask = [1] * len(input_ids)
                assert len(input_ids) == len(input_attention_mask)
                
                b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
                
            t_input_ids = torch.nn.utils.rnn.pad_sequence(b_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            t_input_attention_mask = torch.nn.utils.rnn.pad_sequence(b_input_attention_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return t_input_ids, t_input_attention_mask

        
class CollateAcallSimple:
    def __init__(self, tokenizer, label2id, max_seq_len):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_len = max_seq_len
    
    def __call__(self, batches):
        b_input_ids = []
        b_input_attention_mask = []
        b_input_label_ids = []
        
        if 'label' in batches[0]:#train, eval
            for i, b in enumerate(batches): #per data
                description = b['description']
                keyword = b['keyword']
                question = b['question']
                label = b['label']

                description_ids = self.tokenizer.encode(description)
                keyword_ids = self.tokenizer.encode(keyword)
                question_ids = self.tokenizer.encode(question)
                label_id = self.label2id[str(label)]

                #truncate description(description이 길이가 가장 길 것으로 예상) 
                SPECIAL_TOKENS_NUM = 3 #<s>keyword</s>question</s>
                limit = self.max_seq_len - SPECIAL_TOKENS_NUM
                input_len = len(keyword_ids) + len(question_ids)
                if input_len > limit:
                    gap = input_len - limit
                    possible_len = len(question_ids) - gap
                    question_ids = question_ids[:possible_len]
                    
                #make input
                input_ids = [self.tokenizer.cls_token_id]
                input_ids = input_ids + keyword_ids + [self.tokenizer.sep_token_id]
                input_ids = input_ids + question_ids + [self.tokenizer.sep_token_id]
                
                #make attention mask
                input_attention_mask = [1] * len(input_ids)
                assert len(input_ids) == len(input_attention_mask)
                
                b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
                b_input_label_ids.append(torch.tensor(label_id, dtype=torch.long))
                
            t_input_ids = torch.nn.utils.rnn.pad_sequence(b_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            t_input_attention_mask = torch.nn.utils.rnn.pad_sequence(b_input_attention_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            t_input_label_ids = torch.stack(b_input_label_ids)
            return t_input_ids, t_input_attention_mask, t_input_label_ids
                
        else: #inference
            for b in batches: #per data
                inference_text = b['inference_text']
                text_ids = self.tokenizer.encode(inference_text)
                
                SPECIAL_TOKENS_NUM = 2 #<s>text_ids</s>
                limit = self.max_seq_len - SPECIAL_TOKENS_NUM
                input_len = len(text_ids)
                if input_len > limit:
                    text_ids = text_ids[:limit]
                    
                input_ids = [self.tokenizer.cls_token_id] + text_ids + [self.tokenizer.sep_token_id]
                input_attention_mask = [1] * len(input_ids)
                assert len(input_ids) == len(input_attention_mask)
                
                b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                b_input_attention_mask.append(torch.tensor(input_attention_mask, dtype=torch.long))
                
            t_input_ids = torch.nn.utils.rnn.pad_sequence(b_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            t_input_attention_mask = torch.nn.utils.rnn.pad_sequence(b_input_attention_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return t_input_ids, t_input_attention_mask
            
                 