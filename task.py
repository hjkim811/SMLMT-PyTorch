"""Perform metatask"""
# The script is modified from https://github.com/mailong25/meta-learning-bert/blob/master/task.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import time
import json, pickle
from torch.utils.data import TensorDataset

# LABEL_MAP  = {'positive':0, 'negative':1, 0:'positive', 1:'negative'}

class MetaTask(Dataset):
    def __init__(self, examples, num_task, num_labels, k_support, k_query, max_seq_length, tokenizer):
        """
        Args:
          samples: list of samples
          num_task: number of training tasks.
          k_support: number of support sample per task
          k_query: number of query sample per task
          tokenizer: Tokeneizer from `transformers`
        """
        self.examples = examples
        self.num_task = num_task
        self.num_labels = num_labels
        self.k_support = k_support
        self.k_query = k_query
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.create_batch()

    def create_batch(self):
        self.supports = [task['support'] for task in self.examples]
        self.queries = [task['query'] for task in self.examples]

    def create_feature_set(self, examples):
        all_input_ids      = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_attention_mask = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_segment_ids    = torch.empty(len(examples), self.max_seq_length, dtype = torch.long)
        all_label_ids      = torch.empty(len(examples), dtype = torch.long)

        for id_,example in enumerate(examples):
            input_ids = self.tokenizer.encode(example['text'])
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length] # max_seq_length를 넘어가는 경우에는 max_seq_length에서 cut
            attention_mask = [1] * len(input_ids)
            segment_ids    = [0] * len(input_ids)         

            while len(input_ids) < self.max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                segment_ids.append(0)

            label_id = example['label']
            all_input_ids[id_] = torch.Tensor(input_ids).to(torch.long)
            all_attention_mask[id_] = torch.Tensor(attention_mask).to(torch.long)
            all_segment_ids[id_] = torch.Tensor(segment_ids).to(torch.long)
            all_label_ids[id_] = torch.Tensor([label_id]).to(torch.long)

        tensor_set = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)  
        return tensor_set
    
    def __getitem__(self, index):
        support_set = self.create_feature_set(self.supports[index])
        query_set   = self.create_feature_set(self.queries[index])
        return support_set, query_set

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.num_task
