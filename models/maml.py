"""Perform First-Order MAML."""
# The script is modified from the `https://github.com/mailong25/meta-learning-bert/blob/master/maml.py`

import logging
from torch import nn
import torch.nn as nn # import 충돌 생기려나?
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification
from copy import deepcopy
import gc
import torch
from sklearn.metrics import accuracy_score
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console to logger
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console) 


class Learner(nn.Module):
    """MAML Learner"""
    def __init__(self, args):
        super(Learner, self).__init__()
        
        self.num_labels = args.num_labels
        self.emb_size = args.emb_size
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr  = args.outer_update_lr
        self.inner_update_lr  = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.gpu_id = args.gpu_id
        self.bert_model = args.bert_model
        self.loss = nn.CrossEntropyLoss()
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels = self.num_labels)
        
        # ψ
        self.deep_set_encoder = nn.Sequential(
                                nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2), # [768, 384]
                                nn.Tanh(),
                                nn.Linear(self.model.config.hidden_size//2, self.emb_size + 1) # [384, 257]
                                ) # 마지막에 tanh 또 넣어야 하나?

        self.outer_optimizer = Adam(self.model.parameters(), lr=self.outer_update_lr)
        self.model.train() # sets to train mode

    def forward(self, batch_tasks, training=True):
        """Perform first-order approximation MAML.

        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]
        
        # support = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """
        task_accs = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step if training else self.inner_update_step_eval

        for task_idx, task in enumerate(batch_tasks): # 5번 iterate (= outer_batch_size)
            # 이 아래: 하나의 task에 대한 것 (support data 2*80개 존재함)

            support = task[0] # 각 label에 대한 데이터 각각 80개씩 있음 / = pseudocode의 D^tr
            query   = task[1] # 각 label에 대한 데이터 각각 20개씩 있음
            
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)
            
            # C^n implementation: partition data according to class labels
            C0 = [d for d in support if d[3]==0] # 80개
            C1 = [d for d in support if d[3]==1] # 80개

            # softmax parameter generation에 first batch만 사용함 -> 이걸 위해 16개만 sampling
            # C0_dataloader = DataLoader(C0, sampler=RandomSampler(C0), batch_size=1) 
            # C1_dataloader = DataLoader(C1, sampler=RandomSampler(C1), batch_size=1)
            C0_dataloader = DataLoader(Subset(C0,list(range(self.inner_batch_size))), batch_size=self.inner_batch_size)
            C1_dataloader = DataLoader(Subset(C1,list(range(self.inner_batch_size))), batch_size=self.inner_batch_size) 
            # C0_softmax_param = torch.zeros(self.emb_size + 1).to(self.device) # [257]
            # C1_softmax_param = torch.zeros(self.emb_size + 1).to(self.device) # [257]

            # initialize task-specific parameters
            task_weights = {}
            task_weights['bert'] = deepcopy(self.model) # deepcopy 해도 되나?
            task_weights['mlp'] = nn.Sequential(
                                    nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2), # [768, 384]
                                    nn.Tanh(),
                                    nn.Linear(self.model.config.hidden_size//2, self.emb_size) # [384, 256]
                                    ) # ф / 마지막에 tanh 또 넣어야 하나? / initialize 어떻게?
            logger.info('task-specific parameters initialized')            

            for key in task_weights.keys():
                task_weights[key].to(self.device)
            self.deep_set_encoder.to(self.device)

            for batch in C0_dataloader: # 1번 iterate
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, segment_ids, _ = batch # label은 제외
                outputs = task_weights['bert'](input_ids, attention_mask, segment_ids, output_hidden_states=True) # 여기서 self.model or task_weights['bert'] 써야?
                deep_input = outputs[1][-1][:,0,:] # 각 데이터 하나의 마지막 hidden layer의 CLS representation ([768])
                C0_output = self.deep_set_encoder(deep_input)
            C0_output = torch.mean(C0_output, 0)   

            for batch in C1_dataloader: # 1번 iterate
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, segment_ids, _ = batch # label은 제외
                outputs = task_weights['bert'](input_ids, attention_mask, segment_ids, output_hidden_states=True) # 여기서 self.model or task_weights['bert'] 써야?
                deep_input = outputs[1][-1][:,0,:] # 각 데이터 하나의 마지막 hidden layer의 CLS representation ([768])
                C1_output = self.deep_set_encoder(deep_input)
            C1_output = torch.mean(C1_output, 0) 

            generated_weight = torch.stack((C0_output[:-1], C1_output[:-1])) # [2, 256]
            generated_bias = torch.cat((C0_output[-1:], C1_output[-1:])) # [2]
            generated_linear = nn.Linear(256, 2)
            generated_linear.weight = nn.Parameter(generated_weight)
            generated_linear.bias = nn.Parameter(generated_bias)

            # add generated parameters to task-specific parameter
            task_weights['gen'] = generated_linear # [256, 2] / softmax는 따로 추가할 필요 x (nn.CrossEntropyLoss()에 이미 포함되어있음)
            task_weights['gen'].to(self.device)
            
            params = list(task_weights['bert'].parameters()) + list(task_weights['mlp'].parameters()) + list(task_weights['gen'].parameters()) # length: 201 + 4 + 2 = 207
            inner_optimizer = Adam(params, lr=self.inner_update_lr) # 나중에 SGD로 바꾸기

            # 이거 하기 전부터 이미 train mode인듯 (.training으로 체크)
            # training = True 아니어도 이거 하는거 맞음?
            for key in task_weights.keys():
                task_weights[key].train() 
            
            if training:
                logger.info(f"--- Training Task {task_idx+1} ---")

                # freeze warp layers
                # 여기에 넣는거 맞나?
                num_params = 0
                for name, param in task_weights['bert'].named_parameters():
                    if any(i in name for i in ['intermediate', 'output']) and 'attention' not in name:
                        num_params += torch.numel(param)
                        param.requires_grad = False # freeze layer
                percent = round(num_params/task_weights['bert'].num_parameters()*100, 2)
                print(f'warp layers freezed ({percent}% of BERT parameters)')                

            else:
                logger.info(f"--- Testing Task {task_idx+1} ---")

            for i in range(0,num_inner_update_step):
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader): # 10번 iterate (num_lables*num_support/inner_batch_size = 2*80/16 = 10)
                    
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, segment_ids, label_id = batch

                    mlp_input = task_weights['bert'](input_ids, attention_mask, segment_ids, output_hidden_states=True)[1][-1][:,0,:] # [16, 768]
                    mlp_output = task_weights['mlp'](mlp_input) # [16, 256]
                    pred = task_weights['gen'](mlp_output) # [16, 2] / 논문의 p(y|X)
                    loss = self.loss(pred, label_id)
                    # print('loss:', loss.item())

                    loss.backward()
                    inner_optimizer.step()
                    inner_optimizer.zero_grad()
                    
                    all_loss.append(loss.item())
                
                if i % 4 == 0:
                    logger.info(f"Inner Loss: {np.mean(all_loss)}")

            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).next()
            query_batch = tuple(t.to(self.device) for t in query_batch)
            q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
            q_outputs = fast_model(q_input_ids, q_attention_mask, q_segment_ids, labels = q_label_id)
            
            # In FOMAML, learner adapts on new task by updating
            # the gradient which is derived from fast model 
            # on queries set.
            if training:
                q_loss = q_outputs[0]
                q_loss.backward()
                fast_model.to(torch.device('cpu'))
                for i, params in enumerate(fast_model.parameters()):
                    if task_idx == 0:
                        sum_gradients.append(deepcopy(params.grad))
                    else:
                        sum_gradients[i] += deepcopy(params.grad)

            q_logits = F.softmax(q_outputs[1],dim=1)
            pre_label_id = torch.argmax(q_logits,dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()

            acc = accuracy_score(pre_label_id,q_label_id)
            task_accs.append(acc)
            
            del fast_model, inner_optimizer
            torch.cuda.empty_cache()
        
        if training:
            # Average gradient across tasks
            for i in range(0,len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_task)

            #Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.model.parameters()):
                params.grad = sum_gradients[i]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            
            del sum_gradients
            gc.collect()
        
        return np.mean(task_accs)
