"""Perform First-Order MAML."""
import logging
from torch import nn
import torch.nn as nn # import overloading?
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset, RandomSampler, SubsetRandomSampler
from torch.optim import Adam, SGD
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
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels = self.num_labels)
        self.model.train() # sets to train mode
        self.deep_set_encoder = nn.Sequential(
                                nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2), # [768, 384]
                                nn.Tanh(),
                                nn.Linear(self.model.config.hidden_size//2, self.emb_size + 1) # [384, 257]
                                ) # ψ / initialize 다르게?
        self.deep_set_encoder.train() # sets to train mode

        outer_params = list(self.model.parameters()) + list(self.deep_set_encoder.parameters()) # length: 201 + 4 =205
        self.outer_optimizer = Adam(outer_params, lr=self.outer_update_lr) # SGD로?
        

    def forward(self, batch_tasks, step, training=True):
        """Perform first-order approximation MAML.

        batch = [(support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset),
                 (support TensorDataset, query TensorDataset)]
        
        # support = TensorDataset(all_input_ids, all_attention_mask, all_segment_ids, all_label_ids)
        """
        task_accs = []
        task_qloss = []
        sum_gradients_bert = []
        sum_gradients_dse = [] # deep set encoder
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step if training else self.inner_update_step_eval

        for task_idx, task in enumerate(batch_tasks): # 5번 iterate (= outer_batch_size)
            # 이 아래: 하나의 task에 대한 것 (support data n*80개 존재)

            support = task[0] # 각 label에 대한 데이터 각각 80개씩 있음 / = pseudocode의 D^tr
            query   = task[1] # 각 label에 대한 데이터 각각 10개씩 있음       

            support_dataloader = DataLoader(support, sampler=RandomSampler(support), batch_size=self.inner_batch_size)

            class_partition = []
            for label in range(self.num_labels):
                class_partition.append([d for d in support if d[3]==label])
            # len(class_partition) = self.num_labels
            
            # softmax parameter generation에 first batch만 사용함 -> 이걸 위해 16개만 sampling
            # first batch 중복으로 사용되는 문제 있나? Indices를 바꿔주기?
            class_dataloader = []
            for label in range(self.num_labels):
                class_dataloader.append(DataLoader(class_partition[label], sampler=SubsetRandomSampler(list(range(self.inner_batch_size))), batch_size=self.inner_batch_size))
            # len(class_dataloader) = self.num_labels
            
            # initialize task-specific parameters
            task_weights = {}
            task_weights['bert'] = deepcopy(self.model) 
            task_weights['mlp'] = nn.Sequential(
                                    nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2), # [768, 384]
                                    nn.Tanh(),
                                    nn.Linear(self.model.config.hidden_size//2, self.emb_size) # [384, 256]
                                    ) # ф / initialize 다르게?
            # logger.info('task-specific parameters initialized')

            # unify device for parameter generation
            for key in task_weights.keys():
                task_weights[key].to(self.device)
            self.deep_set_encoder.to(self.device)

            class_W = []
            class_b = []
            for dataloader in class_dataloader:
                batch = iter(dataloader).next()
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, attention_mask, segment_ids, _ = batch # label은 제외
                with torch.no_grad():
                    outputs = task_weights['bert'](input_ids, attention_mask, segment_ids, output_hidden_states=True) 
                deep_input = outputs[1][-1][:,0,:] # 각 데이터 하나의 마지막 hidden layer의 CLS representation ([768])  
                class_W.append(torch.mean(self.deep_set_encoder(deep_input), 0)[:-1]) # [256] tensor를 append
                class_b.append(torch.mean(self.deep_set_encoder(deep_input), 0)[-1:]) # [1] tensor를 append
                # len(class_W) = len(class_b) = self.num_labels            

            # add generated parameters to task-specific parameter
            task_weights['W'] = torch.stack(class_W) # [n, 256]
            task_weights['b'] = torch.cat(class_b) # [n]

            # unify device for inner loop (support set)
            task_weights['W'] = task_weights['W'].to(self.device) # tensor는 assign 해줘야
            task_weights['b'] = task_weights['b'].to(self.device) # tensor는 assign 해줘야

            # bert와 mlp는 optimizer로 update
            # W와 b는 수동으로 update (non-leaf tensor이기 때문에 optimizer를 사용할 수 없음)
            inner_params = list(task_weights['bert'].parameters()) + list(task_weights['mlp'].parameters()) # length: 201 + 4 = 205
            inner_optimizer = Adam(inner_params, lr=self.inner_update_lr) # SGD로?

            # train mode 확인 방법: .training
            # W와 b는 nn.Module이 아니라 tensor이기 때문에 아래 작업 필요 x
            task_weights['bert'].train() # train mode: False -> True
            task_weights['mlp'].train() # train mode: True -> True
            
            if training:
                # logger.info(f"--- Training Task {step*self.outer_batch_size+task_idx+1} ---")

                # freeze warp layers
                num_params = 0
                for name, param in task_weights['bert'].named_parameters():
                    if any(i in name for i in ['intermediate', 'output']) and 'attention' not in name:
                        num_params += torch.numel(param)
                        param.requires_grad = False # freeze layer
                percent = round(num_params/task_weights['bert'].num_parameters()*100, 2)
                # logger.info(f'warp layers freezed ({percent}% of BERT parameters)')                

            else:
                logger.info(f"--- Testing Task ---")

            # inner loop
            for i in range(0,num_inner_update_step): 
                all_loss = []
                for inner_step, batch in enumerate(support_dataloader): # 10번 iterate (num_lables*num_support/inner_batch_size = 2*80/16 = 10)

                    task_weights['W'].retain_grad()
                    task_weights['b'].retain_grad()

                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, attention_mask, segment_ids, label_id = batch

                    mlp_input = task_weights['bert'](input_ids, attention_mask, segment_ids, output_hidden_states=True)[1][-1][:,0,:] # [16, 768]
                    mlp_output = task_weights['mlp'](mlp_input) # [16, 256]
                    pred = torch.matmul(mlp_output, torch.transpose(task_weights['W'],0,1)) + task_weights['b'] # [16, 2] / 논문의 p(y|X)
                    loss = self.loss(pred, label_id)

                    loss.backward(retain_graph=True)                 

                    inner_optimizer.step() # task_weights['bert'], task_weights['mlp'] update
                    task_weights['W'] -= self.inner_update_lr*task_weights['W'].grad # 수동으로 update
                    task_weights['b'] -= self.inner_update_lr*task_weights['b'].grad # 수동으로 update
                    inner_optimizer.zero_grad()
                    # W와 b의 grad는 자동으로 None으로 바뀜

                    # 현재: bert와 mlp는 Adam으로 udpate, W와 b는 SGD로 update -> 나중에 통일하기
                    # Adam과 SGD에서 잘 작동하는 lr가 다를 수 있음
                    
                    all_loss.append(loss.item())

                # if i % 3 == 0: # 원래: 4
                #     logger.info(f"Inner Loss: {round(np.mean(all_loss),4)}")        

            # outer update 할 때는 BERT의 모든 layer unfreeze
            if training:
                for name, param in task_weights['bert'].named_parameters():
                    param.requires_grad = True
                # logger.info('unfreeze warp layers for outer update')   

            query_dataloader = DataLoader(query, sampler=None, batch_size=len(query))
            query_batch = iter(query_dataloader).next()
            query_batch = tuple(t.to(self.device) for t in query_batch)
            q_input_ids, q_attention_mask, q_segment_ids, q_label_id = query_batch
            q_mlp_input = task_weights['bert'](q_input_ids, q_attention_mask, q_segment_ids, output_hidden_states=True)[1][-1][:,0,:] # [20, 768]
            q_mlp_output = task_weights['mlp'](q_mlp_input) # [20, 256]
            q_pred = torch.matmul(q_mlp_output, torch.transpose(task_weights['W'],0,1)) + task_weights['b'] # [20, 2]

            # In FOMAML, learner adapts on new task by updating
            # the gradient which is derived from fast model 
            # on queries set.
            if training:
                q_loss = self.loss(q_pred, q_label_id)
                q_loss.backward()
                # logger.info(f"Outer Loss: {round(float(q_loss.detach().cpu().numpy()), 4)}")
                task_qloss.append(float(q_loss.detach().cpu().numpy()))

                # unify device for outer loop
                task_weights['bert'].to(torch.device('cpu'))
                self.deep_set_encoder.to(torch.device('cpu'))

                # meta paramter: bert
                for i, (name, params) in enumerate(task_weights['bert'].named_parameters()):
                    if task_idx == 0:
                        if params.grad == None:
                            # logger.info(name, 'None')
                            pass
                        sum_gradients_bert.append(deepcopy(params.grad))
                    else:
                        if params.grad == None:
                            # logger.info(name, 'None')
                            pass
                        else:
                            sum_gradients_bert[i] += deepcopy(params.grad)

                # meta paramter: deep set encoder   
                for i, params in enumerate(self.deep_set_encoder.parameters()):
                    if task_idx == 0:
                        sum_gradients_dse.append(deepcopy(params.grad))
                    else:
                        sum_gradients_dse[i] += deepcopy(params.grad)

            q_logits = F.softmax(q_pred, dim=1)
            pre_label_id = torch.argmax(q_logits, dim=1)
            pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
            q_label_id = q_label_id.detach().cpu().numpy().tolist()

            acc = accuracy_score(pre_label_id,q_label_id)
            task_accs.append(acc)
            
            del task_weights, inner_optimizer
            torch.cuda.empty_cache()
        
        if training:
            # Average gradient across tasks
            for i in range(0,len(sum_gradients_bert)):
                if sum_gradients_bert[i] == None:
                    # logger.info('unable to divide because it is None')
                    pass
                else:
                    sum_gradients_bert[i] = sum_gradients_bert[i] / float(num_task)

            for i in range(0,len(sum_gradients_dse)):
                sum_gradients_dse[i] = sum_gradients_dse[i] / float(num_task)

            # Assign gradient for original model, then using optimizer to update its weights
            # gradient 계산은 task-specific parameter에서 하고 meta parameter 업데이트(계산)할 때는 그 값을 복사해서 사용
            for i, params in enumerate(self.model.parameters()):
                params.grad = sum_gradients_bert[i]

            for i, params in enumerate(self.deep_set_encoder.parameters()):
                params.grad = sum_gradients_dse[i]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            
            del sum_gradients_bert, sum_gradients_dse
            gc.collect()
        
        if training:
            return np.mean(task_accs), np.mean(task_qloss)
        else:
            return np.mean(task_accs), None
