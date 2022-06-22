"""Train meta-learning BERT."""
# The script is modified from `https://github.com/mailong25/meta-learning-bert/blob/master/main.py`

import os
import argparse
import logging
import json
import pathlib
import sys
import time
import random
import numpy as np
import sklearn

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertModel, BertTokenizer

from collections import Counter
from task import MetaTask
from models.maml import Learner

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="")

    # Model
    parser.add_argument("--bert_model", type=str, default="bert-base-cased")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--emb_size", type=int, default=256,
                        help="Label and output embedding size")
    parser.add_argument("--output_dir", type=str, default="results/tmp",
                        help="Directory for saving checkpoint and log file.")

    # Training 
    parser.add_argument("--train_data", type=str, default="dataset.json")
    parser.add_argument("--test_data", type=str, default="dataset.json")
    parser.add_argument("--epochs", type=int, default=1)
    
    parser.add_argument("--outer_batch_size", type=int, default=5,
                        help="Batch size of training tasks")
    parser.add_argument("--inner_batch_size", type=int, default=16,
                        help="Batch size of support set")

    parser.add_argument("--inner_update_step", type=int, default=7)
    parser.add_argument("--inner_update_step_eval", type=int, default=7)
    parser.add_argument("--gpu_id", type=int, default=0)

    # Meta task
    parser.add_argument("--num_support", type=int, default=80,
                        help="Number of support set")
    parser.add_argument("--num_query", type=int, default=10,
                        help="Number of query set")
    
    parser.add_argument("--num_train_task", type=int, default=50,
                        help="Number of meta training tasks")
    parser.add_argument("--num_test_task", type=int, default=3,
                        help="Number of meta testing tasks")

    # Optimizer
    parser.add_argument("--outer_update_lr", type=float, default=1e-5)
    parser.add_argument("--inner_update_lr", type=float, default=5e-5)

    # Curriculum
    parser.add_argument("--curriculum", type=bool, default=False, 
                        help="Use curriculum learning")

    return parser.parse_args()

def get_output_dir(output_dir, file):
    """Joint path for output directory."""
    return pathlib.Path(output_dir,file)


def build_dirs(output_dir, logger):
    """Build hierarchical directories."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info(f"Create folder for output directory: {output_dir}")

def task_batch_generator(taskset, is_shuffle, batch_size):
    """Yield a batch of tasks from train or test set."""
    idxs = list(range(0, len(taskset)))
     
    if is_shuffle:
        random.shuffle(idxs)

    for i in range(0, len(idxs), batch_size):
        yield [taskset[idxs[j]] for j in range(i, min(i+batch_size, len(taskset)))]  


def set_random_seed(seed):
    """Set new random seed."""
    torch.backends.cudnn.determinstic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]


def main():    
    # Training arguments
    args = get_args()

    # output dir
    output_dir = args.output_dir

    # Logger
    logger = logging.getLogger(__name__)
    build_dirs(output_dir, logger)
    build_dirs(pathlib.Path(output_dir, "ckpt"), logger)
    
    log_file = get_output_dir(output_dir, 'example.log')
    logging.basicConfig(filename=log_file,
                        filemode="w",
                        format="%(asctime)s, %(levelname)s %(message)s",
                        datefmt="%H:%M:%S",
                        level=logging.INFO)

    # Add console to logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.info(args)
    
    # Saving arguments
    write_path = get_output_dir(output_dir, 'hyparams.txt')
    with open(write_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        logger.info(f"Saving hyperparameters to: {write_path}")

    ########## Load dataset ##########
    logger.info("Loading Datasets")
    train_data = json.load(open(args.train_data))
    test_data = json.load(open(args.test_data))

    # Load한 데이터에서 num_train_task, num_test_task만큼만 사용
    # 일단 여기에는 randomness 안 넣음
    train_examples = train_data[:args.num_train_task]
    test_examples = test_data[:args.num_test_task]
    if args.curriculum:
        train_examples.reverse() # 쉬운 task가 먼저 오도록

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False) # 일단 BERT-cased로

    # Meta-Learner
    learner = Learner(args)

    ### Sample testing tasks ###
    test_tasks = MetaTask(test_examples,
                          num_task=args.num_test_task,
                          num_labels=args.num_labels,
                          k_support=args.num_support,
                          k_query=args.num_query,
                          max_seq_length=args.max_seq_length,
                          tokenizer=tokenizer)

    global_step = 1
    global_train_outer_loss = list() # 최종 length: num_task/outer_batch_size
    global_train_acc = list() # 최종 length: num_task/outer_batch_size
    global_test_acc = list() # 최종 length: num_task/(outer_batch_size*20)
    
    # Train perplexity:  epoch * num_task * ceil(k_support/batch_size) * inner_update_step
    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch+1} ---")
               
        # Build training task set (num_task)
        train_tasks = MetaTask(train_examples,
                               num_task=args.num_train_task, # 50
                               num_labels=args.num_labels, # 2 (일단은 2만)
                               k_support=args.num_support, # 80
                               k_query=args.num_query, # 10
                               max_seq_length=args.max_seq_length, # 128
                               tokenizer=tokenizer)

        logger.info(f"Processing {len(train_tasks)} training tasks")

        ### Sample task batch from training tasks ###
        # Sample task batch from total tasks in size of `min(num_task, batch_size)`
        # Each task contains `k_support` + `k_query` examples
        task_batch = create_batch_of_tasks(train_tasks, 
                                           is_shuffle=not args.curriculum,
                                           batch_size=args.outer_batch_size) # default는 4지만 argument는 5

        # meta_batch has shape (batch_size, k_support*k_query) -> k_support+k_query인데 오타인듯
        for step, meta_batch in enumerate(task_batch): # 10번 iterate (num_task/outer_batch_size = 50/5 = 10)
            acc, q_loss = learner(meta_batch, step, training=True)
            logger.info(f"Training batch: {step+1} ({(step+1)*args.outer_batch_size} tasks done) \t training accuracy: {round(acc, 4)} \t average outer loss: {round(q_loss, 4)}\n")
            global_train_acc.append(round(acc, 4))
            global_train_outer_loss.append(round(q_loss, 4))

            if global_step % 20 == 0: # task 1000개로 할 때는 % 20으로
                # Evaluate Test every 1 batch
                logger.info("--- Evaluate test tasks ---")
                test_accs = list()
                # fixed seed for test task
                set_random_seed(1)
                test_db = task_batch_generator(test_tasks,
                                               is_shuffle=True,
                                               batch_size=1)
            
                for idx, test_batch in enumerate(test_db):
                    acc, _ = learner(test_batch, step, training=False)
                    test_accs.append(acc)
                    logger.info(f"Testing Task: {idx+1} \t accuracy: {round(acc, 4)}")
            
                logger.info(f"Epoch: {epoch+1}\tTesting batch: {step+1}\tTest accuracy: {round(np.mean(test_accs), 4)}\n")
                global_test_acc.append(round(np.mean(test_accs), 4))

                # Report results
                logger.info("--- Report ---")
                logger.info("--- Outer Loss ---")
                logger.info(global_train_outer_loss)
                logger.info("--- Training Accuracy ---")
                logger.info(global_train_acc)
                logger.info("--- Test Accuracy ---")
                logger.info(global_test_acc)
            
                # Save model
                pt_file = get_output_dir(args.output_dir, f"ckpt/pytorch_model_epoch-{epoch+1}_task-{(step+1)*args.outer_batch_size}.bin")
                # torch.save(learner, pt_file)
                torch.save(learner.model.state_dict(), pt_file) # save only BERT parameters
                logger.info(f"Saving checkpoint to {pt_file}\n")

                # Reset the random seed
                set_random_seed(int(time.time() % 10))
            
            global_step += 1
            

if __name__ == "__main__":
    main()


