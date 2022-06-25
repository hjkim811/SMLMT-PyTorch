# SMLMT-PyTorch
This is a repository for creating tasks and running meta-training using Subset Masked Language Modeling Tasks (SMLMT). <br />
SMLMT proposes a self-supervised approach to generate a large, rich, meta-learning task distribution from unlabeled text. <br />
Then, the model can be meta-trained using a optimization-based framework based on MAML.

## Environment
```
conda env create -f environment.yml
conda activate meta-bert
```

## Task Creation
Refer create_task.ipynb. <br />
Created task files using review text in laptop and restaurant domain can be found [here](https://github.com/hjkim811/SMLMT-PyTorch/tree/main/task). <br />
Each file contains 1000 tasks with 80 support samples and 10 query samples, respectively.

## Meta-Training
### Laptop domain, n = 2
```
python run_meta_training.py \
 --train_data task/task_laptop_random_1000_500_2_80_10.json \
 --test_data task/task_laptop_random_10_500_2_80_10_test.json \
 --output_dir results/result_laptop_random_1000_500_2_80_10 \
 --num_labels 2 \
 --num_train_task 1000
```
### Laptop domain, n = 3
```
python run_meta_training.py \
 --train_data task/task_laptop_random_1000_500_3_80_10.json \
 --test_data task/task_laptop_random_10_500_3_80_10_test.json \
 --output_dir results/result_laptop_random_1000_500_3_80_10 \
 --num_labels 3 \
 --num_train_task 1000
```
### Restaurant domain, n = 2
```
python run_meta_training.py \
 --train_data task/task_rest_random_1000_500_2_80_10.json \
 --test_data task/task_rest_random_10_500_2_80_10_test.json \
 --output_dir results/result_rest_random_1000_500_2_80_10 \
 --num_labels 2 \
 --num_train_task 1000
```
### Restaurant domain, n = 3
```
python run_meta_training.py \
 --train_data task/task_rest_random_1000_500_3_80_10.json \
 --test_data task/task_rest_random_10_500_3_80_10_test.json \
 --output_dir results/result_rest_random_1000_500_3_80_10 \
 --num_labels 3 \
 --num_train_task 1000
```
 
## References
- Code is based on this repository: https://github.com/pjlintw/Meta-BERT#installation
- Paper: Bansal, Trapit, et al. (2020). [Self-Supervised Meta-Learning for Few-Shot Natural Language Classification Tasks](https://arxiv.org/pdf/2009.08445.pdf)
