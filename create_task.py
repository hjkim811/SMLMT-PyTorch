import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
from random import sample, seed, shuffle
import itertools
from tqdm.notebook import tqdm
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

def sent_containing_specific_word(vocab, corpus, sampling_num):
    '''
    Returns list of 'sampling_num' tokenized sentences containing 'vocab'
    vocab: one vocabulary word (string)
    corpus: list of sentences
    sampling_num: number of sentences to sample
    '''
    shuffle(corpus)
    containing = []
    for sent in corpus:
        if vocab in word_tokenize(sent):
            containing.append(word_tokenize(sent))
            if len(containing) == sampling_num: break

    assert len(containing) == sampling_num, f'Number of sentences containing "{vocab}" is less than {sampling_num}'

    return containing

def mask_sentence(vocab, corpus, sampling_num):
    '''
    Returns list of sentences where 'vocab' is replaced by [MASK]
    vocab: one vocabulary word (string)
    corpus: list of sentences
    sampling_num: number of sentences to sample
    '''    
    sampled = sent_containing_specific_word(vocab, corpus, sampling_num)
    return [TreebankWordDetokenizer().detokenize(['[MASK]' if word==vocab else word for word in sent]) for sent in sampled]

def create_task(vocab_list, corpus, num_support, num_query, mode='random'):
    '''
    Creates one task
    len(task) = len(vocab_list) * (num_support + num_query)
    vocab_list: list of vocabulary words 
                len(vocab_list) becomes N (number of classes)
    corpus: list of sentences
    '''   
    task = dict()
    task['support'] = list()
    task['query'] = list()

    if mode == 'random':
        shuffle(vocab_list) # for random label assignment
        for idx, vocab in enumerate(vocab_list):
            for i, masked_sent in enumerate(mask_sentence(vocab, corpus, num_support + num_query)):
                sent = dict()
                sent['text'] = masked_sent
                sent['label'] = idx
                sent['word'] = vocab
                if i < num_support:
                    task['support'].append(sent)
                else:
                    task['query'].append(sent)

    # Also returns similarity in curriculum mode
    elif mode == 'curriculum':
        shuffle(vocab_list[0]) # for random label assignment
        task['similarity'] = str(round(vocab_list[1], 4)) # float32는 나중에 json으로 저장이 안 됨
        for idx, vocab in enumerate(vocab_list[0]):
            for i, masked_sent in enumerate(mask_sentence(vocab, corpus, num_support + num_query)):
                sent = dict()
                sent['text'] = masked_sent
                sent['label'] = idx
                sent['word'] = vocab
                if i < num_support:
                    task['support'].append(sent)
                else:
                    task['query'].append(sent)

    else:
        raise ValueError('No such mode available')    
        
    return task

def vocab_sampler(vocabs, n, num_task, mode='random'):
    '''
    Returns combination of vocabulary words
    vocabs: set of vocabulary words to sample from
    n: number of vocabulary words in one pair (for n-way task)
    num_task: number of pairs to create
    '''
    combs = []
    
    if mode == 'random':
        while (len(combs) < num_task):
            comb = list(set(sample(vocabs, n)))
            if comb not in combs:
                combs.append(comb)
        
    elif mode == 'curriculum':
        from transformers import BertModel, BertTokenizer
        model_name = 'bert-base-cased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        emb_layer = list(model.parameters())[0].detach().numpy()

        def get_emb(word):
            embs = [emb_layer[input_id] for input_id in tokenizer.encode(word, add_special_tokens=False)]
            return np.mean(np.array(embs), axis=0)

        def cos_sim_word(a, b):
            a = get_emb(a)
            b = get_emb(b)
            return dot(a, b)/(norm(a)*norm(b))

        def sim_score(vocab_list):
            # Similarity when n >= 3: the max value among all similarities of nC2 combinations
            return max([cos_sim_word(comb2[0], comb2[1]) for comb2 in itertools.combinations(vocab_list, 2)])        

        while (len(combs) < num_task):
            comb = list(set(sample(vocabs, n)))
            if comb not in combs:
                combs.append(comb) 

        combs = [(comb, sim_score(comb)) for comb in combs]
        combs = sorted(combs,key=lambda l:l[1], reverse=True)

    else:
        raise ValueError('No such mode available')
                    
    assert len(combs) == num_task, 'num_task not matched'
    
    return combs