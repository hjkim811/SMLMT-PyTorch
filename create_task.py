import json
from random import sample, seed, shuffle
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

# def create_task(vocab_list, corpus, num_support, num_query):
#     '''
#     Creates one task
#     len(task) = len(vocab_list) * (num_support + num_query)
#     vocab_list: list of vocabulary words 
#                 len(vocab_list) becomes N (number of classes)
#     corpus: list of sentences
#     '''   
#     shuffle(vocab_list) # for random label assignment
#     task = list()    
#     for idx, vocab in enumerate(vocab_list):
#         for masked_sent in mask_sentence(vocab, corpus, num_support + num_query):
#             sent = dict()
#             sent['text'] = masked_sent
#             sent['label'] = idx
#             sent['word'] = vocab
#             task.append(sent)
        
#     return task

def create_task(vocab_list, corpus, num_support, num_query):
    '''
    Creates one task
    len(task) = len(vocab_list) * (num_support + num_query)
    vocab_list: list of vocabulary words 
                len(vocab_list) becomes N (number of classes)
    corpus: list of sentences
    '''   
    shuffle(vocab_list) # for random label assignment
    task = dict()
    task['support'] = list()
    task['query'] = list()
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
        
    return task

def vocab_sampler(vocabs, n, num_task, mode):
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
    else:
        raise ValueError('No such mode available')
                    
    assert len(combs) == num_task, 'num_task not matched'
    
    return combs