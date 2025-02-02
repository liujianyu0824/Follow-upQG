from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

from collections import Counter
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import re
from collections import Counter
import numpy as np
import numpy as np


def calculate_ttr(text):
    text = re.sub(r'[^\w\s]', '', text).lower()

    words = text.split()
    
    num_types = len(set(words))
    
    num_tokens = len(words)
    
    # 计算TTR
    if num_tokens == 0:
        return 0.0
    ttr = num_types / num_tokens
    return ttr


def distinct_1(text):
    words = text.split()
    unique_unigrams = set(words)
    return len(unique_unigrams) / len(words)

def distinct_2(text):
    words = text.split()
    bigrams = zip(words, words[1:])
    unique_bigrams = set(bigrams)
    return len(unique_bigrams) / (len(words) - 1)


def shannon_entropy(text):
    # 计算字符的概率分布
    prob_dist = np.array(list(Counter(text).values())) / len(text)
    # 计算熵
    return -np.sum(prob_dist * np.log2(prob_dist))

def conditional_entropy(text1, text2):
    # 确保text1比text2长(如果不是,交换一下)
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    
    # 计算text1中每个字符的边缘概率
    prob_dist1 = np.array(list(Counter(text1).values())) / len(text1)
    
    # 将text2重复多次,直到长度与text1相同
    text2_repeated = (text2 * (len(text1) // len(text2) + 1))[:len(text1)]
    
    # 统计text1每个字符下text2每个字符出现的次数
    cond_count = {}
    for x, y in zip(text1, text2_repeated):
        if x not in cond_count:
            cond_count[x] = {}
        if y not in cond_count[x]:
            cond_count[x][y] = 0
        cond_count[x][y] += 1
    
    # 计算条件概率和条件熵
    cond_entropy = 0
    for x, count_x in cond_count.items():
        px = prob_dist1[list(Counter(text1).keys()).index(x)]
        entropy_x = 0
        for y, count_xy in count_x.items():
            py_x = count_xy / sum(count_x.values()) 
            entropy_x -= py_x * np.log2(py_x)
        cond_entropy += px * entropy_x
        
    return cond_entropy

def mutual_information(text1, text2):
    # 计算text2的熵
    h_text2 = shannon_entropy(text2)
    
    # 计算给定text1的情况下text2的条件熵
    cond_h_text2_text1 = conditional_entropy(text1, text2)
    
    # 计算互信息
    mi = h_text2 - cond_h_text2_text1
    
    return mi, h_text2

with open('/result/t5_followupQ.json', 'r') as f:
    original_data = json.load(f)


# 加载预训练模型和tokenizer
model_name = '/embedding/gpt2-base'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


def preprocess(text):
    return [word.lower() for word in text.split()]

Consistency = []
Distinct_1 = []
Distinct_2 = []
TTR = []

mutual_information = []

unigram = []
bigram = []


for data in tqdm(original_data, desc="Processing data"):
    if len(data['generated_follow-up']) == 0 :
        continue
    elif type(data['generated_follow-up'][0]) == dict:
        continue
    elif type(data['generated_follow-up']) == str:
        text2 = data['generated_follow-up']
    elif data['generated_follow-up'][0] != '':
        # text2 = ' '.join(data['generated_follow-up'])
        text2 = data['generated_follow-up'][0]
    else:
        continue

    text1 = data['question'] + data['answer']
    text3 = data['follow-up']

    texts = [text1, text2]

    processed_texts = [preprocess(text) for text in texts]


    # 创建字典和语料库
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # 训练LDA模型
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, random_state=42)
    
    # 计算主题一致性
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    # print(f"Coherence Score: {coherence_lda}")

    Consistency.append(coherence_lda)

    # 将文本编码为模型输入
    inputs = tokenizer(text2, return_tensors='pt')

    # 检查 token 索引是否在模型的词汇表范围内
    if torch.max(inputs['input_ids']) >= model.config.vocab_size:
        print("Error: Token index out of range")
        continue

    ttr = calculate_ttr(text2)
    TTR.append(ttr)

    d1 = distinct_1(text2)
    d2 = distinct_2(text2)
    Distinct_1.append(d1)
    Distinct_2.append(d2)

    # 计算条件熵和互信息
    words = text2.split()
    bigrams = zip(words, words[1:])
    uni_bigrams = list(bigrams)
    for word in words:
        unigram.append(word)
    for bi in uni_bigrams:
        bigram.append(bi)

    # 计算互信息
    mi, h_text2 = mutual_information(text1, text2)
    mutual_information.append(mi)


micro_distinct_1 = len(set(unigram)) / len(unigram)
micro_distinct_2 = len(set(bigram)) / len(bigram)

print(f"Average Consistency Score: {sum(Consistency)/len(Consistency)}")
print(f"Average TTR: {sum(TTR)/len(TTR)}")
print(f"Average Distinct_1: {sum(Distinct_1)/len(Distinct_1)}")
print(f"Average Distinct_2: {sum(Distinct_2)/len(Distinct_2)}")
print(f"Micro Average Distinct_1: {micro_distinct_1}")
print(f"Micro Average Distinct_2: {micro_distinct_2}")

print(f"Average Mutual Information: {sum(mutual_information)/len(mutual_information)}")

