from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

from collections import Counter
import math
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

with open('/result/generated_followupQ_beta=nan.json', 'r') as f:
    original_data = json.load(f)

BLEU_1 = []
BLEU_2 = []
Perplexity = []
Topic_consistency = []

# 数据预处理
def preprocess(text):
    return [word.lower() for word in text.split()]

for data in tqdm(original_data, desc="Processing data"):
    text1 = data['question'] + data['answer']
    text2 = data['related_node_definition']

    texts = [text1, text2]

    processed_texts = [preprocess(text) for text in texts]

    # 计算BLEU-1和BLEU-2分数
    reference = [processed_texts[0]]  # 把第一个文本作为参考
    candidate = processed_texts[1]    # 把第二个文本作为候选

    smoothie = SmoothingFunction().method4
    bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)

    BLEU_1.append(bleu_1)
    BLEU_2.append(bleu_2)


    # 创建字典和语料库
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # 训练LDA模型
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, random_state=42)


    # 加载预训练模型和tokenizer
    model_name = '/embedding/gpt2-base'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 将文本编码为模型输入
    inputs = tokenizer(text1 + text2, return_tensors='pt')

    # 检查 token 索引是否在模型的词汇表范围内
    if torch.max(inputs['input_ids']) >= model.config.vocab_size:
        print("Error: Token index out of range")
        continue

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss)


    Perplexity.append(perplexity.item())


    # 计算主题一致性（PMI）
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    # print(f"Coherence Score: {coherence_lda}")

    Topic_consistency.append(coherence_lda)


print(f"BLEU-1 Score: {sum(BLEU_1)/len(BLEU_1)}")
print(f"BLEU-2 Score: {sum(BLEU_2)/len(BLEU_2)}")
print(f"Perplexity: {sum(Perplexity)/len(Perplexity)}")
print(f"Coherence Score: {sum(Topic_consistency)/len(Topic_consistency)}")
