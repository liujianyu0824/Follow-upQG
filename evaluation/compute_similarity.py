from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm

# 加载预训练的BERT模型
model = SentenceTransformer('/embedding/all-MiniLM-L6-v2')
model.cuda()

with open('/result/no_reranker_beta=1.json', 'r') as f:
    original_data = json.load(f)


sim_wiki_q = []
sim_wiki_fq = []
sim_q_fq = []

for data in tqdm(original_data[:300], desc="Processing data"):
    sentence1 = data['related_node_definition']
    sentence2 = data['question']
    # sentence3 = data['llm_knowledge']

    if len(data['generated_follow-up']) == 0 :
        continue
    elif type(data['generated_follow-up'][0]) == dict:
        continue
    elif type(data['generated_follow-up']) == str:
        sentence3 = data['generated_follow-up']
    elif data['generated_follow-up'][0] != '':
        # text2 = ' '.join(data['generated_follow-up'])
        sentence3 = data['generated_follow-up'][0]
    else:
        continue


    # 将句子转换为嵌入向量
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    embedding3 = model.encode(sentence3, convert_to_tensor=True)

    # 计算余弦相似度
    wiki_q_similarity = util.pytorch_cos_sim(embedding1, embedding2)
    wiki_fq_similarity = util.pytorch_cos_sim(embedding1, embedding3)
    q_fq_similarity = util.pytorch_cos_sim(embedding2, embedding3)


    sim_wiki_q.append(wiki_q_similarity.item())
    sim_wiki_fq.append(wiki_fq_similarity.item())
    sim_q_fq.append(q_fq_similarity.item())

print(f"Wiki Question Similarity: {sum(sim_wiki_q)/len(sim_wiki_q)}")
print(f"Wiki Follow-up Similarity: {sum(sim_wiki_fq)/len(sim_wiki_fq)}")
print(f"Question Follow-up Similarity: {sum(sim_q_fq)/len(sim_q_fq)}")

