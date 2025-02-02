import os
import re
import json
# import subprocess

from openai import OpenAI
import httpx
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util

from query2graph import query2graph
from es_rerank import *

# 连接到 Elasticsearch
es = Elasticsearch("http://localhost:9200")


with open("/result/our_method.json", "r") as f:
    original_data = json.load(f)

def extract_json(passage):
    pattern = re.compile(r'({.*?(\n}))', re.DOTALL)
    match = pattern.search(passage)

    extracted_content = match.group(1)
    passage = json.loads(extracted_content)
    return passage

client = OpenAI(
    base_url="", 
    api_key="",
    http_client=httpx.Client(
        base_url="",
        follow_redirects=True,
    )
)


i = 0
for data in original_data:
    # Recognition Module
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a data extractor."},
            {"role": "user", "content": 'I will provide you with a Question-Answer pair, you need to extract one TOPIC word for that conversation and the three most related KEYWORDS for the Q&A content from it. The content is as follows:\n"""\n' + '"Question":' + data["question"] + '\n"Answer": ' + data["answer"] + '''\n"""\nPlease return the result in json:
        {
            "topic":[],
            "keywords":[]
        }
        You only need to return json data, no extra content is needed.'''}
        ]
    )

    
    json_data = extract_json(completion.choices[0].message.content)            


    if len(json_data['topic']) == 0:
        json_data['topic'].append(json_data['keywords'][0])

    # 定义查询
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "title": json_data['topic'][0]
                        }
                    },
                ]
            }
        }
    }
 

    json_data['keywords'] = sorted(json_data['keywords'], key=len, reverse=True) # 按长度降序排列关键字,更长的keywords携带的信息量更大

    # 执行查询
    response = es.search(index="extracted_wikidata", body=query)

    if len(response['hits']['hits']) == 0:
        query["query"]['bool']['must'][0] = {"match": {"title": json_data['topic'][0]}}
        response = es.search(index="extracted_wikidata", body=query)
        if len(response['hits']['hits']) == 0:
            query["query"]['bool']['must'].pop()
            for keyword in json_data['keywords']:
                if len(response['hits']['hits']) >1:
                    query["query"]['bool']['must'].append({"match": {"text": keyword}})
                    response = es.search(index="extracted_wikidata", body=query)
                    if len(response['hits']['hits']) == 0:
                        query["query"]['bool']['must'].pop()
                        response = es.search(index="extracted_wikidata", body=query)
                    elif len(response['hits']['hits']) == 1:
                        break

    # 执行重排
    passages = []
    titles = []
    for hit in response['hits']['hits']:
        titles.append(hit['_source']['title'])
        wikipedia_content = hit['_source']['text'].split()
        passages.append(' '.join(wikipedia_content[:128]))

    question = json_data['topic'][0] + ','.join(json_data['keywords'])

    # # print(f"question: {question}")
    # # print(f"passages: {passages}")

    topk_scores, indexes = inference(question, passages, batch_size=4)

    es_search_title = titles[indexes[0]]
    # es_search_title = response['hits']['hits'][0]['_source']['title']

    # Selection Module
    # 构建graph
    os.system(f'. /.env && llmgraph concepts-general "https://en.wikipedia.org/wiki/{es_search_title.replace(" ", "_")}" --levels 2')

    # 解析标准输出以找到生成的 HTML 文件路径
    output_folder = f'/_output/concepts-general/{es_search_title.lower().replace(" ", "-")}'
    html_file = f'{output_folder}/concepts-general_{es_search_title.lower().replace(" ", "-")}_v1.2.3_level2_fully_connected.html'

    nodes, path, weighted_importance, sorted_visit_labels = query2graph(html_file, start_node_label=es_search_title)

    # 节点随机游走RankPage得分
    # for label, count in sorted_visit_labels:
    #     print(f"{label}: {count}")


    # 加载预训练的BERT模型
    encode_model = SentenceTransformer('/embedding/all-MiniLM-L6-v2')
    encode_model.cuda()

    similarity_scores = {}
    for node in nodes:
        # 将句子转换为嵌入向量
        wikipedia_content = node['wikipedia_content'].split()
        sentence1 = ' '.join(wikipedia_content[:100])
        sentence2 = question
        embedding1 = encode_model.encode(sentence1, convert_to_tensor=True)
        embedding2 = encode_model.encode(sentence2, convert_to_tensor=True)

        # 计算余弦相似度
        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
        similarity_scores[node['label']] = cosine_similarity.item()

    # # print(similarity_scores)

    b = 1  # beta value

    for k,v in similarity_scores.items():
        weighted_importance[k] = weighted_importance[k] + b * v

    # print(weighted_importance)

    id_to_label = {node['id']: node['label'] for node in nodes}
    sorted_weighted_importance = sorted(weighted_importance.items(), key=lambda item: item[1], reverse=True)
    sorted_weighted_labels = [(id_to_label[node_id], importance) for node_id, importance in sorted_weighted_importance]

    # 将随机游走除初始节点外访问次数最高的节点作为related node，并返回其定义
    for label, count in sorted_weighted_labels:
        if label != es_search_title:
            for node in nodes:
                if node['label'] == label:
                    wikipedia_content = node['wikipedia_content'].split()
                    linked_item_description = ' '.join(wikipedia_content[:100])
                    break
            break

    # # Ablation——no_kgselection
    # random_id = random.randint(1, len(sorted_visit_labels)-1)
    # random_node = sorted_visit_labels[random_id][0]
    # for node in nodes:
    #     if node['label'] == random_node:
    #         wikipedia_content = node['wikipedia_content'].split()
    #         linked_item_description = ' '.join(wikipedia_content[:100])
    #         break

    # print(f"related node: {label}\ndefinition: {linked_item_description}")


    # Fusion Module
    relation_linked_item = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": '"Question":' + data['question'] + '\n"Answer": ' + data['answer'] + '\nPlease continue writing the following sentences with a few sentences based on the content of the Question-Answer pair to reflect the association with the question-answer pair. You only need to output the continued sentence: \n' + '"""\n' + linked_item_description + '\n"""\n'}
        ]
    )

    relation_linked_item = relation_linked_item.choices[0].message.content

    #Ablation——no_llmknowledge
    # relation_linked_item = ''
    # print(relation_linked_item)
#  which related to the given "Related Knowledge"
    followup_question = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": '"Question": ' + data['question'] + '\n"Answer": ' + data['answer'] + '\n"Related Knowledge": ' + linked_item_description + relation_linked_item + '''\n\nBased on these information, please raise a follow-up question that are relevant to the QA content and that are thoughtful. Please return the result in json:
        {
            "followup_question":[]
        }
        You only need to return json data, no extra content is needed.'''}
        ]
    )


    followup_question = extract_json(followup_question.choices[0].message.content)

    
    while len(followup_question['followup_question']) == 0:
        followup_question = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": '"Question": ' + data['question'] + '\n"Answer": ' + data['answer'] + '\n"Related Knowledge": ' + linked_item_description + relation_linked_item + '''\n\nBased on these information, please raise a follow-up question that are relevant to the QA content and that are thoughtful. Please return the result in json:
                {
                    "followup_question":[]
                }
                You only need to return json data, no extra content is needed.'''}
                ]
            )
        followup_question = extract_json(followup_question.choices[0].message.content)
    # print(followup_question)

    original_data[original_data.index(data)]['related_node'] = label
    # original_data[original_data.index(data)]['related_node'] = random_node
    original_data[original_data.index(data)]['related_node_definition'] = linked_item_description

    original_data[original_data.index(data)]['llm_knowledge'] = relation_linked_item
    original_data[original_data.index(data)]['generated_follow-up'] = followup_question['followup_question']

    with open("/result/our_method.json", "w") as f:
        json.dump(original_data, f, indent=4)

    i += 1
    print('i=', i)
    



