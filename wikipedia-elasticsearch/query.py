from elasticsearch import Elasticsearch
import json

def search_elasticsearch(index_name, query):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    # print(es.indices.get(index="extracted_wikidata"))
    response = es.search(index=index_name, body=query)
    
    return response

# 更宽松的查询条件，匹配包含 'Python' 关键字的文档
query = {
    "query": {
        "match": {
            "title": "Python"
        }
    }
}


index_name = 'extracted_wikidata'
result = search_elasticsearch(index_name, query)

# 将结果转换为字典格式以便进行 JSON 序列化
result_dict = result

# 打印查询到的文档
if result_dict['hits']['total']['value'] > 0:
    for hit in result_dict['hits']['hits']:
        print(f"Document ID: {hit['_id']}")
        print(f"text: {hit['_source']['title']}")
else:
    print("No documents found")

# 输出完整的响应结果（如果需要调试）
# print(json.dumps(result_dict, indent=2))
