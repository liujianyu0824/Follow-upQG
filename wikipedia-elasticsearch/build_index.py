import json
import os


def load_txt(path):
    with open(path, encoding='UTF-8', errors='ignore') as f:
        data = [i.strip() for i in f.readlines() if len(i) > 0]
    return data

def save_txt(data, path):
    with open(path, 'w', encoding='UTF-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def add_format(data):
    output_data = []
    for item in data:
        item = json.loads(item)
        output_data.append({
            "index": {
                "_id": str(item['id'])
            }
        })
        output_data.append(item)
    return output_data

def walk(root_path):
    for root, _, files in os.walk(root_path):
        for file in files:
            path = os.path.join(root, file)
            print('[TRANSFORMATION] {}'.format(path))
            input_data = load_txt(path)
            output_data = add_format(input_data)
            save_txt(output_data, str(path))

def elastic_import(root_path, index_name):
    for root, _, files in os.walk(root_path):
        for file in files:
            path = os.path.join(root, file)
            print('[IMPORT] {}'.format(path))
            os.system("curl -s -H 'Content-Type: application/x-ndjson' -XPOST localhost:9200/{}/_bulk?pretty --data-binary @{}".format(index_name, path))
    

if __name__ == '__main__':
    index_name = 'extracted_wikidata'
    walk('/wiki_dump/{}'.format(index_name))
    elastic_import('/wiki_dump/{}'.format(index_name), index_name=index_name)

