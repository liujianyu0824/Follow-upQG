from bs4 import BeautifulSoup
import json
import random
import networkx as nx

def query2graph(file, start_node_label):
    # 读取HTML文件
    with open(file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # 查找包含节点和边的<script>标签
    script_tag = soup.find('script', string=lambda x: x and 'nodes = new vis.DataSet' in x)

    # 提取节点和边的数据
    nodes_data = script_tag.string.split('nodes = new vis.DataSet(')[1].split(']);')[0] + ']'
    edges_data = script_tag.string.split('edges = new vis.DataSet(')[1].split(']);')[0] + ']'

    # 将提取的数据解析为Python对象
    nodes = json.loads(nodes_data)
    edges = json.loads(edges_data)

    # 创建NetworkX图
    G = nx.Graph()
    for node in nodes:
        G.add_node(node['id'])
    for edge in edges:
        G.add_edge(edge['from'], edge['to'])

    # 计算PageRank值
    pagerank = nx.pagerank(G)

    # 创建节点ID到标签的映射
    id_to_label = {node['id']: node['label'] for node in nodes}

    # 创建边的邻接表
    adj_list = {node['id']: [] for node in nodes}
    for edge in edges:
        adj_list[edge['from']].append(edge['to'])
        adj_list[edge['to']].append(edge['from'])

    # 查找与“Artificial intelligence”相连接的所有节点
    target_node_id = None
    for node in nodes:
        if node['label'] == start_node_label:
            target_node_id = node['id']
            break

    if target_node_id is None:
        print(f"未找到标签为'{start_node_label}'的节点。")
    else:
        # 进行随机游走
        steps = 100  # 设置随机游走的步数
        path, visit_counts = random_walk(target_node_id, steps, adj_list)

        # 将访问次数按从大到小排序并输出
        # 计算加权的重要性
        weighted_importance = {node_id: visit_counts[node_id] * pagerank[node_id] for node_id in visit_counts if visit_counts[node_id] > 0}

        # 归一化处理
        max_importance = max(weighted_importance.values())
        min_importance = min(weighted_importance.values())
        normalized_importance = {node_id: (importance - min_importance) / (max_importance - min_importance) for node_id, importance in weighted_importance.items()}
        
        sorted_normalized_importance = sorted(normalized_importance.items(), key=lambda item: item[1], reverse=True)
        sorted_normalized_labels = [(id_to_label[node_id], importance) for node_id, importance in sorted_normalized_importance]

        return nodes, path, normalized_importance, sorted_normalized_importance


def random_walk(start_node_id, steps, adj_list):
    current_node = start_node_id
    path = [current_node]
    visit_counts = {node_id: 0 for node_id in adj_list}  # 初始化访问次数
    visit_counts[current_node] += 1  # 记录起始节点的访问次数

    for _ in range(steps):
        if current_node not in adj_list or not adj_list[current_node]:
            break
        current_node = random.choice(adj_list[current_node])
        path.append(current_node)
        visit_counts[current_node] += 1  # 记录当前节点的访问次数

    return path, visit_counts


nodes, path, weighted_importance, sorted_visit_labels = query2graph("/_output/concepts-general/substance-abuse/concepts-general_substance-abuse_v1.2.3_level2_fully_connected.html", start_node_label="Substance abuse")

# random_id = random.randint(1, len(sorted_visit_labels)-1)
# random_node = sorted_visit_labels[random_id][0]

# for node in nodes:
#     if node['label'] == random_node:
#         wikipedia_content = node['wikipedia_content'].split()
#         linked_item_description = ' '.join(wikipedia_content[:100])
#         break

# # # 节点随机游走RankPage得分
# for label, count in sorted_visit_labels:
#     print(f"{label}: {count}")
