import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np

# 加载边信息
edges_df = pd.read_csv('Ta_Real_Topo_Test.csv')

# 创建有向图
G = nx.DiGraph()

# 添加边
for index, row in edges_df.iterrows():
    G.add_edge(row['Origin'], row['Destination'], weight=row['Ta_scaled'], flow=0)

# 移除不需要的边
unwanted_edges = [
    ('Jianghan', 'Donghu'), ('Jianghan', 'Jingkai'), ('Jianghan', 'Hanyang'), ('Jianghan', 'Qingshan'), ('Jianghan', 'Hongshan'),
    ('Jingkai', 'Qiaokou'), ('Jingkai', 'Jianghan'), ('Jingkai', 'Jiangan'), ('Jingkai', 'Wuchang'), ('Jingkai', 'Donghu'),
    ('Jingkai', 'Qingshan'), ('Jingkai', 'Dongxihu'), ('Jingkai', 'Hongshan'), ('Dongxihu', 'Jiangan'), ('Dongxihu', 'Wuchang'),
    ('Dongxihu', 'Qingshan'), ('Dongxihu', 'Hongshan'), ('Dongxihu', 'Donghu'), ('Donghu', 'Jiangan'), ('Donghu', 'Qiaokou'),
    ('Donghu', 'Hanyang'), ('Donghu', 'Jingkai'), ('Qingshan', 'Hanyang')
]
for (u, v) in unwanted_edges:
    if G.has_edge(u, v):
        G.remove_edge(u, v)
for (v, u) in unwanted_edges:
    if G.has_edge(v, u):
        G.remove_edge(v, u)

# 调整权重
for u, v, data in G.edges(data=True):
    if (u, v) in [('Jianghan', 'Jiangan'), ('Jiangan', 'Wuchang'), ('Jianghan', 'Qiaokou')]:
        data['weight'] *= 0.1
    if 'Dongxihu' in (u, v) or 'Jingkai' in (u, v):
        data['weight'] *= 3.5

# 加载交通流数据
traffic_df = pd.read_csv('Traffic_Flow_Real_Topo_Test.csv')

# 计算和更新路径
def save_paths(G, traffic_df, num_paths=5):
    paths_dict = {}
    count = 1
    for index, row in traffic_df.iterrows():
        source = row['source']
        destination = row['destination']
        key = f"{source}-{destination}"
        if nx.has_path(G, source, destination):
            paths = list(nx.shortest_simple_paths(G, source, destination, weight='weight'))[:num_paths]
            paths_dict[key] = [list(path) for path in paths]
        print(f"{count}/100 Finished")
        count += 1
    with open('saved_paths.json', 'w') as f:
        json.dump(paths_dict, f, indent=2)

# 取消注释以重新生成路径
# save_paths(G, traffic_df, num_paths=5)

def load_paths(filename='saved_paths.json'):
    with open(filename, 'r') as f:
        return json.load(f)

paths_dict = load_paths()

# 梯度下降和阻抗计算
# alpha = 0.01
iterations = 100
tolerance = 1e-3

def calculate_total_impedance(G):
    return sum(data['weight'] * data['flow'] for _, _, data in G.edges(data=True))

def update_weights_dynamically(G, alpha, decay=0.5):
    """ 根据当前的流量动态更新权重 """
    for u, v, data in G.edges(data=True):
        # 计算新权重，这里使用了衰减因子和流量的线性组合
        current_flow = data['flow']
        old_weight = data['weight']
        new_weight = (1 - decay) * old_weight + decay * (1 / (current_flow + 1))  # 防止除以零
        data['weight'] = max(new_weight, 0.01)  # 防止权重太低

def gradient_descent_update_using_saved_paths(G, traffic_df, paths_dict):
    """ 使用保存的路径来更新流量 """
    for _, _, data in G.edges(data=True):
        data['flow'] = 0
    for index, row in traffic_df.iterrows():
        key = f"{row['source']}-{row['destination']}"
        if key in paths_dict:
            paths = paths_dict[key]
            flow_per_path = row['flow'] / len(paths)
            for path in paths:
                for i in range(len(path) - 1):
                    G[path[i]][path[i + 1]]['flow'] += flow_per_path

# 主循环中引入动态权重更新
previous_impedance = calculate_total_impedance(G)  # 初始化阻抗值
for i in range(iterations):
    alpha = data['weight']
    gradient_descent_update_using_saved_paths(G, traffic_df, paths_dict)
    update_weights_dynamically(G, alpha)  # 更新权重
    current_impedance = calculate_total_impedance(G)  # 计算当前的总阻抗
    if abs(previous_impedance - current_impedance) < tolerance:
        print(f"Convergence reached at iteration {i+1}.")
        break
    print(f"Iteration {i+1}: Total Impedance = {current_impedance}")
    previous_impedance = current_impedance  # 更新上一次迭代的阻抗值以便下次比较



# 绘图和流量显示
node_positions = {
    'Jianghan': (0, 1), 'Jiangan': (0, 0), 'Wuchang': (1, 0), 'Qiaokou': (-1, 0), 'Dongxihu': (-1, 1),
    'Donghu': (2, 0), 'Jingkai': (-1, -2), 'Hongshan': (1, -1), 'Qingshan': (1, 1), 'Hanyang': (-1, -1)
}

def print_edge_flows_and_draw_graph(G, pos):
    plt.figure(figsize=(12, 8))
    edge_flows = np.array([data['flow'] for _, _, data in G.edges(data=True)])
    norm = plt.Normalize(vmin=edge_flows.min(), vmax=edge_flows.max())
    cmap = plt.get_cmap('RdYlGn_r')
    edge_colors = [cmap(norm(flow)) for flow in edge_flows]
    edge_widths = [5 * (flow / max(edge_flows) + 0.1) for flow in edge_flows]
    print("Edge flows:")
    for (u, v, data) in G.edges(data=True):
        print(f"({u}, {v}) - Flow: {data['flow']}")
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Traffic flow')
    plt.title("Traffic Flow Visualization")
    plt.show()

print_edge_flows_and_draw_graph(G, node_positions)
