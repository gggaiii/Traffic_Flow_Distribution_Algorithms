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

unwanted_edges = [('Jianghan', 'Donghu'),
                  ('Jianghan', 'Jingkai'),
                  ('Jianghan', 'Hanyang'),
                  ('Jianghan', 'Qingshan'),
                  ('Jianghan', 'Hongshan'),
                  ('Jingkai', 'Qiaokou'),
                  ('Jingkai', 'Jianghan'),
                  ('Jingkai', 'Jiangan'),
                  ('Jingkai', 'Wuchang'),
                  ('Jingkai', 'Donghu'),
                  ('Jingkai', 'Qingshan'),
                  ('Jingkai', 'Dongxihu'),
                  ('Jingkai', 'Hongshan'),
                  ('Dongxihu', 'Jiangan'),
                  ('Dongxihu', 'Wuchang'),
                  ('Dongxihu', 'Qingshan'),
                  ('Dongxihu', 'Hongshan'),
                  ('Dongxihu', 'Donghu'),
                  ('Donghu', 'Jiangan'),
                  ('Donghu', 'Qiaokou'),
                  ('Donghu', 'Hanyang'),
                  ('Donghu', 'Jingkai'),
                  ('Qingshan', 'Hanyang')]  # 可以根据需要扩展此列表
for (u, v) in unwanted_edges:
    if G.has_edge(u, v):
        G.remove_edge(u, v)
for (v, u) in unwanted_edges:
    if G.has_edge(v, u):
        G.remove_edge(v, u)

for u, v, data in G.edges(data=True):
    # 特别处理某些边的优先级，如Jianghan与Jiangan、Jiangan与Wuchang
    if (u, v) == ('Jianghan', 'Jiangan') or (u, v) == ('Jiangan', 'Wuchang') or (u, v) == ('Jianghan', 'Qiaokou'):
        data['weight'] *= 0.1
    # 增加对含有"Dongxihu"或"Jingkai"的边的权重调整
    if 'Dongxihu' in (u, v) or 'Jingkai' in (u, v):
        data['weight'] *= 3.5

# 加载交通流数据
traffic_df = pd.read_csv('Traffic_Flow_Real_Topo_Test.csv')

'''def save_paths(G, traffic_df, num_paths=5):
    paths_dict = {}
    count = 1
    for index, row in traffic_df.iterrows():
        source = row['source']
        destination = row['destination']
        key = f"{source}-{destination}"  # 使用字符串形式的键
        if nx.has_path(G, source, destination):
            paths = list(nx.shortest_simple_paths(G, source, destination, weight='weight'))[:num_paths]
            paths_dict[key] = [list(path) for path in paths]

        print(str(count)+'/100 Finished')
        count = count+1

    # 保存到 JSON 文件
    with open('saved_paths.json', 'w') as f:
        json.dump(paths_dict, f, indent=2)

# 调用函数，保存路径
save_paths(G, traffic_df, num_paths=5)'''

def load_paths(filename='saved_paths.json'):
    with open(filename, 'r') as f:
        paths_dict = json.load(f)
    return paths_dict

paths_dict = load_paths()

# 参数设置
alpha = 0.01  # 学习率
iterations = 100  # 迭代次数
tolerance = 1e-3  # 容忍误差，用于提前停止迭代

def calculate_total_impedance(G):
    """计算总阻抗"""
    total_impedance = 0
    for u, v, data in G.edges(data=True):
        total_impedance += data['weight'] * data['flow']
    return total_impedance

def gradient_descent_update_using_saved_paths(G, traffic_df, alpha, paths_dict):
    # 先清空当前所有边的流量
    for _, _, data in G.edges(data=True):
        data['flow'] = 0

    # 使用保存的路径分配流量
    for index, row in traffic_df.iterrows():
        source = row['source']
        destination = row['destination']
        flow_amount = row['flow']
        key = f"{source}-{destination}"

        if key in paths_dict:
            paths = paths_dict[key]
            num_paths_found = len(paths)
            flow_per_path = flow_amount / num_paths_found

            for path in paths:
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    G[u][v]['flow'] += flow_per_path

    # 更新权重
    for u, v, data in G.edges(data=True):
        data['weight'] = max(data['weight'] - alpha * data['flow'], 0.01)

for i in range(iterations):
    previous_impedance = calculate_total_impedance(G)
    gradient_descent_update_using_saved_paths(G, traffic_df, 0.01, paths_dict)
    current_impedance = calculate_total_impedance(G)
    print(f"Iteration {i+1}: Total Impedance = {current_impedance}")

    # 检查收敛性
    if abs(previous_impedance - current_impedance) < tolerance:
        print("Convergence reached.")
        break

node_positions = {
    'Jianghan': (0, 1),
    'Jiangan': (0, 0),
    'Wuchang': (1, 0),
    'Qiaokou': (-1, 0),
    'Dongxihu': (-1, 1),
    'Donghu': (2, 0),
    'Jingkai': (-1, -2),
    'Hongshan': (1, -1),
    'Qingshan': (1, 1),
    'Hanyang': (-1, -1)
}

'''def print_edge_flows_and_draw_graph(G, pos):
    # 设置图的绘制样式
    plt.figure(figsize=(12, 8))

    # 为了让边的粗细可视化，需要确定每条边的流量
    edge_flows = [data['flow'] for _, _, data in G.edges(data=True)]
    edge_colors = edge_flows  # 也可以设置边的颜色与流量相关
    edge_widths = [flow / max(edge_flows) * 10 if max(edge_flows) > 0 else 1 for flow in edge_flows]  # 归一化流量并调整为合适的宽度范围

    # 打印每条边的流量信息
    print("Edge flows:")
    for (u, v, data) in G.edges(data=True):
        print(f"({u}, {v}) - Flow: {data['flow']}")

    # 绘制网络图
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.6)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.Blues, alpha=0.7)

    plt.title('Network Traffic Visualization')
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.Blues), ax=plt.gca(), orientation='vertical', label='Edge Flow Intensity')
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 调用函数打印流量和绘制图，传递自定义位置
print_edge_flows_and_draw_graph(G, node_positions)'''

def print_edge_flows_and_draw_graph(G, pos):
    plt.figure(figsize=(12, 8))

    # 获取所有边的流量，并将其归一化以用于颜色映射和宽度计算
    edge_flows = np.array([data['flow'] for _, _, data in G.edges(data=True)])
    norm = plt.Normalize(vmin=edge_flows.min(), vmax=edge_flows.max())

    # 创建颜色映射对象，从绿色到红色
    cmap = plt.get_cmap('RdYlGn_r')

    # 根据流量设置边的颜色和宽度
    edge_colors = [cmap(norm(flow)) for flow in edge_flows]
    edge_widths = [5 * (flow / max(edge_flows) + 0.1) for flow in edge_flows]  # 保证即使是最小流量，边也清晰可见

    # 打印每条边的流量信息
    print("Edge flows:")
    for (u, v, data) in G.edges(data=True):
        print(f"({u}, {v}) - Flow: {data['flow']}")

    # 绘制网络图
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)

    # 显示颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Traffic Flow Intensity')

    plt.title('Network Traffic Visualization')
    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 使用前面设置的节点位置调用函数
print_edge_flows_and_draw_graph(G, node_positions)
