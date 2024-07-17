import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import geopandas as gpd

# 加载边信息
edges_df = pd.read_csv('Ta_Real_Topo_Test.csv')

# 创建有向图
G = nx.DiGraph()

# 添加边
for index, row in edges_df.iterrows():
    G.add_edge(row['Origin'], row['Destination'], weight=row['Ta_scaled'], flow=0, capacity=row['Ta_scaled'])

# 移除不需要的边
unwanted_edges = [
    ('Jianghan', 'Donghu'), ('Jianghan', 'Jingkai'), ('Jianghan', 'Hanyang'), ('Jianghan', 'Qingshan'), ('Jianghan', 'Hongshan'),
    ('Jingkai', 'Qiaokou'), ('Jingkai', 'Jianghan'), ('Jingkai', 'Jiangan'), ('Jingkai', 'Wuchang'), ('Jingkai', 'Donghu'),
    ('Jingkai', 'Qingshan'), ('Jingkai', 'Dongxihu'), ('Jingkai', 'Hongshan'), ('Dongxihu', 'Jiangan'), ('Dongxihu', 'Wuchang'),
    ('Dongxihu', 'Qingshan'), ('Dongxihu', 'Hongshan'), ('Dongxihu', 'Donghu'), ('Donghu', 'Jiangan'), ('Donghu', 'Qiaokou'),
    ('Donghu', 'Hanyang'), ('Donghu', 'Jingkai'), ('Qingshan', 'Hanyang'), ('Qiaokou', 'Hongshan'), ('Qiaokou', 'Qingshan'),
    ('Qiaokou', 'Wuchang'), ('Jiangan', 'Hongshan')
]
for (u, v) in unwanted_edges:
    if G.has_edge(u, v):
        G.remove_edge(u, v)
        G.remove_edge(v, u)

# 调整权重
for u, v, data in G.edges(data=True):
    if (u, v) in [('Jianghan', 'Jiangan'), ('Jiangan', 'Wuchang'), ('Jianghan', 'Qiaokou')]:
        data['weight'] *= 0.5
    if (u, v) in [('Jiangan', 'Jianghan'), ('Wuchang', 'Jiangan'), ('Qiaokou', 'Jiangan')]:
        data['weight'] *= 0.5
    if 'Dongxihu' in (u, v) or 'Jingkai' in (u, v) or 'Hongshan' in (u, v):
        data['weight'] *= 3.5

# 节点位置信息（使用经纬度）
node_positions = {
    'Jianghan': (114.271, 30.600), 'Jiangan': (114.289, 30.580), 'Wuchang': (114.315, 30.530),
    'Qiaokou': (114.230, 30.570), 'Dongxihu': (114.130, 30.620), 'Donghu': (114.350, 30.560),
    'Jingkai': (114.210, 30.490), 'Hongshan': (114.330, 30.500), 'Qingshan': (114.340, 30.630),
    'Hanyang': (114.220, 30.550)
}

# 载入Shapefile底图
gdf = gpd.read_file('D:\\City_Data\\武汉建筑数据\\420100.shp')
# gdf = gpd.read_file('D:\\毕业论文\\工程项目\\Network Analysis\\武汉市\\武汉市.shp')

# 创建图形和轴
fig, ax = plt.subplots(figsize=(12, 12))

# 增强边界线
gdf.boundary.plot(ax=ax, color='grey', linewidth=1.5)

# 计算旅行时间等函数应在此添加
# 调整权重和容量，防止没有容量的数据
for u, v, data in G.edges(data=True):
    if 'capacity' not in data or data['capacity'] <= 0:
        data['capacity'] = 500  # 假设默认容量值

# 加载交通流数据
traffic_df = pd.read_csv('Traffic_Flow_Real_Topo_Test.csv')


# 计算旅行时间的函数，使用简化的BPR函数
def calculate_travel_time(G):
    # alpha = 1
    beta = 4
    for u, v, data in G.edges(data=True):
        alpha = data['weight']
        flow = data['flow']
        capacity = data['capacity']
        free_travel_time = 1
        travel_time = free_travel_time * (1 + alpha * ((flow / capacity) ** beta))
        data['travel_time'] = travel_time


def calculate_total_travel_time(G):
    """计算网络中所有边的总旅行时间"""
    total_travel_time = sum(data['travel_time'] * data['flow'] for _, _, data in G.edges(data=True))
    return total_travel_time


def frank_wolfe(G, traffic_df, max_iterations=100, convergence_threshold=0.01, min_edge_flow=100):
    previous_total_travel_time = float('inf')
    for iteration in range(max_iterations):
        # 每次迭代前清空流量
        for _, _, data in G.edges(data=True):
            data['flow'] = 0

        # 分配每个OD对的流量到最短路径
        for index, row in traffic_df.iterrows():
            source = row['source']
            destination = row['destination']
            flow = row['flow']
            if nx.has_path(G, source, destination):
                shortest_path = nx.shortest_path(G, source, destination, weight='travel_time')
                for i in range(len(shortest_path) - 1):
                    G[shortest_path[i]][shortest_path[i + 1]]['flow'] += flow

        # 强制每条边至少有最小流量
        for u, v, data in G.edges(data=True):
            data['flow'] = max(data['flow'], min_edge_flow)

        # 更新旅行时间
        calculate_travel_time(G)

        # 计算当前的总旅行时间
        current_total_travel_time = calculate_total_travel_time(G)

        # 检查收敛条件
        travel_time_change = abs(current_total_travel_time - previous_total_travel_time)
        if travel_time_change < convergence_threshold:
            print(f"Iteration {iteration + 1}: Converged with total travel time = {current_total_travel_time/10**13}")
            break

        print(f"Iteration {iteration + 1}: Total travel time = {current_total_travel_time/10**14}")

        # 更新前一次迭代的旅行时间
        previous_total_travel_time = current_total_travel_time


# 假设G已创建并添加了相关边和数据，traffic_df是Pandas DataFrame
# 调用Frank-Wolfe算法
frank_wolfe(G, traffic_df)

# 创建一个空的列表来收集边的数据
edges_data = []

# 绘图和显示流量
def print_edge_flows_and_draw_graph(G, pos):
    edge_flows = np.array([data['flow'] for _, _, data in G.edges(data=True)])
    norm = plt.Normalize(vmin=edge_flows.min(), vmax=edge_flows.max())
    cmap = plt.get_cmap('RdYlGn_r')
    edge_colors = [cmap(norm(flow)) for flow in edge_flows]
    edge_widths = [5 * (flow / max(edge_flows) + 0.1) for flow in edge_flows]
    print("Edge flows:")
    for (u, v, data) in G.edges(data=True):
        edges_data.append({'origin': u, 'destination': v, 'flow': data['flow']})
        print(f"({u}, {v}) - Flow: {data['flow']}")
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Traffic flow', ax=ax)
    plt.title("Traffic Flow Visualization on Geographic Base Map")
    plt.show()

    # 使用收集的数据创建一个DataFrame
    edges_df = pd.DataFrame(edges_data)

    # 输出DataFrame到CSV文件
    edges_df.to_csv('edge_flows.csv', index=False)


# 显示结果
print_edge_flows_and_draw_graph(G, node_positions)
