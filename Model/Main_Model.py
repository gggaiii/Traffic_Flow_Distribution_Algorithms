import csv
import warnings
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
warnings.filterwarnings("ignore")

class District:
    def __init__(self, name, pop_density, poi_num, road_situation, accessibility, importance):
        self.road_situation = road_situation
        self.accessibility = accessibility
        self.importance = importance
        self.name = name
        self.pop_density = pop_density
        self.poi_num = poi_num

# 创建交通网络
def create_traffic_network(csv_file, info_file):
    G = nx.Graph()

    # 读取节点信息
    nodes_info = {}
    with open(info_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            name = row['name']
            pop_density = float(row['pop_density'])
            poi_num = int(row['poi_num'])
            road_situation = float(row['road_situation'])
            accessibility = float(row['accessibility'])
            importance = float(row['importance'])
            nodes_info[name] = (pop_density, poi_num, road_situation, accessibility, importance)

    # 读取边权重
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            source = row['Origin']
            destination = row['Destination']
            weight = float(row['Ta_scaled'])
            G.add_edge(source, destination, weight=weight)

            # 添加节点到网络
            if source not in G:
                G.add_node(source, data=District(source, *nodes_info[source]))
            if destination not in G:
                G.add_node(destination, data=District(destination, *nodes_info[destination]))

    return G

# 更新流量值
def update_traffic_flow(G, flow_csv):
    with open(flow_csv, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            source = row['source']
            destination = row['destination']
            flow = float(row['flow'])
            if G.has_edge(source, destination):
                G[source][destination]['flow'] = flow

    return G

# 使用Bachmann用户最优平衡分配算法分配流量
'''
def balance_traffic(G, paths):
    total_loss = 0.0
    result_data = []  # 存储结果的列表

    for path in paths:
        total_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
        total_loss += total_weight

    if total_loss == 0.0:
        # 处理除以零的情况
        print("无路径可分配流量")
        return G

    for path in paths:
        path_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
        flow_ratio = path_weight / total_loss
        for i in range(len(path) - 1):
            G[path[i]][path[i+1]]['flow'] = flow_ratio

        print("路径:", path)
        print("流量分配:", flow_ratio)
        result_data.append({'Path': path, 'Flow_Distribution': flow_ratio})  # 将结果添加到列表中
        result_df = pd.DataFrame(result_data)  # 创建DataFrame

    print("总抗阻:", total_loss)
    result_df.to_csv('Traffic_Distribution_Result.csv', index=False, encoding='GBK')  # 导出为CSV文件

    return G
'''

def balance_traffic(G, paths):
    total_loss = 0.0
    result_data = []  # 存储结果的列表

    for path in paths:
        total_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
        total_loss += total_weight

    if total_loss == 0.0:
        # 处理除以零的情况
        print("无路径可分配流量")
        return G

    # 定义损失函数
    def loss_function():
        loss = 0.0
        for path in paths:
            path_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
            flow_ratio = path_weight / total_loss
            for i in range(len(path) - 1):
                loss += (G[path[i]][path[i+1]]['flow'] - flow_ratio) ** 2  # 计算每个边的流量分配与目标比例之间的差异
        return loss

    # 使用梯度下降法迭代优化
    alpha = 0.01  # 学习率
    num_iterations = 1000  # 迭代次数

    for iteration in range(num_iterations):
        current_loss = loss_function()  # 计算当前损失函数值

        # 计算梯度
        gradients = {}
        for path in paths:
            path_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
            flow_ratio = path_weight / total_loss
            for i in range(len(path) - 1):
                gradient = 2 * (G[path[i]][path[i+1]]['flow'] - flow_ratio)
                if (path[i], path[i+1]) in gradients:
                    gradients[(path[i], path[i+1])] += gradient
                else:
                    gradients[(path[i], path[i+1])] = gradient

        # 更新流量分配
        for path in paths:
            for i in range(len(path) - 1):
                G[path[i]][path[i+1]]['flow'] -= alpha * gradients[(path[i], path[i+1])]

        new_loss = loss_function()  # 计算更新后的损失函数值
        print('Loss: '+ str(new_loss))

        if abs(new_loss - current_loss) < 1e-6:  # 判断损失函数的变化是否小于阈值
            break

    # 输出最终结果
    for path in paths:
        path_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
        flow_ratio = path_weight / total_loss
        for i in range(len(path) - 1):
            G[path[i]][path[i+1]]['flow'] = flow_ratio

        print("路径:", path)
        print("流量分配:", flow_ratio)
        result_data.append({'Path': path, 'Flow_Distribution': flow_ratio})  # 将结果添加到列表中
        result_df = pd.DataFrame(result_data)  # 创建DataFrame

    print("总抗阻:", total_loss)
    result_df.to_csv('Traffic_Distribution_Result.csv', index=False, encoding='GBK')  # 导出为CSV文件

    return G

def visualize_traffic_network(G):
    # 设置节点的位置
    pos = nx.spring_layout(G)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)

    # 绘制边
    edge_labels = nx.get_edge_attributes(G, 'flow')
    edge_widths = [flow * 100 for flow in edge_labels.values()]  # 缩放因子为0.1
    edge_colors = [cm.RdYlGn((flow - min(edge_labels.values())) / (max(edge_labels.values()) - min(edge_labels.values()))) for flow in edge_labels.values()]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)

    # 绘制节点上的标签
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')

    # 绘制边的权重标签
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # 显示图形
    plt.axis('off')
    plt.title('Traffic Network')
    plt.show()

def visualize_allocation(G, allocation):
    # 设置节点的位置
    pos = nx.spring_layout(G)

    # 提取节点的分配值
    node_values = [allocation[node] for node in G.nodes()]

    # 绘制节点，根据分配值设置颜色和大小
    nx.draw_networkx_nodes(G, pos, node_color=node_values, cmap='cool', node_size=500, alpha=0.8)

    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray')

    # 绘制节点上的标签
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='cool')
    sm.set_array(node_values)
    plt.colorbar(sm, label='Allocation')

    # 显示图形
    plt.axis('off')
    plt.title('Allocation')
    plt.show()

def main():
    csv_file = 'D:/毕业论文/工程项目/Model/Ta.csv'  # 替换为你的边权重 CSV 文件路径
    flow_csv = 'D:/毕业论文/工程项目/Model/Traffic_Flow.csv'  # 替换为你的流量 CSV 文件路径
    info_file = 'D:/毕业论文/工程项目/Model/Information.csv'  # 替换为你的节点信息 CSV 文件路径

    G = create_traffic_network(csv_file, info_file)
    G = update_traffic_flow(G, flow_csv)

    # paths = [["A", "B", "C"], ["A", "D"]]
    # 读取 traffic_data.csv 文件
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行

        # 创建一个空的路径列表
        paths = []

        # 遍历每一行数据
        for row in reader:
            source = row[0]  # 源节点信息
            destination = row[1]  # 目标节点信息

            # 将源节点和目标节点添加到路径列表中
            paths.append((source, destination))
    G = balance_traffic(G, paths)
    visualize_traffic_network(G)

if __name__ == '__main__':
    main()