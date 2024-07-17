说明文档
程序运行环境
Windows 11 Version21H1
开发语言
Python 3.10
库函数(首字母排序)
folium 0.15.0
geopandas 0.12.2
geopy 2.4.1
matplotlib 3.6.2
networkx 3.0
numpy 1.24.2
osm 1.4
osmnet 0.1.7
osmnx 1.7.0
pandana 0.7
pandas 1.3.5
requests 2.31.0
scikit-learn 1.3.1
shapely 2.0.1
transbigdata 0.5.3


程序结构与执行顺序

========

--1 OD分析

----1.1 OD Hotmap
--------生成OD概括地图
--------运行结果：OD-Hotmap.png

----1.2 OD_AgeTime
--------生成年龄和时间分布
--------运行结果：Age-Time.png

----1.3 OD AgeMethod
--------生成年龄与出行方式分布（箱型图）
--------运行结果：Age-Method.png

========

--2 Network Analysis

----2.1 Accessibility（需要网络，耗时较长）
--------生成规定时间区域内的可达性范围
--------运行结果：Accessibility.png

----2.2 District
--------获取武汉市行政区划Polygon

----2.3 Metro Analysis Shanghai
--------上海地铁客流断面分析与可视化
--------运行结果shanghai metro analysis 0-3.png

----2.4 Wuhan Road
--------获取武汉市道路拓扑结构
--------运行结果：Edge Node 武汉市文件夹

========

--3 Model

----3.1 Ta Calculate RealTopo
--------计算径路阻抗系数Ta
--------运行结果：csv

----3.2 Main Model Real Topo Test
--------网络雏形测试

----3.3 Gradient Model
--------用梯度下降法收敛，使用最短寻路平均分配
--------迭代61次收敛，Cost = 195.232

----3.4 FrankWolfe Model
--------添加FrankWolfe动态分配方法，并不贴近实际需求
--------迭代21次收敛，Cost = 203.829

----3.5 UserBest Model
--------用户平衡分配方法
--------迭代4次收敛，Cost = 152.507
