import osmnx as ox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
import geopandas as gpd

import itertools
from scipy.spatial.distance import cdist
from math import radians, cos, sin, asin, sqrt
from pyproj import Proj, transform


# 计算地图距离（沿着道路的距离）
# 计算地图距离（沿着道路的距离）
def compute_optimized_map_distances(G, nodes, nearest_node_depot=None):
    cust_length = len(nodes)
    if nearest_node_depot is None:
        nearest_node_depot = nodes
    depot_length = len(nearest_node_depot)
    distance_matrix = np.full((depot_length, cust_length), np.inf)  # 用无穷大初始化距离矩阵
    for i in range(depot_length):
        for j in range(cust_length):
            # for i, j in itertools.combinations(range(length), 2):
            try:
                # 尝试计算两个节点之间的最短路径长度
                distance = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length')
            except nx.NetworkXNoPath:
                # 如果不存在路径，则距离保持为无穷大
                continue
            # 更新距离矩阵
            distance_matrix[i, j] = np.ceil(distance / 100)
            # distance_matrix[j, i] = np.ceil(distance/100)  # 距离矩阵是对称的
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix


# 手动实现的haversine函数
def haversine(lon1, lat1, lon2, lat2):
    # 将经纬度转换成弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    earth_radius = 6371000  # 地球平均半径，单位：米
    dist = earth_radius * c
    return dist


# 使用numpy矩阵操作和手动实现的haversine函数计算距离矩阵
def compute_optimized_euclidean_distances(lats, lons, depot_lats=None, depot_lons=None):
    cust_length = len(lats)
    if depot_lats is None:
        depot_lats, depot_lons = lats, lons
    depot_length = len(depot_lats)
    distance_matrix = np.zeros((depot_length, cust_length))
    for i in range(depot_length):
        for j in range(cust_length):  # range(i + 1, cust_length):
            distance_matrix[i, j] = np.ceil(haversine(depot_lons[i], depot_lats[i], lons[j], lats[j]) / 100)
            # distance_matrix[j, i] = distance_matrix[i, j]
    return distance_matrix


def convert_distance_to_web_mercator(lon1, lat1, distance):
    """
    将基于haversine计算得到的距离转换为Web Mercator坐标系中的距离。

    lon1, lat1: 用作参考点的经纬度坐标（度数）
    distance: 与参考点的距离（米）
    """
    # 创建WGS 84地理坐标系（EPSG:4326）和Web Mercator投影坐标系（EPSG:3857）的对象
    geodetic = Proj(init='epsg:4326')
    mercator = Proj(init='epsg:3857')
    # 将参考点的地理坐标转换为Web Mercator坐标
    x1, y1 = transform(geodetic, mercator, lon1, lat1)
    # 在参考点东面distance米的点的地理坐标
    lon2, lat2 = transform(mercator, geodetic, x1 + distance, y1)
    # 计算两个点之间在地理坐标系中的距离，作为转换后的距离
    web_mercator_distance = haversine(lon1, lat1, lon2, lat2)

    return web_mercator_distance


def get_ellipse_config(a, b, radius, foci_distance):
    foci_distance = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    major_axis = radius / 2  # 使用radius作为长轴长度
    minor_axis = np.sqrt(max((major_axis ** 2 - (foci_distance / 2) ** 2), 0))  # 计算短轴长度
    center_x = (a[0] + b[0]) / 2
    center_y = (a[1] + b[1]) / 2
    angle = np.arctan2((b[1] - a[1]), (b[0] - a[0])) * (180 / np.pi)
    return center_x, center_y, major_axis, minor_axis, angle


def map_visualize(dis_matrix, facility, radius, ellipse=True):
    print(list(ctx.providers.keys()))
    place = "龙华区, 深圳, 中国"
    # 获取地图数据
    city = ox.geocode_to_gdf(place)
    north, south, east, west = city["bbox_north"].iloc[0], city["bbox_south"].iloc[0], \
        city["bbox_east"].iloc[0], city["bbox_west"].iloc[0]
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")

    # 将道路网络数据转换为GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # 使用contextily添加底图
    # 创建图像和轴
    fig, ax = plt.subplots(figsize=(8, 11))

    # 在ax上绘制路网
    edges.plot(ax=ax, linewidth=0.8, color='black')

    # 计算以Web Mercator为投影的坐标范围
    # west, south, east, north = edges.to_crs(epsg=3857).total_bounds
    edges_wm = edges.to_crs(epsg=3857)
    west, south, east, north = edges_wm.total_bounds

    # 调整ax的显示范围以适应路网
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    # 添加底图
    # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.HOT, crs=edges_wm.crs, zoom='auto', alpha=0.8)
    # ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs=edges_wm.crs, zoom='auto')
    # ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=edges_wm.crs, zoom='auto')
    # ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=edges_wm.crs, zoom='auto', alpha=1)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager, crs=edges_wm.crs, zoom='auto', alpha=1)

    # 隐藏坐标轴
    ax.axis('off')

    # 获取潜在的无人机仓库地址
    depot_tags = {
        # 'landuse': ['brownfield'],  # 'commercial', 'industrial', 'grass',
        'amenity': ['parking']
        # ['school', 'hospital']  # ['parking']  # ['school', 'hospital']  # ['parking', 'school', 'hospital'],
    }
    gdf = ox.features_from_place(place, depot_tags)
    # gdf.info()
    # gdf['area'] = gdf['geometry'].area
    # gdf = gdf[gdf['area'] > 500]
    # gdf['geometry'] = gdf['geometry'].apply(
    #     lambda geom: geom.representative_point() if geom.type != 'Point' else geom)
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.representative_point() if geom.type != 'Point' else geom)
    points_depots = gdf[gdf['geometry'].type == 'Point']
    depot_nearest_node = ox.distance.nearest_nodes(G, points_depots['geometry'].x, points_depots['geometry'].y)
    depot_node_lons = [G.nodes[node]['x'] for node in depot_nearest_node]
    depot_node_lats = [G.nodes[node]['y'] for node in depot_nearest_node]
    depot_df_nearest_nodes = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(depot_node_lons, depot_node_lats)],
                                              crs='EPSG:4326')
    depot_df_nearest_nodes = depot_df_nearest_nodes.to_crs(epsg=3857)
    ax.scatter(depot_df_nearest_nodes.geometry.x, depot_df_nearest_nodes.geometry.y, c='green', s=10, marker='s',
               zorder=4)

    # 获取POI信息
    tags = {
        'amenity': True,
        'shop': False,
        'leisure': False
    }
    pois = ox.features_from_place(place, tags)
    pois.info()
    pois['geometry'] = pois['geometry'].apply(
        lambda geom: geom.representative_point() if geom.type != 'Point' else geom)
    # 提取POI的坐标
    points_all = pois[pois['geometry'].type == 'Point']

    n_sample_num = len(points_all)  # 100  # 例如，随机取10个点
    # 如果points中的点少于你想抽取的数量，则抽取全部点
    if len(points_all) < n_sample_num:
        points = points_all
    else:
        points = points_all.sample(n=n_sample_num, random_state=666)
    # 转换POI坐标系
    # points = points.to_crs(epsg=3857)

    # 获取最近道路节点的坐标
    nearest_node = ox.distance.nearest_nodes(G, points['geometry'].x, points['geometry'].y)
    node_lons = [G.nodes[node]['x'] for node in nearest_node]
    node_lats = [G.nodes[node]['y'] for node in nearest_node]

    # 将nearest_node中的点标记在图上，这里选择红色标记
    # ax.scatter(node_lons, node_lats, c='red', s=10, zorder=3, transform=ax.transData)  # s参数是点的大小，zorder是层级，确保点在最上面显示

    # 在绘制之前，应该将nearest_node的坐标转换为Web Mercator坐标系，以匹配底图的坐标系
    # 首先将坐标转换为GeoDataFrame
    df_nearest_nodes = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(node_lons, node_lats)], crs='EPSG:4326')
    # 然后转换坐标系为Web Mercator
    df_nearest_nodes = df_nearest_nodes.to_crs(epsg=3857)

    # 在转换后的坐标系上绘制nearest_node的位置
    ax.scatter(df_nearest_nodes.geometry.x, df_nearest_nodes.geometry.y, c='black', s=10, zorder=3)

    facility_nodes = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(np.array(depot_node_lons)[facility],
                                                                            np.array(depot_node_lats)[facility])],
                                      crs='EPSG:4326')
    facility_nodes = facility_nodes.to_crs(epsg=3857)
    ax.scatter(facility_nodes.geometry.x, facility_nodes.geometry.y, c='red', marker='s', zorder=4)  # s=10, )
    # 画椭圆
    if ellipse is True:
        for i in range(len(facility)):
            for j in range(len(facility)):
                if i < j:
                    lon1 = (depot_node_lons[facility[i]] + depot_node_lons[facility[j]]) / 2
                    lat1 = (depot_node_lats[facility[i]] + depot_node_lats[facility[j]]) / 2
                    # lon1 = min(node_lons[facility[i]], node_lons[facility[j]])
                    # lat1 = min(node_lats[facility[i]], node_lats[facility[j]])
                    radius_m = convert_distance_to_web_mercator(lon1, lat1, (radius * 100 + 2000))
                    foci_distance = convert_distance_to_web_mercator(lon1, lat1,
                                                                     dis_matrix[facility[i], facility[j]] * 100)
                    center_x, center_y, major_axis, minor_axis, angle = get_ellipse_config(
                        (facility_nodes.geometry.iloc[i].x, facility_nodes.geometry.iloc[i].y),
                        (facility_nodes.geometry.iloc[j].x, facility_nodes.geometry.iloc[j].y), radius_m,
                        foci_distance)
                    if minor_axis > 0:
                        ellipse_patch = plt.matplotlib.patches.Ellipse((center_x, center_y), 2 * major_axis,
                                                                       2 * minor_axis,
                                                                       angle=angle, edgecolor='blue', facecolor='none',
                                                                       zorder=5)
                        ax.add_patch(ellipse_patch)
                        # ax.scatter([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='red', marker='s')
        # 展示图表
        plt.show()
        # 保存图表
        fig.savefig(f'longhua_ellipse.png', dpi=600)
    else:  # 画圆
        for i in range(len(facility)):
            lon1 = depot_node_lons[facility[i]]
            lat1 = depot_node_lats[facility[i]]
            radius_m = convert_distance_to_web_mercator(lon1, lat1, (radius * 100 + 1000))  # 2-center 1000  center 600
            circle_patch = plt.matplotlib.patches.Circle(
                (facility_nodes.geometry.iloc[i].x, facility_nodes.geometry.iloc[i].y), radius_m, color='blue',
                fill=False, zorder=5)
            ax.add_patch(circle_patch)
        # 展示图表
        plt.show()
        # 保存图表
        fig.savefig(f'longhua_circle.png', dpi=600)


if __name__ == '__main__':
    print(list(ctx.providers.keys()))
    place = "龙华区, 深圳, 中国"
    # 获取地图数据
    city = ox.geocode_to_gdf(place)
    # 转换为平方米，以下使用EPSG:3395(WGS 84 / World Mercator)作为投影参考系进行面积计算
    total_area_sq_meters = city.to_crs('EPSG:3395').geometry.area.sum() / 1e6
    print("总面积：", total_area_sq_meters)
    north, south, east, west = city["bbox_north"].iloc[0], city["bbox_south"].iloc[0], \
        city["bbox_east"].iloc[0], city["bbox_west"].iloc[0]
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive")

    # 将道路网络数据转换为GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # 使用contextily添加底图
    # 创建图像和轴
    fig, ax = plt.subplots(figsize=(12, 12))

    # 在ax上绘制路网
    edges.plot(ax=ax, linewidth=0.8, color='black')

    # 计算以Web Mercator为投影的坐标范围
    # west, south, east, north = edges.to_crs(epsg=3857).total_bounds
    edges_wm = edges.to_crs(epsg=3857)
    west, south, east, north = edges_wm.total_bounds

    # 调整ax的显示范围以适应路网
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)

    # 添加底图
    # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.HOT, crs=edges_wm.crs, zoom='auto', alpha=0.8)
    # ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs=edges_wm.crs, zoom='auto')
    # ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=edges_wm.crs, zoom='auto')
    # ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=edges_wm.crs, zoom='auto', alpha=1)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager, crs=edges_wm.crs, zoom='auto', alpha=1)

    # 隐藏坐标轴
    ax.axis('off')

    # 获取潜在的无人机仓库地址
    depot_tags = {
        # 'landuse': ['brownfield'],  # 'commercial', 'industrial', 'grass',
        'amenity': ['parking']  # ['school', 'hospital']  # ['parking', 'school', 'hospital'],
    }
    gdf = ox.features_from_place(place, depot_tags)
    # gdf.info()
    # gdf['area'] = gdf['geometry'].area
    # gdf = gdf[gdf['area'] > 500]
    # gdf['geometry'] = gdf['geometry'].apply(
    #     lambda geom: geom.representative_point() if geom.type != 'Point' else geom)
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.representative_point() if geom.type != 'Point' else geom)
    points_depots = gdf[gdf['geometry'].type == 'Point']
    depot_nearest_node = ox.distance.nearest_nodes(G, points_depots['geometry'].x, points_depots['geometry'].y)
    depot_node_lons = [G.nodes[node]['x'] for node in depot_nearest_node]
    depot_node_lats = [G.nodes[node]['y'] for node in depot_nearest_node]
    depot_df_nearest_nodes = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(depot_node_lons, depot_node_lats)],
                                              crs='EPSG:4326')
    depot_df_nearest_nodes = depot_df_nearest_nodes.to_crs(epsg=3857)
    ax.scatter(depot_df_nearest_nodes.geometry.x, depot_df_nearest_nodes.geometry.y, c='green', s=10, marker='s',
               zorder=4)

    # 获取POI信息
    tags = {
        'amenity': True,
        'shop': False,
        'leisure': False
    }
    pois = ox.features_from_place(place, tags)
    pois.info()
    pois['geometry'] = pois['geometry'].apply(
        lambda geom: geom.representative_point() if geom.type != 'Point' else geom)
    # 提取POI的坐标
    points_all = pois[pois['geometry'].type == 'Point']

    n_sample_num = len(points_all)  # 100  # 例如，随机取10个点
    # 如果points中的点少于你想抽取的数量，则抽取全部点
    if len(points_all) < n_sample_num:
        points = points_all
    else:
        points = points_all.sample(n=n_sample_num, random_state=666)
    # 转换POI坐标系
    # points = points.to_crs(epsg=3857)

    # 获取最近道路节点的坐标
    nearest_node = ox.distance.nearest_nodes(G, points['geometry'].x, points['geometry'].y)
    node_lons = [G.nodes[node]['x'] for node in nearest_node]
    node_lats = [G.nodes[node]['y'] for node in nearest_node]

    # 将nearest_node中的点标记在图上，这里选择红色标记
    # ax.scatter(node_lons, node_lats, c='red', s=10, zorder=3, transform=ax.transData)  # s参数是点的大小，zorder是层级，确保点在最上面显示

    # 在绘制之前，应该将nearest_node的坐标转换为Web Mercator坐标系，以匹配底图的坐标系
    # 首先将坐标转换为GeoDataFrame
    df_nearest_nodes = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(node_lons, node_lats)], crs='EPSG:4326')
    # 然后转换坐标系为Web Mercator
    df_nearest_nodes = df_nearest_nodes.to_crs(epsg=3857)

    # 在转换后的坐标系上绘制nearest_node的位置
    ax.scatter(df_nearest_nodes.geometry.x, df_nearest_nodes.geometry.y, c='black', s=10, zorder=3)

    # 展示图表
    plt.show()

    # 保存图表
    fig.savefig(f'longhua_{n_sample_num}.png', dpi=600)

    path = f'longhua.npy'

    # nearest_node = nearest_node + depot_df_nearest_nodes  # 合并需求点和depot
    # 计算最近道路节点之间的地图距离
    # map_distances = compute_optimized_map_distances(G, nearest_node, depot_df_nearest_nodes)
    # np.save(f"map_{n_sample_num}_" + path, map_distances)
    # 计算最近道路节点之间的直线距离
    euclidean_distances = compute_optimized_euclidean_distances(node_lats, node_lons, depot_node_lats, depot_node_lons)
    np.save(f"euclidean_{n_sample_num}_" + path, euclidean_distances)

    cdf = 100
