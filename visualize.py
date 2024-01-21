import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


def convert_matrix_to_coord(dis_matrix):
    mds = MDS(n_components=2, metric=True, dissimilarity='precomputed', max_iter=1000, random_state=42, normalized_stress=False)
    points = mds.fit_transform(np.array(dis_matrix))
    return points


def get_ellipse_config(a, b, radius, foci_distance):
    # foci_distance = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    major_axis = radius / 2  # 使用radius作为长轴长度
    minor_axis = np.sqrt(max((major_axis ** 2 - (foci_distance / 2) ** 2), 0))  # 计算短轴长度
    center_x = (a[0] + b[0]) / 2
    center_y = (a[1] + b[1]) / 2
    angle = np.arctan2((b[1] - a[1]), (b[0] - a[0])) * (180 / np.pi)
    return center_x, center_y, major_axis, minor_axis, angle


class Visualize:

    def __init__(self):
        pass

    def plt_ellipse_cover(self, dis_matrix, points=None, facility=None, radius=0, instance_name="", coor_lim=None):
        if points is None:
            points = convert_matrix_to_coord(dis_matrix)

        # plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(points[:, 0], points[:, 1], zorder=1)
        # ax.set_title(f"Visualization of the ellipse cover in instance {instance_name} features a major axis with a maximum length of {int(radius)}.")
        ax.set_title(
            f"Ellipse cover in {instance_name} with {int(radius)}.")
        # 设置坐标轴比例相同
        ax.set_aspect('equal', adjustable='box')
        if coor_lim is not None:
            ax.set_xlim(coor_lim[0])
            ax.set_ylim(coor_lim[1])

        # 绘制椭圆
        # fig, ax = plt.subplots(figsize=(8, 6))
        for i in facility:
            for j in facility:
                if i < j:
                    center_x, center_y, major_axis, minor_axis, angle = get_ellipse_config(points[i], points[j], radius, dis_matrix[i,j])
                    if minor_axis > 0:
                        ellipse = plt.matplotlib.patches.Ellipse((center_x, center_y), 2 * major_axis, 2 * minor_axis,
                                                                 angle=angle, edgecolor='blue', facecolor='none',
                                                                 zorder=0)
                        ax.add_patch(ellipse)
                        ax.scatter([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='red', marker='s')
            #         break
            # break
        # plt.grid(True)
        # for i, txt in enumerate(range(len(points))):
        #     ax.annotate(txt, (points[i, 0], points[i, 1]))
        plt.show()
        a = 1

    def plt_circle_cover(self, dis_matrix, points=None, facility=None, radius=0, instance_name="", ellipse=False, coor_lim=None):
        if points is None:
            points = convert_matrix_to_coord(dis_matrix)

        # plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(points[:, 0], points[:, 1], zorder=1)
        # ax.set_title(f"Visualization of the circle cover in instance {instance_name} features a diameter with a maximum length of {int(2*radius)}.")
        ax.set_title(
            f"Circle cover in {instance_name} with {int(2 * radius)}.")
        # 设置坐标轴比例相同
        ax.set_aspect('equal', adjustable='box')
        if coor_lim is not None:
            ax.set_xlim(coor_lim[0])
            ax.set_ylim(coor_lim[1])
        # 绘制圆
        # fig, ax = plt.subplots(figsize=(8, 6))
        for i in facility:
            circle = plt.matplotlib.patches.Circle(points[i], radius, color='blue', fill=False)
            ax.add_patch(circle)
            # 在圆心添加文本标签
            ax.text(points[i, 0], points[i, 1], f'{i}', ha='center', va='center')
            ax.scatter([points[i, 0]], [points[i, 1]], color='red', marker='s')
            #         break
            # break
        if ellipse:
            for i in facility:
                for j in facility:
                    if i < j:
                        center_x, center_y, major_axis, minor_axis, angle = get_ellipse_config(points[i], points[j],
                                                                                               2 * radius, dis_matrix[i, j])
                        if minor_axis > 0:
                            ellipse = plt.matplotlib.patches.Ellipse((center_x, center_y), 2 * major_axis,
                                                                     2 * minor_axis,
                                                                     angle=angle, edgecolor='green', facecolor='none',
                                                                     linestyle='--', zorder=0)
                            ax.add_patch(ellipse)
                            ax.scatter([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='red',
                                       marker='s')
        # plt.grid(True)
        # for i, txt in enumerate(range(len(points))):
        #     ax.annotate(txt, (points[i, 0], points[i, 1]))
        plt.show()
        a = 1