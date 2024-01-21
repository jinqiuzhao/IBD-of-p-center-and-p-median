import copy
import math

import numpy as np
import re
import os
import tsplib95


class Loc_Data:

    def __init__(self, cus_num=100, f_num=20, p=5):
        self.customer_loc = None
        self.facility_loc = None
        self.dis_matrix = None
        self.facility_num = f_num  # 设施潜在位置
        self.f_limit = p  # 需要选择的设施数量
        self.customer_num = cus_num
        self.coordinate = None

    def generate_instance(self, seed=0, fac_same=True):
        path = f'data/random_instance_{self.customer_num}_{self.f_limit}.npy'
        if not os.path.isfile(path):
            np.random.seed(seed)
            self.customer_loc = np.random.uniform(0, self.customer_num/10, [self.customer_num, 2])
            if fac_same:
                # f_index = np.random.choice(range(self.customer_num), self.facility_num)
                # self.facility_loc = self.customer_loc[f_index, :]
                # 编号相同
                self.facility_loc = self.customer_loc
            else:
                self.facility_loc = np.random.uniform(0,  self.customer_num/10, [self.facility_num, 2])
            dis = self.facility_loc[:, None, :] - self.customer_loc[None, :, :]
            self.dis_matrix = np.linalg.norm(dis, axis=-1)
            self.dis_matrix = np.ceil(self.dis_matrix)
            # 保存生成的数据
            data = np.vstack([np.full((1, self.facility_num), self.f_limit), self.dis_matrix])
            np.save(path, data)
        else:
            data = np.load(path)
            self.dis_matrix = data[1:, :]
            self.f_limit = int(data[0, 0])
            self.facility_num = self.dis_matrix.shape[0]
            self.customer_num = self.dis_matrix.shape[1]
        return 0

    def generate_rectangular_instance(self, long, wide):
        self.customer_num = long * wide
        self.facility_num = self.customer_num
        path = f'data/Rectangular_instance_{long}-{wide}.npy'
        # path = f'data/Square_instance_{long}-{wide}.npy'
        if not os.path.isfile(path):
            self.customer_loc = np.zeros((wide, long, 2), dtype=np.int)
            self.dis_matrix = np.ones((self.customer_num, self.customer_num), dtype=np.int) * np.inf
            for i in range(wide):
                for j in range(long):
                    self.customer_loc[i, j] = [i, 2*j]  # Rectangular_instance
                    # self.customer_loc[i, j] = [i, j]  # Square_instance
            self.customer_loc = self.customer_loc.reshape([self.customer_num, 2])
            for i in range(self.customer_num):
                for j in range(self.customer_num):
                    self.dis_matrix[i, j] = np.abs(self.customer_loc[i, 0] - self.customer_loc[j, 0]) \
                                            + np.abs(self.customer_loc[i, 1] - self.customer_loc[j, 1])
            # 保存生成的数据
            data = np.vstack([np.full((1, self.facility_num), self.f_limit), self.dis_matrix])
            np.save(path, data)
        else:
            data = np.load(path)
            self.dis_matrix = data[1:, :]
            self.f_limit = int(data[0, 0])
            self.facility_num = self.dis_matrix.shape[0]
            self.customer_num = self.dis_matrix.shape[1]
        return 0

    def generate_square_instance(self, long, wide):
        self.customer_num = long * wide
        self.facility_num = self.customer_num
        # path = f'data/Rectangular_instance_{long}-{wide}.npy'
        path = f'data/Square_instance_{long}-{wide}.npy'
        if not os.path.isfile(path):
            self.customer_loc = np.zeros((wide, long, 2), dtype=np.int)
            self.dis_matrix = np.ones((self.customer_num, self.customer_num), dtype=np.int) * np.inf
            for i in range(wide):
                for j in range(long):
                    # self.customer_loc[i, j] = [i, 2*j]  # Rectangular_instance
                    self.customer_loc[i, j] = [i, j]  # Square_instance
            self.customer_loc = self.customer_loc.reshape([self.customer_num, 2])
            for i in range(self.customer_num):
                for j in range(self.customer_num):
                    self.dis_matrix[i, j] = np.abs(self.customer_loc[i, 0] - self.customer_loc[j, 0]) \
                                            + np.abs(self.customer_loc[i, 1] - self.customer_loc[j, 1])
            # 保存生成的数据
            data = np.vstack([np.full((1, self.facility_num), self.f_limit), self.dis_matrix])
            np.save(path, data)
        else:
            data = np.load(path)
            self.dis_matrix = data[1:, :]
            self.f_limit = int(data[0, 0])
            self.facility_num = self.dis_matrix.shape[0]
            self.customer_num = self.dis_matrix.shape[1]
        return 0

    def generate_ring_radial_instance(self, radial, edge):
        self.customer_num = radial * edge
        self.facility_num = self.customer_num
        path = f'data/Ring_radial_instance_{radial}-{edge}.npy'
        if not os.path.isfile(path):
            self.customer_loc = np.zeros((radial, edge, 2))
            self.dis_matrix = np.ones((self.customer_num, self.customer_num)) * np.inf
            for i in range(radial):
                r = edge/5 + i
                for j in range(edge):
                    self.customer_loc[i, j] = [r * math.cos(2 * math.pi * j / edge), r * math.sin(2 * math.pi * j / edge)]
                    if j > 0:
                        self.dis_matrix[i*edge+j-1, i*edge+j] = np.linalg.norm(self.customer_loc[i, j] - self.customer_loc[i, j-1])
                        self.dis_matrix[i * edge + j, i * edge + j - 1] = self.dis_matrix[i*edge+j-1, i*edge+j]
                    if i > 0:
                        self.dis_matrix[(i-1) * edge + j, i * edge + j] = np.linalg.norm(
                            self.customer_loc[(i-1), j] - self.customer_loc[i, j])
                        self.dis_matrix[i * edge + j, (i - 1) * edge + j] = self.dis_matrix[(i-1) * edge + j, i * edge + j]
                self.dis_matrix[i * edge + 0, i * edge + j] = np.linalg.norm(self.customer_loc[i, j] - self.customer_loc[i, 0])
                self.dis_matrix[i * edge + j, i * edge + 0] = self.dis_matrix[i * edge + 0, i * edge + j]
            self.customer_loc = self.customer_loc.reshape([self.customer_num, 2])
            # self.dis_matrix = np.ceil(self.dis_matrix)
            self.dis_matrix = np.round(self.dis_matrix)
            # 计算完整的最短距离矩阵
            for k in range(self.customer_num):
                for i in range(self.customer_num):
                    for j in range(self.customer_num):
                        self.dis_matrix[i, j] = min(self.dis_matrix[i, j], self.dis_matrix[i, k] + self.dis_matrix[k, j])
            # 保存生成的数据
            data = np.vstack([np.full((1, self.facility_num), self.f_limit), self.dis_matrix])
            np.save(path, data)
        else:
            data = np.load(path)
            self.dis_matrix = data[1:, :]
            self.f_limit = int(data[0, 0])
            self.facility_num = self.dis_matrix.shape[0]
            self.customer_num = self.dis_matrix.shape[1]
        return 0

    def read_pmed(self, path):
        # edges = set()
        if path[-3:] == 'txt':
            f = open(path, 'r')
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line[:-1].strip()  # 去除换行符等
                str = re.split(' ', line)
                assert len(str) == 3
                if count == 0:
                    self.customer_num = int(str[0])
                    self.facility_num = self.customer_num
                    self.f_limit = int(str[-1])
                    self.dis_matrix = np.ones((self.customer_num, self.customer_num)) * np.inf
                    # self.dis_matrix = np.full([self.customer_num, self.customer_num], np.nan)
                    for i in range(self.customer_num):
                        self.dis_matrix[i, i] = 0
                    count += 1
                else:
                    # edges.add((int(str[0])-1, int(str[1])-1))
                    # edges.add((int(str[1])-1, int(str[0])-1))
                    self.dis_matrix[int(str[0])-1, int(str[1])-1] = float(str[2])
                    self.dis_matrix[int(str[1])-1, int(str[0])-1] = float(str[2])
                    count += 1
            # 计算完整的最短距离矩阵
            for k in range(self.customer_num):
                for i in range(self.customer_num):
                    for j in range(self.customer_num):
                        self.dis_matrix[i, j] = min(self.dis_matrix[i, j], self.dis_matrix[i, k] + self.dis_matrix[k, j])
            # 找出每个需求点最远的(self.f_limit - 1)个点索引
            # edge_index = np.argpartition(self.dis_matrix, -(self.f_limit - 1), axis=0)[-(self.f_limit - 1):]
            # for j in range(edge_index.shape[1]):
            #     for i in edge_index[:, j]:
            #         self.dis_matrix[i, j] = np.inf
            new_path = path[:-3] + 'npy'
            data = np.vstack([np.full((1, self.facility_num), self.f_limit), self.dis_matrix])
            np.save(new_path, data)
        else:
            data = np.load(path)
            self.dis_matrix = data[1:, :]
            self.f_limit = int(data[0, 0])
            self.facility_num = self.dis_matrix.shape[0]
            self.customer_num = self.dis_matrix.shape[1]

        return 0

    def read_city_data(self, path):
        self.dis_matrix = np.load(path)
        self.facility_num = self.dis_matrix.shape[0]
        self.customer_num = self.dis_matrix.shape[1]
        return 0

    def read_tsp(self, path='data/tsp/a280', read_file=False):
        path1 = path + '.npy'
        if not os.path.isfile(path1) or read_file:
            path1 = path + '.tsp'
            problem = tsplib95.load(path1)
            # b = list(problem.get_nodes())
            self.customer_num = len(list(problem.get_nodes()))
            self.facility_num = self.customer_num
            self.f_limit = 0
            self.dis_matrix = np.ones((self.customer_num, self.customer_num)) * np.inf
            self.coordinate = []
            for i in range(self.facility_num):
                self.coordinate.append(problem.node_coords[i+1])
                for j in range(self.customer_num):
                    if i == j:
                        self.dis_matrix[i, j] = 0
                    else:
                        s = problem.node_coords[i+1]
                        e = problem.node_coords[j+1]
                        self.dis_matrix[i, j] = tsplib95.distances.euclidean(s, e, round=round)
            new_path = path + '.npy'
            data = np.vstack([np.full((1, self.facility_num), self.f_limit), self.dis_matrix])
            np.save(new_path, data)
        else:
            data = np.load(path1)
            self.dis_matrix = data[1:, :]
            self.f_limit = int(data[0, 0])
            self.facility_num = self.dis_matrix.shape[0]
            self.customer_num = self.dis_matrix.shape[1]



if __name__ == '__main__':
    instance = Loc_Data(100, 20)
    # instance.generate_instance()
    path = f"data/pmed1.txt"
    instance.read_pmed(path)
