import numpy as np
from gurobipy import *

class Test:

    def __init__(self):
        pass

    def test_facility_destroy(self, dis_matrix, facility=None, endurance=0, d_num=1, d_radio=None, instance_name="", seed=0):
        np.random.seed(seed)
        if d_radio is not None:
            d_num = int(np.ceil(d_radio * len(facility)))
        new_facility = np.random.choice(facility, len(facility)-d_num, replace=False)
        delet_facility = np.delete(facility, np.isin(facility, new_facility))
        satisify = np.zeros(dis_matrix.shape[0])
        service_dis = dis_matrix[:, new_facility]
        for j in range(dis_matrix.shape[0]):
            if j in new_facility:
                satisify[j] = 1
            elif 2 * np.min(service_dis[j]) <= endurance:
                satisify[j] = 1
            elif np.min(service_dis[j]) + np.min(dis_matrix[j, delet_facility]) <= endurance:
                satisify[j] = 1
        radio = np.sum(satisify) / len(satisify)
        return radio

    def test_demand_change(self, dis_matrix, facility=None, endurance=0, drop=0.1, add=0.1, instance_name="", seed=0):
        np.random.seed(seed)
        cus_set = np.array([i for i in range(dis_matrix.shape[0]) if i not in facility])
        cus_dis = dis_matrix[:, facility]
        nearest_f = np.argmin(cus_dis, axis=1)
        unique_values, counts = np.unique(nearest_f, return_counts=True)
        drop_num = int(np.ceil(len(cus_set) * drop))
        add_num = int(np.ceil(len(cus_set) * add))
        change_cus = np.random.choice(cus_set, drop_num + add_num, replace=False)
        drop_set = change_cus[:drop_num]
        add_set = change_cus[drop_num:]
        # res_demands = np.zeros_like(counts)
        res_demands = {i: 0 for i in facility}
        for i in drop_set:
            res_demands[facility[nearest_f[i]]] += 1
        # 最大覆盖模型
        cover_model = Model('maxi_cover model')
        index = [(i,j) for i in add_set for j in facility]
        x = cover_model.addVars(index, vtype=GRB.BINARY, name="x")
        y = cover_model.addVars(index, vtype=GRB.BINARY, name="y")
        cover_model.setObjective(x.sum(), GRB.MAXIMIZE)
        for i in add_set:
            cover_model.addConstr(quicksum(x[i, j] for j in facility) <= 1)
            cover_model.addConstr(quicksum(y[i, j] for j in facility) <= 1)
            cover_model.addConstr(quicksum(y[i, j] for j in facility) >= quicksum(x[i, j] for j in facility))
        for j in facility:
            cover_model.addConstr(quicksum(x[i, j] for i in add_set) <= res_demands[j])
        for i in add_set:
            # 同站起降
            # cover_model.addConstr(quicksum(2 * dis_matrix[i, j] * x[i, j] for j in facility) <= endurance)
            # 异站起降
            cover_model.addConstr(quicksum(dis_matrix[i, j] * x[i, j] for j in facility) + quicksum(dis_matrix[i, j] * y[i, j] for j in facility) <= endurance)
        cover_model.update()
        cover_model.setParam('MIPFocus', 2)
        cover_model.optimize()
        cover_num = cover_model.ObjVal
        # radio = (dis_matrix.shape[0] - drop_num + cover_num) / (dis_matrix.shape[0] - drop_num + add_num)
        radio = cover_num / add_num
        return radio


