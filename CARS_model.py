# -*- coding: utf-8 -*-
import copy
import os
import random

from data import Loc_Data
import numpy as np
import pandas as pd
from gurobipy import *
import time

# Global variable definition
Dis_m = None  # distance matrix
Cus_n = 0  # the number of customers
Fac_n = 0  # the number of candidate facilities
Fac_L = 0  # the number of facilities selected
Sort_dis_index = None  # the sorted distance index
Sqs_index = None  # sorting result
Cut_index = None  # cuts recorder


def build_p_median_MIP():
    org_model = Model('p_median')
    y = org_model.addVars(Fac_n, vtype=GRB.BINARY, name="y")
    obj_lhs = LinExpr()
    t = {}
    for j in range(Cus_n):
        K_set = np.unique(Sqs_index[:, j])
        t[j] = org_model.addVars(len(K_set), vtype=GRB.BINARY,
                                 name='t' + str(j))
        org_model.addConstr(t[j].sum() == 1)
        for k in K_set:
            S_jk =  np.where(Sqs_index[:, j] == k)[0]
            i = S_jk[0]
            obj_lhs.addTerms(Dis_m[i, j], t[j][k])
            org_model.addConstr(quicksum(y[i] for i in S_jk) >= t[j][k])

    org_model.setObjective(obj_lhs, GRB.MINIMIZE)
    org_model.addConstr(y.sum() == Fac_L)
    org_model.update()
    # org_model._vars = y  # .tolist()
    # self.org_model.Params.PreCrush = 1
    # org_model.Params.lazyConstraints = 1
    # org_model.Params.Heuristics = 0.001
    return org_model

def build_p_center_MIP():
    org_model = Model('p_center')
    z = org_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="z")
    y = org_model.addVars(Fac_n, vtype=GRB.BINARY, name="y")
    t = {}
    for j in range(Cus_n):
        K_set = np.unique(Sqs_index[:, j])
        t[j] = org_model.addVars(len(K_set), vtype=GRB.BINARY,
                                 name='t' + str(j))
        org_model.addConstr(t[j].sum() == 1)
        obj_lhs = LinExpr()
        for k in K_set:
            S_jk =  np.where(Sqs_index[:, j] == k)[0]
            i = S_jk[0]
            obj_lhs.addTerms(Dis_m[i, j], t[j][k])
            org_model.addConstr(quicksum(y[i] for i in S_jk) >= t[j][k])
        org_model.addConstr(z >= obj_lhs)

    org_model.setObjective(z, GRB.MINIMIZE)
    org_model.addConstr(y.sum() == Fac_L)
    org_model.update()
    # org_model._vars = y  # .tolist()
    # self.org_model.Params.PreCrush = 1
    # org_model.Params.lazyConstraints = 1
    # org_model.Params.Heuristics = 0.001
    return org_model


def cal_sqs_info(dis_m, sort_index):
    """
    Computes an indexed set of equidistant weights
    """
    sqs_info = np.zeros((Fac_n, Cus_n), dtype=int)
    for j in range(dis_m.shape[1]):
        k = 0
        last_dis = copy.copy(dis_m[sort_index[0, j], j])
        for s in sort_index[:, j]:
            if dis_m[s, j] > last_dis:
                k += 1
            sqs_info[s, j] = int(k)
            last_dis = copy.copy(dis_m[s, j])
    return sqs_info


def load_data(data_type, data_set=None):
    """
    data_type:
        1: p-med dataset，data_set in range(1, 41)
        2: tsp dataset，data_set in ["pcb1173"，"u1817"] is dataset name
        3: square dataset，data_set in [10, 20, 30, 40, 50] is the grid siz
        4：rectangular dataset，data_set in [10, 20, 30, 40, 50] is the grid size
        5：ring_radial dataset，data_set in [10, 20, 30, 40, 50] is the grid size
        6：city map dataset，data_set in ["Portland", "Manhattan", "beijing", "chengdu"] is the city name
    :return:
        instance
    """
    global Fac_L
    if data_type == 1:
        print(f"\n\n pmeds{data_set}")
        data_path = f"data/pmed{data_set}.npy"
        if not os.path.exists(data_path):
            data_path = f"data/pmed{data_set}.txt"
        instance = Loc_Data()
        instance.read_pmed(data_path)
        Fac_L = instance.f_limit
    elif data_type == 2:
        data_path = f"data/tsp/{data_set}"    #u1060 rl1304，u1817"   # "pcb1173
        instance = Loc_Data()
        instance.read_tsp(data_path)
        Fac_L = 5
    elif data_type == 3:
        node_num, p = 10000, 5
        instance = Loc_Data(cus_num=node_num, f_num=node_num, p=p)
        instance.generate_square_instance(data_set, data_set)
        Fac_L = instance.f_limit   # i
    elif data_type == 4:
        node_num, p = 10000, 5
        instance = Loc_Data(cus_num=node_num, f_num=node_num, p=p)
        instance.generate_rectangular_instance(data_set, data_set)
        Fac_L = instance.f_limit   # i
    elif data_type == 5:
        node_num, p = 10000, 5
        instance = Loc_Data(cus_num=node_num, f_num=node_num, p=p)
        instance.generate_ring_radial_instance(data_set, data_set)
        Fac_L = instance.f_limit   # i
    else:
        assert data_type == 6
        print(f"\n\n city_instance_{data_set}")
        node_num, p = 10000, 5
        instance = Loc_Data(cus_num=node_num, f_num=node_num, p=p)
        data_path = f"data/{data_set}.npy"  # Portland, Manhattan, beijing, chengdu
        instance.read_city_data(data_path)
        Fac_L = instance.f_limit
    return instance


if __name__ == "__main__":
    df = pd.DataFrame(columns=["pmed No.", "Optima", "opt_time", "total_time", "Iter_num"])
    result = []
    time_spend = []
    """
        data_type:
            1: p-med dataset，data_set in range(1, 41)
            2: tsp dataset，data_set in ["pcb1173"，"u1817"] is dataset name
            3: square dataset，data_set in [10, 20, 30, 40, 50] is the grid siz
            4：rectangular dataset，data_set in [10, 20, 30, 40, 50] is the grid size
            5：ring_radial dataset，data_set in [10, 20, 30, 40, 50] is the grid size
            6：city map dataset，data_set in ["Portland", "Manhattan", "beijing", "chengdu"] is the city name

        """
    data_type = 1
    data_sets = range(1, 41)
    # data_sets = ["rat575","pcb1173", "u1060", "dsj1000"]
    # data_sets = [10, 20, 30, 40, 50]
    # data_sets = ["Portland", "Manhattan", "beijing", "chengdu"]
    # data_sets = ["Manhattan"]

    for i in data_sets:
        instance = load_data(data_type, data_set=i)
        Dis_m = instance.dis_matrix
        Cus_n = instance.customer_num
        Fac_n = instance.facility_num
        Cut_index = np.zeros((Cus_n, Fac_n), dtype=int)
        Sort_dis_index = np.argsort(Dis_m, axis=0)
        Sqs_index = cal_sqs_info(Dis_m, Sort_dis_index)
        t_initial = time.time()

        # p-median or p-center
        # orig_model = build_p_median_MIP()
        orig_model = build_p_center_MIP()

        opt_time = time.time()
        orig_model.optimize()
        UB = orig_model.ObjVal

        result.append(UB)
        time_spend.append(time.time() - opt_time)
        print('============================================')
        print('Instance ', i)
        print('Optimal objective: ', UB)
        # print('Optimal Location', facility)
        print("opt_time:", time.time() - opt_time)
        print(f'whole time cost: time = {time.time() - t_initial}')
        df.loc[len(df.index)] = [i, UB, time.time() - opt_time, time.time() - t_initial, 0]

    print("Result:", result)
    print("time_spend", time_spend)
    df.to_csv("pemd_CARS_result.csv")
    print(df)
