import copy
import os
from data import Loc_Data
import numpy as np
import pandas as pd
from gurobipy import *
import time
from ultis import get_UB1, get_UB2

# Global variable definition
Dis_m = None  # distance matrix
Cus_n = 0  # the number of customers
Fac_n = 0  # the number of candidate facilities
Fac_L = 0  # the number of facilities selected
Sort_dis = None  # sorted distance
Sort_dis_index = None  # the sorted distance index
Sqs_index = None  # sorting result
Cut_index = None  # cuts recorder
Cut_pool = []  # cuts set


def solve_P3(S_set, a_cof, y_val=None):
    model = Model("P3 problem")
    y = model.addVars(Fac_n, lb=0, ub=1, vtype=GRB.BINARY, name="y")
    # z = model.addVars(len(S_set), lb=0, ub=1, vtype=GRB.BINARY, name="z")
    z = model.addVars(len(S_set), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z")

    model.setObjective(quicksum(Sort_dis[int(S_set[k])] * z[k] for k in range(len(S_set))), GRB.MINIMIZE)
    model.addConstrs(quicksum(a_cof[i, j, k] * y[j] for j in range(Fac_n)) >= z[k] for i in range(Cus_n) for k in range(len(S_set)))
    model.addConstr(quicksum(y[j] for j in range(Fac_n)) <= Fac_L)
    model.addConstr(quicksum(z[k] for k in range(len(S_set))) == 1)

    if y_val is not None:
        for i in range(Fac_n):
            model.getVarByName(f"y[{i}]").setAttr(GRB.Attr.Start, y_val[i])

    model.update()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 3600)
    # model.setParam('PreSolve', 2)
    # model.setParam('MIPFocus', 2)
    model.optimize()
    if model.Status == 2:
        obj = model.ObjVal
        # z = [model.getVarByName(f"z[{k}]").x for k in range(len(S_set))]
        # y = [model.getVarByName(f"y[{i}]").x for i in range(Fac_n)]
        # print(z)
    else:
        obj = np.inf
    return obj  # , y


def Bound_solve():
    opt_time = time.time()

    # UB1, _ = get_UB1(Dis_m, Fac_L)
    UB, UB1, f_set = get_UB2(Dis_m, Fac_L)
    obj = np.max(np.min(Dis_m[:, list(f_set)], axis=1))
    UB = min(UB, obj)
    LB = np.ceil(UB1/2)
    print(LB, UB)
    UB_index = np.where(Sort_dis == UB)[0][0]
    LB_index = np.abs(Sort_dis - LB).argmin()
    inter = 0
    while abs(Sort_dis[UB_index] - Sort_dis[LB_index]) / (Sort_dis[LB_index] + 0.001) > 0.01:
    # while UB_index - LB_index >= 1:
        inter += 1
        # y_var_1 = None
        # y_var_2 = None
        a_index = int(np.floor((UB_index + LB_index) / 2))
        b_index = int(UB_index - 1)
        S_set = [a_index, b_index]
        a_cof = np.zeros([Cus_n, Fac_n, len(S_set)])
        for k in range(len(S_set)):
            a_cof[:, :, k] = (Dis_m <= Sort_dis[S_set[k]])
        # obj = solve_P3(S_set, a_cof)
        obj1 = solve_P3([a_index], a_cof[:, :, [0]])  #, y_var_1)
        obj2 = solve_P3([b_index], a_cof[:, :, [1]])  #, y_var_2)
        obj = min(obj1, obj2)
        # print(obj)
        print(obj1, obj2)
        if obj == Sort_dis[a_index]:
            UB_index = a_index
        elif obj == Sort_dis[b_index]:
            LB_index = a_index + 1
            UB_index = b_index
        else:
            LB_index = b_index + 1
        if time.time() - opt_time >= 3600:
            break

    UB = Sort_dis[UB_index]
    LB = Sort_dis[LB_index]
    # UB = min(UB, LB)
    return UB, LB, opt_time, inter


def load_data(data_type, data_set=None, fac_n=None):
    """
    data_type:
        1: p-med dataset，data_set in range(1, 41)
        2: tsp dataset，data_set in ["pcb1173"，"u1817"] is dataset name
        3: square dataset，data_set in [10, 20, 30, 40, 50] is the grid siz
        4：rectangular dataset，data_set in [10, 20, 30, 40, 50] is the grid size
        5：ring_radial dataset，data_set in [10, 20, 30, 40, 50] is the grid size
        6：city map dataset，data_set in ["Portland", "Manhattan", "beijing", "chengdu"] is the city name
    :return:
        instance:
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
        data_path = f"data/tsp/{data_set}"  # u1060 rl1304，u1817"   # "pcb1173
        instance = Loc_Data()
        instance.read_tsp(data_path)
        if fac_n is not None:
            Fac_L = fac_n
        else:
            Fac_L = 5  # 5
    elif data_type == 3:
        node_num, p = 10000, 5
        instance = Loc_Data(cus_num=node_num, f_num=node_num, p=p)
        instance.generate_square_instance(data_set, data_set)
        if fac_n is not None:
            Fac_L = fac_n
        else:
            Fac_L = instance.f_limit  # i
    elif data_type == 4:
        node_num, p = 10000, 5
        instance = Loc_Data(cus_num=node_num, f_num=node_num, p=p)
        instance.generate_rectangular_instance(data_set, data_set)
        Fac_L = instance.f_limit  # i
    elif data_type == 5:
        node_num, p = 10000, 5
        instance = Loc_Data(cus_num=node_num, f_num=node_num, p=p)
        instance.generate_ring_radial_instance(data_set, data_set)
        if fac_n is not None:
            Fac_L = fac_n
        else:
            Fac_L = instance.f_limit  # i
    else:
        assert data_type == 6
        print(f"\n\n city_instance_{data_set}")
        node_num, p = 10000, 5
        instance = Loc_Data(cus_num=node_num, f_num=node_num, p=p)
        data_path = f"data/{data_set}.npy"  # Portland, Manhattan, beijing, chengdu
        instance.read_city_data(data_path)
        if fac_n is not None:
            Fac_L = fac_n
        else:
            Fac_L = instance.f_limit
    return instance


if __name__ == "__main__":
    df = pd.DataFrame(columns=["pmed No.", "Optima", "LB", "opt_time", "total_time", "Iter_num"])
    result = []
    iter_num = []
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
    # data_sets = ["fnl4461", "rl5934", "pla7397", "rl11849", "usa13509", "brd14051", "xray14012_1", "d18512", "pla33810"]
    # ["rat575", "dsj1000", "pcb1173", "u1432", "u1817", "pcb3038", "fnl4461", "rl5934", "pla7397", "rl11849", "usa13509",
    #  "brd14051", "xray14012_1", "d18512", "pla33810"]
    # data_sets = [10, 20, 30, 40, 50]
    # data_sets = ["beijing"]  # ["Manhattan", "chengdu", "Portland", "beijing"]
    fac_number = [5]

    for i in data_sets:
        for f_n in fac_number:
            instance = load_data(data_type, data_set=i, fac_n=f_n)
            t_initial = time.time()
            Dis_m = instance.dis_matrix
            std = np.std(Dis_m)
            mean = np.mean(Dis_m)
            Cus_n = instance.customer_num
            Fac_n = instance.facility_num
            Cut_index = np.zeros((Cus_n, Fac_n), dtype=int)
            Sort_dis = np.sort(np.unique(Dis_m.flatten()))
            # Sort_dis_index = np.argsort(Dis_m, axis=0)
            # Sqs_index = cal_sqs_info(Dis_m, Sort_dis_index)
            # t_initial = time.time()

            # Solve the tightest known bound
            min_index, max_index, LB = 0, len(Sort_dis)-1, np.inf
            while max_index - min_index > 1:
                mid = int(np.ceil((min_index+max_index)/2))
                S_set = [mid]
                a_cof = np.zeros([Cus_n, Fac_n, len(S_set)])
                for k in range(len(S_set)):
                    a_cof[:, :, k] = (Dis_m <= Sort_dis[S_set[k]])
                # obj = solve_P3(S_set, a_cof)
                obj = solve_P3(S_set, a_cof[:, :, [0]])  # , y_var_1)
                print(min_index, max_index, obj)
                if obj == np.inf:
                    min_index = mid
                else:
                    max_index = mid
                    LP = obj
            inter = 0
            opt_time = 0
            UB = LP

            # IBD algorithm
            # UB, LB, opt_time, inter = Bound_solve()

            result.append(UB)
            iter_num.append(inter)
            time_spend.append(time.time() - opt_time)
            print('============================================')
            print('Instance ', i)
            print('Optimal objective: ', UB)
            # print('Optimal Location', facility)
            print("opt_time:", time.time() - opt_time)
            print(f'whole time cost: time = {time.time() - t_initial}')
            df.loc[len(df.index)] = [i, UB, LB, time.time() - opt_time, time.time() - t_initial, inter]

    print("Result:", result)
    print("time_spend", time_spend)
    df.to_csv("pemd_pcenter_result.csv")
    print(df)