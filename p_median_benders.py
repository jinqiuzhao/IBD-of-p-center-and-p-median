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


def build_MP(int_sol=False):
    MP = Model("master problem")
    z = MP.addVars(Cus_n, lb=0, vtype=GRB.CONTINUOUS, name="z")
    y = MP.addVars(Fac_n, lb=0, ub=1, vtype=GRB.BINARY, name="y")
    MP.setObjective(z.sum(), GRB.MINIMIZE)
    MP.addConstr(quicksum(y[i] for i in range(Fac_n)) == Fac_L)
    # self.MP.Params.MIPForce = 2
    # self.MP.setParam('MIPGap', 0.01)
    MP.update()
    MP._vars = MP.getVars()
    return MP


def solve_MP(MP, callback=None, print_sol=False):
    if callback is not None:
        MP.optimize(callback)
    else:
        MP.optimize()
    y_val = np.zeros(Fac_n)
    z_val = np.zeros(Cus_n)
    for i in range(Fac_n):
        y_val[i] = MP.getVarByName(f"y[{i}]").x
    for j in range(Cus_n):
        z_val[j] = MP.getVarByName(f"z[{j}]").x
    facility = np.argpartition(y_val, -Fac_L)[-Fac_L:]
    if print_sol:
        print("facility:", facility)
    return y_val, z_val, facility, MP.ObjVal


def Benders_Decomposition(MP, eps=0.00001):
    t_initial = time.time()
    UB = np.inf
    LB = 0.1
    Gap = np.inf
    eps = eps
    benders_iter = 1
    no_change_cnt = 1
    opt_sol_time = 0
    # MP = build_MP()
    # self.MP.setParam('MIPGap', 0.1)
    MP_relax = MP.relax()
    MP_relax.setParam('OutputFlag', 0)
    # MP_relax.setParam('Method', 3)
    y_val, z_val, facility, LB = solve_MP(MP_relax)
    MP.setParam('OutputFlag', 0)
    low_bound = 0

    """ Main loop of Benders Decomposition """
    print('\n\n ============================================')
    print(' Benders Decomposition Starts ')
    print('============================================')
    while abs(UB - LB) / (LB + 0.0001) > eps:
        # print(f'Iter: {benders_iter} iteration facility: {facility}')
        # print(len(facility))
        t_start = time.time()
        print(f'Iter: {benders_iter} iteration SP time cost: time = {time.time() - t_start}')
        ub_obj = add_benders_cut(MP_relax, y_val, z_val, count=False)
        add_benders_cut(MP, y_val, z_val)
        if UB > ub_obj:
            UB = ub_obj
            no_change_cnt = 1
        t_start = time.time()
        y_val, z_val, facility, LB = solve_MP(MP_relax)
        # print(y_val)
        print(f'Iter: {benders_iter}, MP time cost: time = {time.time() - t_start}')
        Gap = round(100 * (UB - LB) / (LB + 0.0001), 4)
        print(' %7.2f ' % UB, end='')
        print(' %7.2f ' % LB, end='')
        print(' %8.4f ' % Gap, end='%')
        print(f'    current time cost: time = {time.time() - t_initial}')
        print()
        print()
        # assert self.Gap >= -1e-5
        benders_iter += 1
        no_change_cnt += 1
        # if self.benders_iter == 62:
        #     break
        if time.time() - t_start >= 3600:
            break
    add_benders_cut(MP, y_val, z_val)
    print('\n\n ============================================')
    print(' Benders Decomposition ends ')
    print('============================================')
    y_ij = []
    print('Obj:', UB)
    print(f'total time cost: time = {time.time() - t_initial}')
    return UB


def build_original_MIP():
    org_model = Model('origin model')
    z = org_model.addMVar(1, lb=0, vtype=GRB.CONTINUOUS, name="z")
    y = org_model.addMVar(Fac_n, vtype=GRB.BINARY, name="y")
    x = {}
    for i in range(Fac_n):
        x[i] = org_model.addMVar(Cus_n, lb=0, vtype=GRB.CONTINUOUS,
                                 name='x' + str(i))
    org_model.setObjective(z, GRB.MINIMIZE)
    lhs = z - quicksum(Dis_m[i, :] @ x[i] for i in range(Fac_n))
    org_model.addConstr(lhs >= 0)
    org_model.addConstr(y.sum() == Fac_L)
    for j in range(Cus_n):
        org_model.addConstr(quicksum(x[i][j] for i in range(Fac_n)) == 1)
    for i in range(Fac_n):
        cof = np.zeros((Fac_n, Cus_n))
        cof[:, i] = 1
        org_model.addConstr(x[i] <= cof @ y)
    org_model.update()
    # org_model._vars = y  # .tolist()
    # self.org_model.Params.PreCrush = 1
    # org_model.Params.lazyConstraints = 1
    # org_model.Params.Heuristics = 0.001
    return org_model


def add_benders_cut(MP, y_val, z_val, cb=False, cbcut=False, count=True):
    y = [MP.getVarByName(f"y[{i}]") for i in range(Fac_n)]
    z = [MP.getVarByName(f"z[{j}]") for j in range(Cus_n)]
    obj = 0
    for j in range(Cus_n):
        obj_j = 0
        c_dis = Dis_m[:, j].squeeze()
        # sort_index = np.argsort(c_dis)
        sort_index = Sort_dis_index[:, j]
        sum_y = y_val[sort_index[0]]
        k = 0
        while sum_y < 1 and k < len(sort_index):
            k += 1
            sum_y += y_val[sort_index[k]]
        lhs = LinExpr()
        obj_j += c_dis[sort_index[k]]
        for m in range(k):
            a = c_dis[sort_index[k]] - c_dis[sort_index[m]]
            if a > 0:
                lhs.addTerms(a, y[sort_index[m]])
                obj_j -= a * y_val[sort_index[m]]
        obj += obj_j
        if k > 0 and Cut_index[j, k-1] == 0 and z_val[j] < c_dis[sort_index[k]]:
            if cb:
                MP.cbLazy(z[j] + lhs >= c_dis[sort_index[k]])
            elif cbcut:
                MP.cbCut(z[j] + lhs >= c_dis[sort_index[k]])
            else:
                MP.addConstr(z[j] + lhs >= c_dis[sort_index[k]])
                if count:
                    Cut_index[j, k - 1] = 1
    if not cb and not cbcut:
        MP.update()
    return obj


def build_SP(j, y_val=None):
    if y_val is None:
        y_val = np.zeros(Fac_n)
    SP = Model(f"subproblem_{j}")
    x = SP.addVars(Fac_n, lb=0, vtype=GRB.CONTINUOUS, name="x")
    SP.setObjective(quicksum(x[i] * Dis_m[i, j] for i in range(Fac_n)), GRB.MINIMIZE)
    SP.addConstr(quicksum(x[i] for i in range(Fac_n)) == 1, name="constr_1")
    for i in range(Fac_n):
        SP.addConstr(x[i] <= y_val[i], name=f"constr_x[{i}]")
    SP.update()
    SP.setParam('OutputFlag', 0)
    return SP


def update_SP(SP, y_val):
    for i in range(Fac_n):
        SP.getConstrByName(f"constr_x[{i}]").RHS = y_val[i]
    SP.update()


def slove_SPs(SPs, y_val, MP):
    y = [MP.getVarByName(f"y[{i}]") for i in range(Fac_n)]
    z = [MP.getVarByName(f"z[{i}]") for i in range(Cus_n)]
    sp_obj = []
    for j in range(Cus_n):
        update_SP(SPs[j], y_val)
        SPs[j].optimize()
        sp_obj.append(SPs[j].ObjVal)
        beta = SPs[j].getConstrByName(f"constr_1").Pi
        mu = np.zeros(Fac_n)
        lhs = LinExpr()
        for i in range(Fac_n):
            mu[i] = SPs[j].getConstrByName(f"constr_x[{i}]").Pi
            lhs.addTerms(mu[i], y[i])
        MP.addConstr(z[j] >= beta + lhs)
    MP.update()
    return sum(sp_obj)


def pure_decomposition(MP, eps=0.00001):
    print('============================================')
    SPs = []
    for j in range(Cus_n):
        SPs.append(build_SP(j))
    UB = np.inf
    LB = 0
    y_val, z_val, facility, lb = solve_MP(MP)
    LB = max(LB, lb)
    ub = slove_SPs(SPs, y_val, MP)
    # ub = add_benders_cut_sep(org_model, y_val, UB)
    UB = min(UB, ub)
    while abs(UB - LB) / (LB + 0.0001) > 0.0001:
        Gap = round(100 * (UB - LB) / (LB + 0.0001), 4)
        print(' %7.2f ' % UB, end='')
        print(' %7.2f ' % LB, end='')
        print(' %8.4f ' % Gap, end='%')
        print()
        fac = np.where(y_val >= 0.99)[0]
        # print(fac)
        # add_benders_cut_2(org_model, UB, fac)
        y_val, z_val, facility, lb = solve_MP(MP)  # , callback=call_back)
        LB = max(LB, lb)
        ub = slove_SPs(SPs, y_val, MP)
        # ub = add_benders_cut_sep(org_model, y_val, UB)
        UB = min(UB, ub)
    print()
    print(facility)
    return UB


def add_benders_cut_2(MP, obj, fac):
    z = [MP.getVarByName(f"z[{i}]") for i in range(Cus_n)]
    y = [MP.getVarByName(f"y[{i}]") for i in range(Fac_n)]
    y_sum = quicksum(y[i] for i in range(Fac_n) if i not in fac)
    z_sum = quicksum(z[i] for i in range(Cus_n))
    int_cut = z_sum - obj * (1 - y_sum)
    MP.addConstr(int_cut >= 0)
    MP.update()
    return int_cut


def call_back(model, where):
    if where == GRB.callback.MIPSOL:
        var = np.array(model.cbGetSolution(model._vars))
        z_val = var[: Cus_n]
        y_val = var[Cus_n:]
        # fac = np.where(y_val >= 0.99)[0]
        # print(fac)
        UB = add_benders_cut(model, y_val, z_val, cb=True)


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


def solve_MIP():
    # solve MIP model
    orig_model = build_original_MIP()  # .relax()
    opt_time = time.time()
    orig_model.optimize()
    UB = orig_model.ObjVal
    return UB, opt_time


def Benders_solve():
    LB = 0
    UB = np.inf
    org_model = build_MP(int_sol=True)
    opt_time = time.time()

    # org_model.setParam('OutputFlag', 0)
    # UB = pure_decomposition(org_model)
    LB = Benders_Decomposition(org_model)
    # org_model.Params.PoolGap = 0.1
    # org_model.setParam('PreSolve', 2)
    org_model.setParam('OutputFlag', 0)
    org_model.setParam('LazyConstraints', 1)
    # org_model.setParam('MIPFocus', 2)
    # org_model.setParam('PreCrush', 1)
    # org_model.setParam('CutPasses', 3)
    y_val, z_val, facility, lb = solve_MP(org_model, callback=call_back)
    ub = add_benders_cut(org_model, y_val, z_val)
    LB = lb
    UB = min(UB, ub)
    inter = 0
    while abs(UB - LB) / (LB + 0.0001) > 0.0001:
        Gap = round(100 * (UB - LB) / (LB + 0.0001), 4)
        print(' %7.2f ' % UB, end='')
        print(' %7.2f ' % LB, end='')
        print(' %8.4f ' % Gap, end='%')
        print()
        fac = np.where(y_val >= 0.99)[0]
        # add_benders_cut_2(org_model, UB, fac)
        y_val, z_val, facility, lb = solve_MP(org_model, callback=call_back)
        ub = add_benders_cut(org_model, y_val, z_val)
        LB = lb
        UB = min(UB, ub)
        inter += 1
    return UB, LB, opt_time, inter


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
    # data_sets = ["Manhattan", "chengdu", "Portland", "beijing"]

    for i in data_sets:
        instance = load_data(data_type, data_set=i)
        Dis_m = instance.dis_matrix
        Cus_n = instance.customer_num
        Fac_n = instance.facility_num
        Cut_index = np.zeros((Cus_n, Fac_n), dtype=int)
        Sort_dis_index = np.argsort(Dis_m, axis=0)
        # Sqs_index = cal_sqs_info(Dis_m, Sort_dis_index)
        t_initial = time.time()

        # Solve the original MIP model
        # UB, opt_time = solve_MIP()
        # LB = 0
        # inter = 0

        # IBD algorithm
        UB, LB, opt_time, inter = Benders_solve()

        result.append(UB)
        time_spend.append(time.time() - opt_time)
        print('============================================')
        print('Instance ', i)
        print('Optimal objective: ', UB)
        print("opt_time:", time.time() - opt_time)
        print(f'whole time cost: time = {time.time() - t_initial}')
        df.loc[len(df.index)] = [i, UB, time.time() - opt_time, time.time() - t_initial, inter]

    print("Result:", result)
    print("time_spend", time_spend)
    df.to_csv("pemd_gurobi_result.csv")
    print(df)
    print(df["opt_time"].sum())
