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
Sort_dis_index = None  # the sorted distance index
Sqs_index = None  # sorting result
Cut_index = None  # cuts recorder
Sort_dis = None
Cut_pool = []  # cuts set
UB, LB = np.inf, 0


def build_MP(int_sol=False):
    MP = Model("master problem")
    w = MP.addVar(lb=0, vtype=GRB.CONTINUOUS, name="w")
    y = MP.addVars(Fac_n, lb=0, ub=1, vtype=GRB.BINARY, name="y")

    MP.setObjective(w, GRB.MINIMIZE)
    MP.addConstr(quicksum(y[i] for i in range(Fac_n)) == Fac_L)

    # Add all cuts
    # for j in range(Cus_n):
    #     sort_index = Sort_dis_index[:, j]
    #     max_f = sort_index[-1]
    #     max_dis = Dis_m[max_f, j]
    #     lhs = quicksum(y[i] * (max_dis - Dis_m[i, j]) for i in sort_index)
    #     MP.addConstr(w >= max_dis - lhs)

    MP.update()
    MP._vars = MP.getVars()
    # MP.setParam('Method', 2)
    return MP


def solve_MP(MP, callback=None, print_sol=False):
    if callback is not None:
        MP.optimize(callback)
    else:
        MP.optimize()
    if MP.isMIP:
        a = 1
        while MP.SolCount < 1 and a < 20:
            print("cannot find feasible solution!")
            MP.setParam('TimeLimit', 30 * a)
            MP.optimize()
            a += 1
        if a > 1:
            MP.setParam('TimeLimit', 10)
        MP.Params.SolutionNumber = 0
        lb = MP.ObjVal
        if MP.Status != 2:
            lb = MP.PoolObjVal
        y_val = np.array(MP.Xn[1:])
        w_val = MP.Xn[0]
    else:
        y_val = np.zeros(Fac_n)
        w_val = MP.getVarByName(f"w").x
        for i in range(Fac_n):
            y_val[i] = MP.getVarByName(f"y[{i}]").x
        lb = MP.ObjVal
    facility = np.argpartition(y_val, -Fac_L)[-Fac_L:]
    # if MP.Status != 2:
    #     lb = MP.BesetBond
    if print_sol:
        print("facility:", facility)
    return y_val, w_val, facility, lb


def Benders_Decomposition(MP, UB1=np.inf, LB1=0, eps=0.01):
    t_initial = time.time()
    int_UB = UB1
    # global LB, UB
    UB = np.inf  # UB1
    LB = LB1
    Gap = round(100 * (UB - LB) / (LB + 0.0001), 4)
    eps = eps
    benders_iter = 1
    no_change_cnt = 1
    opt_sol_time = 0
    MP_relax = MP.relax()
    MP_relax.setParam('OutputFlag', 0)
    # MP_relax.setParam('PreSolve', 1)
    # MP_relax.setParam('Method', 1)
    # MP_relax.setParam('PrePasses', 1)
    # MP_relax.setParam('PreCrush', 1)
    # MP_relax.setParam('SimplexPricing', 3)
    # MP.setParam('Method', 3)
    y_val, z_val, facility, low_bound = solve_MP(MP_relax)  # MP.relax())
    LB = max(LB, low_bound)
    MP.setParam('OutputFlag', 0)
    # low_bound = 0
    ub_obj = 0
    """ Main loop of Benders Decomposition """
    print('\n\n ============================================')
    print(' Benders Decomposition Starts ')
    print('============================================')
    while UB - np.ceil(LB) >= 1 and Gap > eps:  # not (Gap <= eps):
        # print(f'Iter: {benders_iter} iteration facility: {facility}')
        # print(len(facility))
        t_start = time.time()
        # print(f'Iter: {benders_iter} iteration SP time cost: time = {time.time() - t_start}')
        # if LB1 < ub_obj < int_UB: # UB < UB1:  # Gap < 50:   # UB < UB1: # UB1 > low_bound > LB1:  #
        ub_obj, int_obj, constr_num = add_benders_cut(MP_relax, y_val, LB, int_UB, relax_ub=UB, int_sol=False, updata=0,
                                                      con_0=no_change_cnt, MP_int=MP, )
        # else:
        #     ub_obj, int_obj, constr_num = add_benders_cut(MP_relax, y_val, LB, UB, int_sol=False)
        int_UB = min(int_UB, int_obj)
        if UB > ub_obj:
            UB = ub_obj
            # no_change_cnt = 1
        t_start = time.time()
        y_val, z_val, facility, low_bound = solve_MP(MP_relax)
        # print(time.time() - t_start)
        if LB < low_bound:
            LB = low_bound
            # no_change_cnt = 1
        # print(f'Iter: {benders_iter}, MP time cost: time = {time.time() - t_start}')

        Gap = round(100 * (UB - LB) / (LB + 0.0001), 4)

        print(constr_num)
        print(' %7.2f ' % UB, end='')
        print(' %7.2f ' % LB, end='')
        print(' %8.4f ' % Gap, end='%')
        print(f'    current time cost: time = {time.time() - t_initial}')
        print()

        # print()
        # assert self.Gap >= -1e-5
        benders_iter += 1
        # no_change_cnt += 1
        # if no_change_cnt >= 4:
        #     break
        if constr_num <= 0:
            no_change_cnt += 1
        elif constr_num > 0:
            no_change_cnt = 0
        if no_change_cnt >= 2:
            break

        # if Gap < 1.5:
        #     MP_relax.setParam('OutputFlag', 1)
        #     MP_relax.tune()

    global Cut_index
    Cut_index = np.zeros((Cus_n, len(Sort_dis)), dtype=int)
    ub_obj, int_obj, constr_num = add_benders_cut(MP_relax, y_val, LB, int_UB, relax_ub=int_UB, int_sol=False,
                                                  MP_int=MP, updata=3)
    print(constr_num)
    int_UB = min(int_UB, int_obj)
    # MP.update()
    # MP_relax.setParam('OutputFlag', 1)
    # MP_relax.tune()
    print('\n\n ============================================')
    print(' Benders Decomposition ends ')
    print('============================================')
    y_ij = []
    print('Relax_Obj:', UB)
    print('Int_Obj:', int_UB)
    print(f'total time cost: time = {time.time() - t_initial}')
    print(f"LB:{np.ceil(max(LB, LB1))}, UB:{int_UB}")
    # return UB, int_UB
    return np.ceil(max(LB, LB1)), int_UB


def build_original_MIP():
    org_model = Model('origin model')
    z = org_model.addMVar(1, lb=0, vtype=GRB.CONTINUOUS, name="z")
    y = org_model.addMVar(Fac_n, vtype=GRB.BINARY, name="y")
    x = {}
    for i in range(Fac_n):
        x[i] = org_model.addMVar(Cus_n, lb=0, vtype=GRB.BINARY,
                                 name='x' + str(i))
    org_model.setObjective(z, GRB.MINIMIZE)
    for j in range(Cus_n):
        # lhs = z - Dis_m[i, :] @ x[i]
        lhs = z - quicksum(Dis_m[i, j] * x[i][j] for i in range(Fac_n))
        org_model.addConstr(lhs >= 0)
    org_model.addConstr(y.sum() == Fac_L)
    for j in range(Cus_n):
        org_model.addConstr(quicksum(x[i][j] for i in range(Fac_n)) == 1)
    for i in range(Fac_n):
        cof = np.zeros((Fac_n, Cus_n))
        cof[:, i] = 1
        org_model.addConstr(x[i] <= cof @ y)
    org_model.update()
    org_model._vars = y
    # org_model._vars = y  # .tolist()
    # self.org_model.Params.PreCrush = 1
    # org_model.Params.lazyConstraints = 1
    # org_model.Params.Heuristics = 0.001
    return org_model


def add_benders_cut(MP, y_val, lb, ub, relax_ub=np.inf, cb=False, cbcut=False, int_sol=True, updata=0, MP_int=None,
                    con_0=0):
    t_1 = time.time()
    y = [MP.getVarByName(f"y[{i}]") for i in range(Fac_n)]
    w = MP.getVarByName(f"w")
    y_1 = []
    if MP_int is not None:
        y_1 = [MP_int.getVarByName(f"y[{i}]") for i in range(Fac_n)]
        w_1 = MP_int.getVarByName(f"w")

    constr_num = 0
    y_dis = np.take_along_axis(Dis_m, Sort_dis_index, axis=1)
    y_sort = y_val[Sort_dis_index]
    y_cumsum = np.cumsum(y_sort, axis=1)
    k_js = np.argmax(y_cumsum >= 1, axis=1)
    # vir_fac = set(Sort_dis_index[np.arange(len(k_js)), k_js])
    int_obj_j = Dis_m[np.arange(Dis_m.shape[0]), Sort_dis_index[np.arange(Sort_dis_index.shape[0]), k_js]]
    int_obj = np.max(int_obj_j)
    if int_obj <= np.ceil(lb) and int_sol:
        return int_obj, int_obj, constr_num
    k_ths_js = np.searchsorted(Sort_dis, int_obj_j)
    ub_k_js = np.argmax(y_dis >= min(ub, relax_ub), axis=1)
    ub_k_ths_js = np.searchsorted(Sort_dis, min(ub, relax_ub))
    # if not cbcut:
    y_dis[y_dis < np.ceil(lb)] = np.ceil(
        lb)  # This modifies the original problem, but does not affect the integer solution
    a_matrix = int_obj_j[:, np.newaxis] - y_dis
    a_matrix[a_matrix < 0] = 0
    float_obj_j = int_obj_j - np.sum(np.multiply(a_matrix, y_sort), axis=1)
    float_obj = np.max(float_obj_j)
    if not int_sol and not cbcut:
        part_obj_j = float_obj_j[float_obj_j > np.ceil(lb)]
        num = int(min(np.ceil(len(part_obj_j) * 0.1), 10))
        max_ub = np.partition(part_obj_j, -num)[-num:].min()

        part_obj_j = part_obj_j[part_obj_j < min(ub, relax_ub)]
        num = int(min(np.floor(len(part_obj_j) * 0.1), 1))
        if num <= 0:
            cut_ub = min(ub, relax_ub)
        else:
            cut_ub = np.partition(part_obj_j, num)[:num].max()
        cut_set = np.where(((float_obj_j > np.ceil(lb)) & (float_obj_j <= cut_ub)) |
                           # (float_obj_j >= float_obj)
                           (float_obj_j >= max_ub)
                           )[0]
    elif int_sol and not cb:
        part_obj_j = int_obj_j[int_obj_j > np.ceil(lb)]
        num = int(min(np.ceil(len(part_obj_j) * 0.5), 20))
        max_ub = np.partition(part_obj_j, -num)[-num:].min()

        part_obj_j = part_obj_j[part_obj_j < ub]
        num = int(min(np.floor(len(part_obj_j) * 0.5), 20))
        if num > 0:
            cut_ub = np.partition(part_obj_j, num)[:num].max()
        else:
            cut_ub = ub
        cut_set = np.where(((int_obj_j > np.ceil(lb)) & (int_obj_j <= cut_ub)) |
                           (int_obj_j >= max_ub)
                           )[0]
    elif cbcut:
        # cut_ub = float_obj_j[float_obj_j > np.ceil(lb)].min()
        cut_set = np.where(float_obj_j > lb
                           # & (float_obj_j <= cut_ub)) |
                           # (float_obj_j >= float_obj)
                           )[0]
        # cut_index = copy.copy(Cut_index)
    elif cb:
        # lazycut: All unconstrained cases must be excluded
        cut_set = np.where((int_obj_j > lb)
                           )[0]
    else:
        cut_set = range(Cus_n)
    # print("numpy:", time.time() - t_1)
    t_2 = time.time()
    for j in cut_set:
        # if updata:
        #     cut_index = np.where(Cut_index[j, :] != 0)[0]
        #     for k1 in cut_index:
        #         constr = MP.getConstrByName(f"cut_{j}_{k1}")
        #         if constr is not None:
        #             if constr.RHS <= lb:
        #                 MP.remove(constr)
        #                 Cut_index[j, k1] = 0
        #             else:
        #                 break
        #     for k2 in cut_index[::-1]:
        #         constr = MP.getConstrByName(f"cut_{j}_{k2}")
        #         if constr is not None:
        #             if constr.RHS > ub:
        #                 MP.remove(constr)
        #                 Cut_index[j, k2] = 0
        #             else:
        #                 break
        if j == 46:
            adf = 10
        if k_js[j] <= ub_k_js[j]:
            k = k_js[j]
            k_th = k_ths_js[j]
            if k <= 0 and Cut_index[j, k_th] > 0:
                continue
            a_cof = a_matrix[j, :]
            # a_cof[a_cof > (Sort_dis[k_th] - np.ceil(lb))] = Sort_dis[k_th] - np.ceil(lb)
            lhs = quicksum(a_cof[m] * y[Sort_dis_index[j, m]] for m in range(k) if a_cof[m] > 0)
            if MP_int is not None:
                lhs2 = quicksum(a_matrix[j, m] * y_1[Sort_dis_index[j, m]] for m in range(k) if a_cof[m] > 0)
        else:
            k = ub_k_js[j]
            k_th = ub_k_ths_js
            if k <= 0 and Cut_index[j, k_th] > 0:
                continue
            # a = Dis_m[j, Sort_dis_index[j, k]]
            dif = Sort_dis[k_ths_js[j]] - Sort_dis[ub_k_ths_js]
            a_cof = a_matrix[j, :] - dif
            # a_cof[a_cof < (Sort_dis[k_th] - np.ceil(lb))] -= dif
            a_cof[a_cof < 0] = 0
            # sdf = Sort_dis[k_th] - np.ceil(lb)
            # a_cof[a_cof > (Sort_dis[k_th] - np.ceil(lb))] = Sort_dis[k_th] - np.ceil(lb)
            lhs = quicksum(a_cof[m] * y[Sort_dis_index[j, m]] for m in range(k) if a_cof[m] > 0)
            if MP_int is not None:
                lhs2 = quicksum((a_matrix[j, m] - dif) * y_1[Sort_dis_index[j, m]] for m in range(k) if a_cof[m] > 0)

        if Sort_dis[k_th] > np.ceil(lb):  # and (obj == obj_j or obj_j > ub):  # k > 0 and Cut_index[j, k_th] == 0 and
            if cb:
                if Cut_index[j, k_th] == 0:
                    MP.cbLazy(w + lhs >= Sort_dis[k_th])
                    # Cut_index[j, k_th] = 1
                # Cut_index[j, k_th] = 1
                # print(f"customer {j}: {w} + {lhs} >= {c_dis[sort_index[k]]}")
            elif cbcut:
                if Cut_index[j, k_th] == 0:
                    # if Sort_dis[k_th] >= max(float_obj, ub):  # obj_j > lb:   #  and obj >= ub:  # obj_j:
                    MP.cbCut(w + lhs >= Sort_dis[k_th])
                # print(f"customer {j}: {w} + {lhs} >= {int_obj_j[j]}")
            else:
                MP.addConstr(w + lhs >= Sort_dis[k_th], name="cut_" + str(j) + "_" + str(k_th))
                # print(f"customer {j}: {w} + lhs >= {Sort_dis[k_th]}")
                # if MP_int is None:
                constr_num += 1
                Cut_index[j, k_th] = 1
                # print(f"customer {j}: {w} + {lhs} >= {Sort_dis[k_th]}")
                if MP_int is not None and np.ceil(lb) < Sort_dis[k_th] < ub:
                    # if MP_int is not None and np.ceil(lb) < int_obj_j[j] < ub:
                    # max(ub - constr_num, np.ceil(lb) + 1): # max(relax_ub, np.ceil(lb) + 1):
                    # and c_dis[sort_index[k]] > lb:  # and (ub-mp_obj)/(mp_obj+0.0001) <= 100:
                    MP_int.addConstr(w_1 + lhs2 >= Sort_dis[k_th], name="cut_" + str(j) + "_" + str(k_th))
                    # print("add")
                    # Cut_index[j, k_th] = 1
                    # constr_num += 1
                    # print(f"customer {j}: {w} + lhs >= {Sort_dis[k_th]}")
    # print("add_cut:", time.time() - t_2)
    if updata > 0:
        indices = np.transpose(np.where(Cut_index != 0))
        # cut_index = [(x, y) for x, y in indices]
        for a in indices:
            j, k = a[0], a[1]
            constr = MP.getConstrByName(f"cut_{j}_{k}")
            if constr is not None:
                # if updata >= 3:
                #     if constr.RHS <= lb or constr.RHS > ub:
                #         MP.remove(constr)
                #         Cut_index[j, k] = 0
                if updata >= 2:
                    if constr.RHS > ub:
                        MP.remove(constr)
                        Cut_index[j, k] = 0
                # elif updata >= 1:
                if constr.RHS <= lb:
                    MP.remove(constr)
                    Cut_index[j, k] = 0
        MP.update()
    # print(constr_num)
    return float_obj, int_obj, constr_num

    # for j in cut_set:
    #     obj_j = 0
    #     c_dis = Dis_m[:, j].squeeze()
    #     sort_index = Sort_dis_index[j, :]
    #     cum_sum = np.cumsum(y_val[sort_index])
    #     k = np.argmax(cum_sum >= 1)
    #     k_th = np.argmax(np.unique(c_dis[sort_index]) >= c_dis[sort_index[k]])
    #     ub_k = np.argmax(c_dis[sort_index] >= min(ub, relax_ub))
    #     ub_k_th = np.argmax(np.unique(c_dis[sort_index]) >= min(ub, relax_ub))
    #     # a = np.dot(c_dis[sort_index][:k+1], y_val[sort_index][:k+1])
    #     if np.dot(c_dis[sort_index][:k+1], y_val[sort_index][:k+1]) < lb:
    #         obj = max(obj, lb)
    #         continue
    #
    #     sum_y = y_val[sort_index[0]]
    #     k = 0
    #     ub_k = 0
    #     k_th = 0
    #     ub_k_th = len(sort_index) + 1
    #     last_dis = c_dis[sort_index[0]]
    #     while sum_y < 1 and k < len(sort_index):
    #         k += 1
    #         sum_y += y_val[sort_index[k]]
    #         if c_dis[sort_index[k]] > last_dis:
    #             k_th += 1
    #             last_dis = c_dis[sort_index[k]]
    #         if c_dis[sort_index[k]] >= min(ub, relax_ub) and ub_k_th >= len(sort_index) + 1:
    #             ub_k_th = k_th
    #             ub_k = k
    #             # break
    #     if ub_k_th >= len(sort_index) + 1:
    #         ub_k_th = len(sort_index)
    #         ub_k = len(sort_index)
    #
    #     # if (int_sol or cbcut) and k_th > ub_k_th:
    #     #     obj = max(c_dis[sort_index[k]], obj)
    #     #     if np.any(Cut_index[j, ub_k_th:k_th+1]) >= 1:
    #     #         # Cut_index[j, k_th:] = 1
    #     #         continue
    #     #     else:
    #     #         k_th = ub_k_th
    #     #         k = ub_k
    #     if updata:
    #         cut_index = np.where(Cut_index[j, :] != 0)[0]
    #         for k1 in cut_index:
    #             constr = MP.getConstrByName(f"cut_{j}_{k1}")
    #             if constr is not None:
    #                 if constr.RHS < lb:
    #                     MP.remove(constr)
    #                     Cut_index[j, k1] = 0
    #                 else:
    #                     break
    #         for k2 in cut_index[::-1]:
    #             constr = MP.getConstrByName(f"cut_{j}_{k2}")
    #             if constr is not None:
    #                 if constr.RHS > ub:
    #                     MP.remove(constr)
    #                     Cut_index[j, k2] = 0
    #                 else:
    #                     break
    #
    #     lhs = LinExpr()
    #     lhs2 = LinExpr()
    #     obj_j = c_dis[sort_index[k]]
    #     if obj_j <= np.ceil(lb) and int_sol:
    #         obj = max(obj, lb)
    #         continue
    #     for m in range(k):
    #         # obj_j -= (c_dis[sort_index[k]] - c_dis[sort_index[m]]) * y_val[sort_index[m]]
    #         # if c_dis[sort_index[m]] > ub and int_sol:
    #         #     break  # continue
    #         # if not cbcut:
    #         if c_dis[sort_index[m]] < lb:
    #             a = c_dis[sort_index[min(k, ub_k)]] - np.ceil(lb)  # + 1
    #         else:
    #             a = c_dis[sort_index[min(k, ub_k)]] - c_dis[sort_index[m]]
    #         obj_j -= a * y_val[sort_index[m]]
    #         # else:
    #         #     a = c_dis[sort_index[k]] - c_dis[sort_index[m]]
    #         if a > 0:
    #             lhs.addTerms(a, y[sort_index[m]])
    #             if MP_int is not None:
    #                 lhs2.addTerms(a, y_1[sort_index[m]])
    #             # obj_j -= a * y_val[sort_index[m]]
    #     obj = max(obj_j, obj)
    #     min_obj = min(obj_j, obj)
    #     # if not cbcut:
    #     k = min(k, ub_k)
    #     k_th = min(k_th, ub_k_th)
    #
    #     # if obj_j >= obj:
    #     #     sfd=2
    #
    #     if k > 0 and Cut_index[j, k_th] == 0 and c_dis[sort_index[k]] > np.ceil(lb): #  and (obj == obj_j or obj_j > ub):
    #         if cb:
    #             if c_dis[sort_index[k]] >= ub:
    #                 MP.cbLazy(w + lhs >= c_dis[sort_index[k]])
    #             # Cut_index[j, k_th] = 1
    #             # print(f"customer {j}: {w} + {lhs} >= {c_dis[sort_index[k]]}")
    #         elif cbcut:
    #             if obj_j >= max(obj, ub):  # obj_j > lb:   #  and obj >= ub:  # obj_j:
    #                 MP.cbCut(w + lhs >= c_dis[sort_index[k]])
    #             # print(f"customer {j}: {w} + {lhs} >= {c_dis[sort_index[k]]}")
    #         # else:  # elif c_dis[sort_index[k]] > lb:
    #         # elif (obj_j >= min(ub, relax_ub) and (not int_sol)) or ( # c_dis[sort_index[k]] >= ub and int_sol):  # c_dis[sort_index[k]] >= min(obj, ub)
    #         elif (obj_j > np.ceil(lb) and (not int_sol)) or (  # min(ub, relax_ub)
    #                     c_dis[sort_index[k]] <= max(ub - constr_num, lb + 1) and int_sol) or \
    #                 (constr_num <= 0 and obj_j <= ub + con_0 and int_sol):  # c_dis[sort_index[k]] >= min(obj, ub)
    #             MP.addConstr(w + lhs >= c_dis[sort_index[k]], name="cut_"+str(j)+"_"+str(k_th))
    #             # print(f"customer {j}: {w} + lhs >= {c_dis[sort_index[k]]}")
    #             if int_sol and con_0 == 0:
    #                 constr_num += 1
    #                 Cut_index[j, k_th] = 1
    #                 # if updata:
    #                 #     cut_index = np.where(Cut_index[j, :] != 0)[0]
    #                 #     for k1 in cut_index:
    #                 #         constr = MP.getConstrByName(f"cut_{j}_{k1}")
    #                 #         if constr is not None:
    #                 #             if constr.RHS <= lb:
    #                 #                 MP.remove(constr)
    #                 #                 Cut_index[j, k1] = 0
    #                 #             else:
    #                 #                 break
    #                 #     for k2 in cut_index[::-1]:
    #                 #         constr = MP.getConstrByName(f"cut_{j}_{k2}")
    #                 #         if constr is not None:
    #                 #             if constr.RHS > ub:
    #                 #                 MP.remove(constr)
    #                 #                 Cut_index[j, k2] = 0
    #                 #             else:
    #                 #                 break
    #
    #             if MP_int is not None and np.ceil(lb) < c_dis[sort_index[k]] <= max(ub - constr_num, np.ceil(lb) + 1):   #  and c_dis[sort_index[k]] > lb:  # and (ub-mp_obj)/(mp_obj+0.0001) <= 100:
    #                 # if k_th > ub_k_th:
    #                 #     if np.all(Cut_index[j, ub_k_th:k_th]) < 1:
    #                 #         MP_int.addConstr(w_1 + lhs2 >= c_dis[sort_index[k]], name="cut_"+str(j)+"_"+str(k_th))
    #                 #         Cut_index[j, k_th] = 1
    #                 #         # print(f"customer {j}: {w} + lhs >= {c_dis[sort_index[k]]}")
    #                 # else:
    #                 MP_int.addConstr(w_1 + lhs2 >= c_dis[sort_index[k]], name="cut_"+str(j)+"_"+str(k_th))
    #                 Cut_index[j, k_th] = 1
    #                 constr_num += 1
    #                 # print(f"customer {j}: {w} + lhs >= {c_dis[sort_index[k]]}")
    # if updata:
    #     MP.update()
    # return obj, int_obj, constr_num


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
    w = MP.getVarByName(f"w")
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
        MP.addConstr(w >= beta + lhs)
    MP.update()
    return max(sp_obj)


def pure_decomposition(MP, eps=0.00001):
    print('============================================')
    print('============================================')
    SPs = []
    for j in range(Cus_n):
        SPs.append(build_SP(j))
    UB = np.inf
    LB = 0
    y_val, z_val, facility, lb = solve_MP(MP)
    LB = max(LB, lb)
    ub = slove_SPs(SPs, y_val, MP)
    UB = min(UB, ub)
    while abs(UB - LB) / (LB + 0.0001) > 0.0001:
        Gap = round(100 * (UB - LB) / (LB + 0.0001), 4)
        print(' %7.2f ' % UB, end='')
        print(' %7.2f ' % LB, end='')
        print(' %8.4f ' % Gap, end='%')
        print()
        fac = np.where(y_val >= 0.99)[0]
        y_val, z_val, facility, lb = solve_MP(MP)  # , callback=call_back)
        LB = max(LB, lb)
        ub = slove_SPs(SPs, y_val, MP)
        UB = min(UB, ub)
    print()
    print(facility)
    return UB


def add_benders_cut_sep(MP, y_val, mp_obj, cb=False, cbcut=False):
    y = [MP.getVarByName(f"y[{i}]") for i in range(Fac_n)]
    w = MP.getVarByName(f"w")
    obj = 0
    for j in range(Cus_n):
        obj_j = 0
        c_dis = Dis_m[:, j].squeeze()
        max_k_th = len(set(c_dis))
        sort_index = Sort_dis_index[:, j]
        sum_y = y_val[sort_index[0]]
        k = 0
        k_th = 0
        last_dis = c_dis[sort_index[0]]
        while sum_y < 1 and k < len(sort_index):
            k += 1
            sum_y += y_val[sort_index[k]]
            if c_dis[sort_index[k]] > last_dis:
                k_th += 1
                last_dis = c_dis[sort_index[k]]
        obj_j += c_dis[sort_index[k]]
        obj = max(obj_j, obj)
        if obj_j < obj:
            continue
        if k > 0:
            lhs2 = LinExpr()
            cur_f = []
            for m in range(k):
                a = c_dis[sort_index[k]] - c_dis[sort_index[m]]
                if a != 0:
                    cur_f.append(sort_index[m])
                    if c_dis[sort_index[m + 1]] > c_dis[sort_index[m]]:
                        for i in cur_f:
                            MP.addConstr(w + a * y[i] >= c_dis[sort_index[k]] * (
                                        1 - lhs2 - quicksum(y[g] for g in cur_f if g != i)))
                            print(f"{w} + y[{i}] >= {c_dis[sort_index[k]]} * {1 - lhs2}")
                        for i in cur_f:
                            lhs2.addTerms(1, y[i])
                        cur_f = []

            MP.update()
    return obj


def add_benders_cut_2(MP, obj, fac):
    w = MP.getVarByName(f"w")
    y = [MP.getVarByName(f"y[{i}]") for i in range(Fac_n)]
    y_sum = quicksum(y[i] for i in range(Fac_n) if i not in fac)
    int_cut = w - obj * (1 - y_sum)
    MP.addConstr(int_cut >= 0)
    MP.update()
    return int_cut


def call_back(model, where):
    global UB, LB
    if where == GRB.callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
        var = np.array(model.cbGetNodeRel(model._vars))
        ub = np.ceil(model.cbGet(GRB.callback.MIPNODE_OBJBST))
        lb = np.ceil(model.cbGet(GRB.callback.MIPNODE_OBJBND))  # .MIPNODE_OBJBND)
        LB = max(lb, LB)
        y_val = var[1:]
        w_val = var[0]
        # if w_val <= lb:
        rel_obj, int_obj, _ = add_benders_cut(model, y_val, LB, UB, cbcut=True, int_sol=False, updata=0)

    # Lazycut
    # if where == GRB.callback.MIPSOL:
    #     var = np.array(model.cbGetSolution(model._vars))
    #     w_val = var[0]
    #     y_val = var[1:]
    #     ub = model.cbGet(GRB.callback.MIPSOL_OBJBST)
    #     bond = model.cbGet(GRB.callback.MIPSOL_OBJBND)
    #     LB = max(np.ceil(bond), LB)
    #     # UB = min(ub, UB)
    #     rel_obj, int_obj, _ = add_benders_cut(model, y_val, LB, UB, cb=True, updata=False, int_sol=True)
    #     if int_obj < UB:
    #         UB = int_obj
    #         y_index = np.argsort(y_val)[-Fac_L: ]
    #         y_s = np.zeros(Fac_n)
    #         y_s[y_index] = 1
    #         solution = np.hstack([np.array(int_obj), y_s]).tolist()
    #         model.cbSetSolution(model._vars, solution)
    #     if int_obj <= LB:
    #         model.terminate()


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
    # orig_model.setParam('PreCrush', 1)
    # orig_model.optimize(call_back_lazy)
    UB = orig_model.ObjVal
    return UB, opt_time


def Benders_solve():
    org_model = build_MP(int_sol=True)
    opt_time = time.time()

    global UB, LB
    # UB1, f_set = get_UB1(Dis_m, Fac_L)
    # obj = np.max(np.min(Dis_m[:, list(f_set)], axis=1))
    UB, UB1, f_set = get_UB2(Dis_m, Fac_L)
    obj = np.max(np.min(Dis_m[:, list(f_set)], axis=1))
    UB = min(UB, obj)
    LB = np.ceil(UB1 / 2)
    print(LB, UB)
    LB_0, UB_0 = Benders_Decomposition(org_model, UB, LB)
    LB, UB = LB_0, UB_0
    # LB_0 = LB
    # LB = 0
    # UB = 100000

    org_model.Params.PoolGap = 0.01
    org_model.setParam('TimeLimit', 10)
    # org_model.Params.PoolSolutions = 5
    # org_model.Params.timeLimit = 600
    # org_model.setParam('PreSolve', 2)
    org_model.setParam('OutputFlag', 0)
    # org_model.setParam('LazyConstraints', 1)
    # org_model.setParam('MIPFocus', 3)  # 3 优化边界
    # org_model.setParam('Method', 0)
    # org_model.setParam('PreCrush', 1)
    # org_model.setParam('RINS', 2500)
    # org_model.setParam('Cuts', 1)
    # org_model.setParam('CutPasses', 1)
    # org_model.setParam('Heuristics', 0.001)
    # org_model.tune()
    print('============================================')
    print('============================================')
    for i in range(Fac_n):
        if i in f_set:
            org_model.getVarByName(f"y[{i}]").setAttr(GRB.Attr.Start, 1.0)
        else:
            org_model.getVarByName(f"y[{i}]").setAttr(GRB.Attr.Start, 0.0)
    org_model.addConstr(org_model.getVarByName(f"w") >= LB)
    org_model.update()
    update_model = 0
    y_val, z_val, facility, lb = solve_MP(org_model)  # , callback=call_back)
    if lb > LB:
        update_model += 1
        LB = lb
        org_model.addConstr(org_model.getVarByName(f"w") >= LB)
    # LB = max(LB, lb)
    for m in range(0, org_model.SolCount):
        org_model.Params.SolutionNumber = m
        pool_obj = org_model.PoolObjVal
        y_star = np.array(org_model.Xn[1:])
        if m == 0:
            for i in range(len(y_star)):
                org_model.getVarByName(f"y[{i}]").setAttr(GRB.Attr.Start, y_star[i])
        # else:
        #     _, ub, constr_num = add_benders_cut(org_model, y_star, LB, UB, updata=False)
        #     UB = min(UB, ub)
        #     print(constr_num)
    # org_model.update()

    _, ub, constr_num = add_benders_cut(org_model, y_val, LB, UB, updata=update_model)
    update_model = 0
    # print(constr_num)
    # UB = min(UB, ub)
    if ub < UB:
        update_model += 2
        UB = ub
    inter = 0
    no_change_cnt = 0
    while abs(UB - LB) / (LB + 0.001) > 0.0001:
        Gap = round(100 * (UB - LB) / (LB + 0.0001), 4)
        print(' %7.2f ' % LB, end='')
        print(' %7.2f ' % UB, end='')
        print(' %8.4f ' % Gap, end='%')
        print(f' current time cost: time = {time.time() - t_initial}')
        print()
        print(constr_num)
        y_val, z_val, facility, lb = solve_MP(org_model)  # , callback=call_back)
        # LB = max(LB, lb)
        if lb > LB:
            update_model += 1
            LB = lb
            org_model.addConstr(org_model.getVarByName(f"w") >= np.ceil(LB))
        for m in range(0, org_model.SolCount):
            org_model.setParam('SolutionNumber', m)
            pool_obj = org_model.PoolObjVal
            y_star = np.array(org_model.Xn[1:])
            if m == 0:
                for i in range(len(y_star)):
                    org_model.getVarByName(f"y[{i}]").setAttr(GRB.Attr.Start, y_star[i])
            # else:
            #     _, ub, constr_num = add_benders_cut(org_model, y_star, LB, UB, updata=False)
            #     print(constr_num)
            #     UB = min(UB, ub)
        # org_model.update()
        _, ub, constr_num = add_benders_cut(org_model, y_val, LB, UB, con_0=no_change_cnt, updata=update_model)
        update_model = 0
        # if ub > UB:
        #     adf = 1
        if ub <= LB:
            UB = LB
        elif ub < UB:
            update_model += 2
            UB = ub
        inter += 1
        if constr_num <= 0:
            no_change_cnt += 1
        elif constr_num > 0:
            no_change_cnt = 0
        if time.time() - opt_time >= 3600:
            break
    print()
    # print(facility)
    return UB, LB, opt_time, inter, LB_0


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
    df = pd.DataFrame(columns=["pmed No.", "Fac_n", "Optima", "LB", "opt_time", "total_time", "Iter_num", "LB_0"])
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
    # data_sets = ["u1817"]  # ["rat575","pcb1173", "u1060", "dsj1000"]
    # data_sets = [50]  # [10, 20, 30, 40, 50]
    # data_sets = ["Manhattan", "chengdu", "Portland", "beijing"]
    fac_number = [5]  # [5, 10, 20, 50, 100, 200, 300, 400, 500]

    for i in data_sets:
        for f_n in fac_number:
            instance = load_data(data_type, data_set=i, fac_n=f_n)
            Dis_m = instance.dis_matrix
            std = np.std(Dis_m)
            mean = np.mean(Dis_m)
            Cus_n = instance.customer_num
            Fac_n = instance.facility_num
            Sort_dis_index = np.argsort(Dis_m, axis=1)
            Sort_dis = np.sort(np.unique(Dis_m.flatten()))
            Cut_index = np.zeros((Cus_n, len(Sort_dis)), dtype=int)  # len(Sort_dis))
            # Sqs_index = cal_sqs_info(Dis_m, Sort_dis_index)
            t_initial = time.time()

            # Solve the original MIP model
            # UB, opt_time = solve_MIP()

            # IBD algorithm
            UB, LB, opt_time, inter, LB_0 = Benders_solve()

            result.append(UB)
            iter_num.append(inter)
            time_spend.append(time.time() - opt_time)
            print('============================================')
            print('Instance ', i)
            print('Optimal objective: ', UB)
            # print('Optimal Location', facility)
            print("opt_time:", time.time() - opt_time)
            print(f'whole time cost: time = {time.time() - t_initial}')
            df.loc[len(df.index)] = [i, f_n, UB, LB, time.time() - opt_time, time.time() - t_initial, inter, LB_0]

    print("Result:", result)
    print("time_spend", time_spend)
    df.to_csv("pemd_pcenter_result.csv")
    print(df)
