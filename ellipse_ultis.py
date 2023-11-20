import copy

import numpy as np

def get_UB1(dis_matrix, fac_L):
    # dis_matrix = copy.deepcopy(dis_m)
    # maxf = np.max(dis_matrix)
    max_dis_index = np.argmax(dis_matrix)
    r, c = divmod(max_dis_index, dis_matrix.shape[1])
    f_set = [r, c]
    # dis_matrix[list(f_set), :] = 0
    # dis_matrix[r, c] = 0
    # dis_matrix[c, r] = 0
    while len(f_set) < fac_L:
        # a = dis_matrix[:, list(f_set)]
        # max_dis = np.min(dis_matrix[:, list(f_set)], axis=1)
        max_dis_index = np.argmax(np.min(dis_matrix[:, list(f_set)], axis=1))
        # r, c = divmod(max_dis_index, len(f_set))
        assert max_dis_index not in f_set
        f_set.append(max_dis_index)
        # dis_matrix[r, :] = 0
    # b = np.min(dis_matrix[:, list(f_set)], axis=1)
    obj = np.max(np.min(dis_matrix[:, list(f_set)], axis=1))
    return obj, f_set


def get_UB2(dis_m, fac_l):
    UB1, f_set = get_UB1(dis_m, fac_l)
    best_obj = UB1
    # min_dis = np.argmin(dis_m[:, list(f_set)], axis=1)
    I_set = [list(np.where(np.argmin(dis_m[:, list(f_set)], axis=1) == i)[0]) for i in range(len(f_set))]
    obj_k = np.array([np.max(dis_m[I_set[i], list(f_set)[i]]) for i in range(len(f_set))])
    best_obj, new_f_set = solve_1_p_center(dis_m, best_obj, I_set, obj_k, f_set)
    return best_obj, UB1, new_f_set

def solve_1_p_center(dis_m, best_obj, I_set, obj_k, f_set):
    remain_set = np.where(obj_k >= best_obj)[0]
    new_f_set = copy.copy(f_set)
    for i in remain_set:
        sub_dis = dis_m[I_set[i], :][:, I_set[i]]
        new_obj = np.min(np.max(sub_dis, axis=1))
        new_f = I_set[i][np.argmin(np.max(sub_dis, axis=1))]
        if new_obj == best_obj:
            return new_obj, new_f_set
        else:
            obj_k[i] = new_obj
            new_f_set[i] = new_f
            best_obj = max(obj_k)
            best_obj, new_f_set = solve_1_p_center(dis_m, best_obj, I_set, obj_k, new_f_set)
    return best_obj, new_f_set








