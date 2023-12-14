import numpy as np
import sys
import copy

def offset(D, s_idx, s_val, next_job_idx, next_job_val):
    # find min val to offset, similar to push min_val time forward
    min_val = min(s_val)
    min_idx = np.argmin(s_val)
    s_val = s_val - min_val
    # update completed job list
    D = {key: val + min_val for key, val in D.items()}
    D[s_idx[min_idx]] = 0
    # add new job into list / or remove job out of list
    s_idx[min_idx] = next_job_idx
    s_val[min_idx] = next_job_val
    return D, s_idx, s_val


def one_stage(M, j_seq, machine_num):
    job_num = len(j_seq)
    s_val = [10000] * machine_num # current working list, working time
    s_idx = [-1] * machine_num           # current working list, job seq_num
    D = {}                          # a completed list, key=seq_num, value=waiting time.
    # job数量不如machine数，则放置一次即可
    if job_num <= machine_num:
        # put first
        for j in range(job_num):
            s_idx[j] = j_seq[j]
            s_val[j] = M[j_seq[j]]

        # clean then
        for j in range(job_num):
            D, s_idx, s_val = offset(D, s_idx, s_val, -1, 10000)
        return D

    # 除了首次铺满，还需考虑剩余下放; 下放位置应为：最早空闲的机器
    else:
        # 首先铺满所有machine
        for j in range(machine_num):
            s_idx[j] = j_seq[j]
            s_val[j] = M[j_seq[j]]
        # 其次考虑剩余job
        for j in range(machine_num,job_num):
            next_job_idx = j_seq[j]
            next_job_val = M[j_seq[j]]
            D, s_idx, s_val = offset(D, s_idx, s_val, next_job_idx, next_job_val)

        # 全部放完之后 要逐一把剩余的机器上面的任务都处理完
        for j in range(machine_num):
            D, s_idx, s_val = offset(D, s_idx, s_val, -1, 10000)
        return D

def Update(M, D_pre, D_cur):
    # use list to store combined dict
    D_list = []
    for key,value in D_cur.items():
        val = value + (D_pre[key] - M[key]) #############################
        d_list = [key, val]
        D_list.append(d_list)
    #print("before sort D: ",D_list)

    # sorted list
    D_list = sorted(D_list, key=lambda item:item[1], reverse=True)
    #print("after sorting: D", D_list)

    # transfer list to dict, as return
    D = {}
    for item in D_list:
        key = item[0]
        val = item[1]
        D[key] = val
    return D


# ----------------------------------------------------------------------------------------------------------------------
# multi stage, multi machine
def schedule_time_parallel(M, j_seq, machine_num):

    job_num = len(j_seq)
    stage_num = len(M[0])
    # first stage
    dt = M[:, 0]
    D = one_stage(dt, j_seq, machine_num[0])


    D0 = copy.deepcopy(D) # operation on D0 will not affect the value of D
    key0 = list(D0.keys())[0]
    val0 = list(D0.values())[0]
    # v0 = total time in the first stage
    v1 = dt[key0] + val0 # 整个第一阶段的总花费时间 = 第一个任务的处理时间 + 等待时间
    #print("D: ",D0)
    #print("v1: ",v1)

    # second to last stages
    for i in range(1,stage_num):
        dt = M[:,i]
        j_seq = list(D.keys())
        D_ = one_stage(dt, j_seq, machine_num[i])
        D = Update(dt, D, D_)

        #print("D_, i=:",i,  D_)
        #print("Combined D: ", D)

    D_final = copy.deepcopy(D)
    key_0 = list(D_final.keys())[0]
    # largest one - smallest one， 最后一个的数值可能为负，因此需要根据这个差值来确定处理总时间
    v2 = list(D_final.values())[0] - list(D_final.values())[-1]

    # (v1 - v0) = total pre_waiting and process time in the first stage for key_
    v0 = D0[key_0] ######################
    v = (v1 - v0) + v2

    makespan = v
    return makespan
# ----------------------------------------------------------------------------------------------------------------------

# M: dataset [batch, j_num, node_dim]
# S: sequence [batch, j_num]
# return: a list with size [batch]
def schedule_time_Batch_1(M, S):
    Makespan = [0] * len(S)
    D = []
    for i in range(len(S)):
        data = np.array(M[i])
        seq = np.array(S[i])
        Makespan[i], d = schedule_time_one_machine(data, seq)  # d: a list with size: number of machine
        # d = d - min(d)
        D.append(d)
    return Makespan, D

def schedule_time_Batch_2(M, S, E):
    Makespan = [0] * len(S)
    D = []
    for i in range(len(S)):
        data = np.array(M[i])
        seq = np.array(S[i])
        e = np.array(E[i])
        Makespan[i], d = schedule_time_one_machine_3(data, seq, e)  # d: a list with size: number of machine
        # d = d - min(d)
        D.append(d)
    return Makespan, D

# def schedule_time_Batch_2(M, S, machine_num):
#     S = np.array(S)
#     Makespan = [0] * len(S)

#     for i in range(len(S)):
#         Makespan[i] = schedule_time_parallel(M,  S[i], machine_num)

#     best_idx = np.argmin(Makespan)
#     best_ms = Makespan[best_idx]
#     best_seq = S[best_idx]
#     return best_ms, best_seq




# multi stage, only 1 machine per stage
def schedule_time_one_machine(M,j_seq):
    m_seq = [i for i in range(len(M[0]))]
    D = [0] * len(m_seq)
    D[0] = M[j_seq[0]][m_seq[0]] # 第1个job 第一个阶段
    for i in range(1, len(m_seq)):
        D[i] = D[i - 1] + M[j_seq[0]][m_seq[i]] # 第1个job 剩下的阶段

    for i in range(1, len(j_seq)):
        D[0] += M[j_seq[i]][m_seq[0]]   # 剩下个job 的第1个阶段

        for j in range(1, len(m_seq)):
            D[j] = max(D[j], D[j - 1]) + M[j_seq[i]][m_seq[j]] # 剩余job的 接下来阶段
    makespan = D[len(m_seq) - 1]
    #reward = 1 / (makespan + 1)
    #reward = -makespan
    return makespan, D 


# evauation with preconditioned job. 
def schedule_time_one_machine_3(M,j_seq, E): # E: len=5 
    m_seq = [i for i in range(len(M[0]))]
    # D = [0] * len(m_seq)
    D = E 
    # D[0] += M[j_seq[0]][m_seq[0]] # 第1个job 第一个阶段
    # for i in range(1, len(m_seq)):
    #     D[i] = max(D[i - 1], D[i])  + M[j_seq[0]][m_seq[i]] # 第1个job 剩下的阶段

    for i in range(len(j_seq)):
        D[0] += M[j_seq[i]][m_seq[0]]   # 剩下个job 的第1个阶段

        for j in range(1, len(m_seq)):
            D[j] = max(D[j], D[j - 1]) + M[j_seq[i]][m_seq[j]] # 剩余job的 接下来阶段

    makespan = D[len(m_seq) - 1]
    #reward = 1 / (makespan + 1)
    #reward = -makespan
    return makespan, D 