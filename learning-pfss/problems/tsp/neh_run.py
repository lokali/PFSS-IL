import numpy as np
import time

# from schedule_time import schedule_time_parallel, schedule_time_one_machine
np.random.seed(1)

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
    return makespan


def my_sort(pt):
    seq = []
    for item in pt:
        seq.append(np.sum(item))
    #print("sum_time: ", seq)
    seq_ = sorted(range(len(seq)), key=lambda k: seq[k], reverse=True)
    #print("seq_tmp: ",seq_)
    return seq_

def my_insert(seq_tmp, pt, machine_num, start):
    if len(seq_tmp) < 2:
        return seq_tmp
    #print("test 1: ",seq_tmp)
    seq = seq_tmp[:2]
    #print("test2: ",seq)

    for i in range(len(seq_tmp)-2):
        # the next job to be inserted
        new_job_index = len(seq)
        #print("test 4: new job index",new_job_index)
        new_job = seq_tmp[new_job_index]
        #print("test 5: new job", new_job)
        # det = all the makespan with different insert
        det = []
        for i in range(len(seq)+1):
            # should be seq[:] rather than seq
            # the change in tmp will not change in seq
            tmp = seq[:]
            tmp.insert(i, new_job)
            #print("!test 6: tmp ---", tmp)
            # makespan = schedule_time_parallel(pt, tmp, machine_num)
            makespan = schedule_time_one_machine(pt, tmp)
            det.append(makespan)

        #print("test 7: ",det)
        # det_index = the best insert index with smallest makespan
        det_index = det.index(min(det))
        #print("test 8: ", det_index)
        #print("test 9: before insert ", seq)
        # insert seq, so that len(seq) += 1
        seq.insert(det_index, new_job)
        #print("test 10: after insert ", seq)
        # print("seq: ",seq)
        # print("ready seq length: ", len(seq))
        # print("cost time: ", time.time() - start)
        # print()
    return seq

def neh(pt, machine_num=[5,5,5], start=0):
    # sort from longest time to shortest time
    seq_tmp = my_sort(pt)
    #print("ordered seq: ", seq_tmp)
    #print()

    # seq is the final seq, a list [].
    seq = my_insert(seq_tmp, pt, machine_num, start)
    seq = np.array(seq)

    # find the makespan of the best seq
    # makespan = schedule_time_parallel(pt, seq, machine_num) 
    makespan = schedule_time_one_machine(pt, seq)

    return makespan, seq


if __name__=='__main__':
    machine_num = [10, 10, 10, 10, 10]
    #X = np.random.beta(1, 3, size=(j_num, 5))*100  # beta
    # X = np.random.normal(0, 1, size=(j_num, 5))  # Gaussion, normal
    # X = np.random.randn(j_num, 5)  # standard normal distribution
    # X = np.random.possion(size=(j_num, 5))  # possion distribution

    start = time.time()
    j_num = 100
    X = np.random.rand(j_num, 5)
    ms500, _ = neh(X)
    print("makespan",ms500)
    end = time.time()
    print("cost time: ", end - start)

