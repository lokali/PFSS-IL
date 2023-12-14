import numpy as np
from random import random
from math import exp
from time import time
from utils import schedule_time_one_machine
np.random.seed(1)

def destruction_construction(Dataset, current_sol, d=10):
    """Removes randomly self.d jobs from the current solution, and places
    them back iteratively at the best location.
    """
    # current_sol = np.array(current_sol)
    removed_idx = np.random.choice(len(current_sol), d, replace = False)
    removed_idx = np.array(removed_idx)
    removed_jobs = current_sol[removed_idx]
    current_sol = np.delete(current_sol, removed_idx)
    current_eval = float("inf")

    # Iterate on all removed jobs to place back
    for i in range(d):
        best_new_sol = None
        best_new_eval = float("inf")
        # Initialize the completion time matrix, for optimized partial evaluations
        # comp_times = np.empty((self.n, self.m))
        # self.partial_evaluator(current_sol, pos = 0, comp_times = comp_times)

        # Find the best place to insert the removed job
        for j in range(len(current_sol)):
            new_sol = np.insert(current_sol, j, removed_jobs[i])
            # new_eval = self.partial_evaluator(new_sol, pos = j, comp_times = comp_times)
            new_eval = schedule_time_one_machine(Dataset, new_sol)
            if new_eval < best_new_eval:
                best_new_sol = new_sol
                best_new_eval = new_eval

        current_sol = best_new_sol
        current_eval = best_new_eval

    return current_sol, current_eval

def numpy_move(array, from_pos, to_pos):
        """Moves the element at from_pos in array, and place it before to_pos."""
        val_from = array[from_pos]
        if from_pos > to_pos:
            res = np.delete(array, from_pos)
            res = np.insert(res, to_pos, val_from)
        else:
            res = np.insert(array, to_pos, val_from)
            res = np.delete(res, from_pos)
        return res

def numpy_swap(array, from_pos, to_pos):
    """Swaps elements at from_pos and to_pos in array."""
    res = np.copy(array)
    res[from_pos] = array[to_pos]
    res[to_pos] = array[from_pos]
    return res

def local_search(dataset, current_sol, n=10):
    """Local search on the current solution: with probability 0.5, swap to
    random jobs. Otherwise, move a random job at a random position. Repeat
    until no improvement is made for self.n iterations.
    """
    k = 1
    current_eval = float("inf")
    # comp_times = np.empty((n, m))
    # self.partial_evaluator(current_sol, pos = 0, comp_times = comp_times)

    while k <= n:
        (pos_1, pos_2) = np.random.choice(len(current_sol), 2, replace = False)
        if random() < 0.5:
            new_sol = numpy_move(current_sol, pos_1, pos_2)
        else:
            new_sol = numpy_swap(current_sol, pos_1, pos_2)
        # new_eval = partial_evaluator(new_sol, pos = min(pos_1, pos_2), comp_times = comp_times)
        new_eval = schedule_time_one_machine(dataset, new_sol)

        if new_eval < current_eval:
            current_sol = new_sol
            current_eval = new_eval
            k = 1
        else:
            k += 1

    return current_sol, current_eval

def HIG(Dataset, current_sol,  max_n):
    """
    Hybrid iterated greedy. 
    Main optimization procedure. Continues until max_time seconds elapsed.
    """
    size, m_machine = Dataset.shape 
    start_time = time()
    # current_sol, current_eval = local_search(self.NEH_edd())

    # current_sol = np.random.permutation(size)
    # current_sol = init_seq[:]
    current_eval = float("inf")
    best_sol = current_sol
    best_eval = float("inf")
    i = 0

    while i < max_n:
        new_sol, new_eval = destruction_construction(Dataset, current_sol, 10) # delete and re-insert d jobs. 
        new_sol, new_eval = local_search(Dataset, new_sol, 10) # local search for n jobs.  
        # If the new solution is better than the current one, keep it
        if new_eval < current_eval:
            current_sol = new_sol
            current_eval = new_eval
            if new_eval < best_eval:
                best_sol = new_sol
                best_eval = new_eval
        # Keep the new solution even if it not better, with a small probability
        # elif random() < 0.05:
        #     current_sol = new_sol
        #     current_eval = new_eval
        # self.log_convergence(time() - start_time, i, current_eval)
        #print("Iteration", i, "done")
        i += 1

    # print("Solution: ", best_sol)
    # print("Makespan:", best_eval)
    # print("Iterations: ", i)
    # print("Elapsed time:", time() - start_time)
    return best_sol, best_eval


if __name__=='__main__':
    start = time()
    size = 100 # job number 
    dataset = np.random.rand(size, 5)


    print("\n----------------- HIG  ---------------------------")
    start_time = time()
    _, makespan = HIG(dataset, size, 100)
    print("HIG makespan: ", makespan)
    print('Cost time of HIG: %s' % (time() - start_time))