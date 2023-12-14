#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
from functools import lru_cache
from itertools import permutations
from random import randint, shuffle

import numpy as np

# from geneticFunctions import *
from utils import schedule_time_one_machine

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import shutil
import os.path

from pyomo.environ import *
from pyomo.gdp import *


np.random.seed(1)


def visualize(results, path):
    
    schedule = pd.DataFrame(results)
    JOBS = sorted(list(schedule['Job'].unique()))
    MACHINES = sorted(list(schedule['Machine'].unique()))
    makespan = schedule['Finish'].max()
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
    colors = mpl.cm.Dark2.colors

    schedule.sort_values(by=['Job', 'Start'])
    schedule.set_index(['Job', 'Machine'], inplace=True)

    fig, ax = plt.subplots(2,1, figsize=(12, 5+(len(JOBS)+len(MACHINES))/4))

    for jdx, j in enumerate(JOBS, 1):
        for mdx, m in enumerate(MACHINES, 1):
            if (j,m) in schedule.index:
                xs = schedule.loc[(j,m), 'Start']
                xf = schedule.loc[(j,m), 'Finish']
                ax[0].plot([xs, xf], [jdx]*2, c=colors[mdx%7], **bar_style)
                ax[0].text((xs + xf)/2, jdx, m, **text_style)
                ax[1].plot([xs, xf], [mdx]*2, c=colors[jdx%7], **bar_style)
                ax[1].text((xs + xf)/2, mdx, j, **text_style)
                
    ax[0].set_title('Job Schedule')
    ax[0].set_ylabel('Job')
    ax[1].set_title('Machine Schedule')
    ax[1].set_ylabel('Machine')
    
    for idx, s in enumerate([JOBS, MACHINES]):
        ax[idx].set_ylim(0.5, len(s) + 0.5)
        ax[idx].set_yticks(range(1, 1 + len(s)))
        ax[idx].set_yticklabels(s)
        ax[idx].text(makespan, ax[idx].get_ylim()[0]-0.2, "{0:0.1f}".format(makespan), ha='center', va='top')
        ax[idx].plot([makespan]*2, ax[idx].get_ylim(), 'r--')
        ax[idx].set_xlabel('Time')
        ax[idx].grid(True)
        
    fig.tight_layout()
    # plt.show()
    # plt.savefig(path)
    print("Success end.")



def compare(data):
    n_jobs, n_machines = data.shape 

    ms_list = []
    time_list = []

    # print("----------------- Random ---------------------------")
    start_time = time.time()
    makespan = []
    best_seq = []
    best_makespan = np.inf 
    for i in range(10):
        seq = np.random.permutation(n_jobs)
        ms = schedule_time_one_machine(data, seq)
        makespan.append(ms)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = seq 
    b = np.mean(makespan)
    c = min(makespan)
    rd_cost = time.time() - start_time
    # print("random makespan:  avg | min:", b,c)
    # print('Cost time of random: %s' % (rd_cost))
    ms_list.append(c)
    time_list.append(rd_cost)
    
    from ig import local_search
    # print("\n----------------- Local Search ---------------------------")
    start_time = time.time()
    makespan = []
    seq = best_seq 
    for i in range(10):
        seq, ms = local_search(data, seq, n=10)
        makespan.append(ms)
        if ms < best_makespan:
            best_makespan = ms
            best_seq = seq
    ls_cost = time.time() - start_time + rd_cost
    seq_ls = best_seq 
    # print("local search makespan: ", best_makespan)
    # print('Cost time of local search: %s' % (ls_cost))
    ms_list.append(best_makespan)
    time_list.append(ls_cost)

    from neh import neh
    # print("\n---------------- NEH -------------------------------")
    start_time = time.time()
    ms, seq_neh = neh(data)
    neh_cost = time.time() - start_time
    # print("NEH makespan",ms)
    # print('Cost time of NEH: %s' % (neh_cost))
    ms_list.append(ms)
    time_list.append(neh_cost)

    from ig import HIG 
    # print("\n----------------- HIG ---------------------------")
    start_time = time.time()
    seq_ig, makespan = HIG(data, seq_ls, 10)
    ig_cost = time.time() - start_time + ls_cost
    # print("HIG makespan: ", makespan)
    # print('Cost time of HIG: %s' % (ig_cost))
    ms_list.append(makespan)
    time_list.append(ig_cost)

    return ms_list, time_list 


def get_data(distribution, num_samples, min_size):
    if distribution=='rand':
        nodes_coords = [torch.FloatTensor(min_size, 5).uniform_(0, 1) for i in range(num_samples)]  # random generator 
    if distribution=='rand100':
        nodes_coords = [torch.Tensor(np.random.randint(100, size=(min_size, 5))) for i in range(num_samples)]  # random generator 
    elif distribution=='chi': # gamma distribution with theta=2
        # print('this is chi!!!')
        nodes_coords = [torch.Tensor(np.random.chisquare(1, size=(min_size, 10))) for i in range(num_samples)]
    elif distribution=='beta13':
        nodes_coords = [torch.Tensor(np.random.beta(1, 3, size=(min_size, 5))) for i in range(num_samples)]
    elif distribution=='beta51':
        nodes_coords = [torch.Tensor(np.random.beta(5, 1, size=(min_size, 5))) for i in range(num_samples)]
    elif distribution=='normal':
        nodes_coords = [torch.Tensor(np.random.normal(6, 6, size=(min_size, 5))) for i in range(num_samples)]
        for i in range(num_samples):
            for j in range(min_size):
                for k in range(5):
                    if nodes_coords[i][j,k] <= 0:
                        nodes_coords[i][j,k] = 0
    return nodes_coords 


import torch 
import numpy as np
np.random.seed(1)
if __name__ == "__main__":
    n_samples = 2
    n_machines = 5

    job = [20, 50, 100, 200, 500, 1000]
    n_jobs = job[0] 

    print(f"There are {n_samples} sample(s) with {n_jobs} jobs, {n_machines} machines.") 
    #  | random | ILS | NEH | IG |
    dataset = get_data('chi', n_samples, n_jobs)
    
    ms = []
    cost = []
    for i in range(n_samples):
        data = dataset[i].cpu().numpy()
        ms_list, time_list = compare(data)
        ms.append(ms_list)
        cost.append(time_list)
        # print(ms_list)
        # print(time_list)
    ms = np.array(ms)
    cost = np.array(cost)
    ms = np.mean(ms, 0)
    cost = np.sum(cost, 0) 
    
    ms = [round(i,2) for i in ms]
    cost = [round(i,4) for i in cost]

    print("makespan: ", ms)
    print("Time Cost:", cost)





