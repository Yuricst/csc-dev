"""
Return trip run optimization
"""


import numpy as np
import numpy.linalg as la
import pykep as pk
from pykep import epoch
import pygmo as pg
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import spiceypy as spice
import argparse
import time

import sys
sys.path.append("../")
import pyxplorer as pxp


def run(t0, vinf, n_rev = 4, pop_size = 30, choice_algo = "de", 
        generation=5000,
        separate_objectives=0
    ):
    # construct problem
    tstart = time.time()
    saturn_system_list = pxp.saturn_system()
    titan_period = saturn_system_list["titan"].compute_period(epoch(2345.3, 'mjd2000')) / 86400  # days

    seq = [saturn_system_list["titan"] for el in range(n_rev+1)]
    tof = [
        [0.75*titan_period, 7*titan_period] for el in range(len(seq)-1)
    ]  # in days

    # construct problem
    prob_titan = pxp.mga_1dsm_vilt(
        t0,
        vinf,
        seq,
        tof,
        pxp.MU_SATURN,
        separate_objectives=separate_objectives,
    )


    algo = pxp.algo_factory(choice=choice_algo, generation=generation)
    print(f"Algorithm: {choice_algo}, pop_size: {pop_size}, generation: {generation}")

    pop = pg.population(prob=prob_titan, size=pop_size)
    pop = algo.evolve(pop)
    tend = time.time()
    print(f"Took {(tend-tstart)/60:3.4f} minutes")
    return prob_titan, pop


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run interplanetary transfer")
    parser.add_argument('-n', '--num_rev', help='sequence of flyby: ')
    parser.add_argument('-d', '--data', help='store data to file: ')
    parser.add_argument('-p', '--pop_size', help='population size: ')
    parser.add_argument('-g', '--generation', help='generation size: ')
    parser.add_argument('-a', '--algorithm', help='choice of algorithm ')
    parser.add_argument('-m', '--multiobjective', help='multiobjective ')
    args = parser.parse_args()


    if args.num_rev:
        n_rev = int(args.num_rev)
    else:
        n_rev = 3

    if args.algorithm:
        choice_algo = args.algorithm
    else:
        choice_algo = "gaco"

    if args.pop_size:
        pop_size = int(args.pop_size)
    else:
        pop_size = 30

    if args.generation:
        generation = int(args.generation)
    else:
        generation = 5000

    if args.multiobjective:
        separate_objectives = int(args.multiobjective)
    else:
        separate_objectives = 0


    t0 = pk.epoch_from_string('2044-12-14 00:00:00.000').mjd2000
    vinf = 4000.0  # fixed
    prob_titan, pop = run(t0, vinf, n_rev, pop_size, choice_algo, generation, separate_objectives)

    if separate_objectives == 0:
        DV, r_P, v_P, vinf_final, traj_comp, dsm_info = prob_titan.evaluate_trajectory(pop.champion_x)
        print(f"DSM Cost:           {np.linalg.norm(DV)}")
        print(f"Arrival v-inf Cost: {np.linalg.norm(vinf_final)}")
        print(f"Total Cost:         {np.linalg.norm(vinf_final) + np.linalg.norm(DV)}")


    # get number of files existing
    filenames = os.listdir("../notebooks/optim_res_saturn_approach")
    n_data_already = 0
    for filename in filenames:
        if "seq_fb"+str(n_rev)+ "_" in filename:
            n_data_already += 1
    # save result array
    if args.data:
        choose_save = args.data
    else:
        choose_save = input("save (y/n)? [y]: ")

    # # get combined list of xs and fs
    # for idx,pop_iter in enumerate(pop_list):
    #     if idx == 0:
    #         x_combined = pop_iter.get_x()
    #         f_combined = pop_iter.get_f()
    #     else:
    #         x_combined = np.concatenate((x_combined, pop_iter.get_x()),axis=0)
    #         f_combined = np.concatenate((f_combined, pop_iter.get_f()))

    if choose_save != "n":
        saved_file = "../notebooks/optim_res_saturn_approach/seq_fb"+str(n_rev)+"_"+str(n_data_already+1)
        np.save(saved_file, pop.get_x())
        print(f"Saved data at {saved_file}")

    # plot