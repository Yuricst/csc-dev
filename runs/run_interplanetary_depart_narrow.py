"""
Return trip run optimization
"""


import numpy as np
import numpy.linalg as la
import pykep as pk
from pykep.planet import jpl_lp
from pykep import epoch
import pygmo as pg
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import spiceypy as spice
import argparse

import sys
sys.path.append("../")
import pyxplorer as pxp

pk.util.load_spice_kernel(os.path.join(os.environ.get("SPICE"), "spk", "de440.bsp"))
pk.util.load_spice_kernel(os.path.join(os.environ.get("SPICE"), "lsk", "naif0012.tls"))


from run_interplanetary_return import *


def run_optim_depart(seq_key, algo, t0range=None, pop_size=10, n_t0=100, vinf_launch_max=4.5):
    """Construct pygmo departure transfer problem with MGA1DSM model

    Args:
        seq_key (list): list of body names, e.g. `["saturn", "saturn", "earth"]`
        algo (pg.algorithm): pygmo algorithm object
        t0range (list): earliest and latest launch-epoch window, in str, e.g. '2036-01-01 00:00:00.000'
        pop_size (int): population size for problem
        n_t0 (int): number of windows of launch-epoch

    Returns:
        (tuple): list of decision vectors, list of problems
    """
    # get solar system
    ssdict = pxp.solar_system_spice()

    # tof = [
    #     [0.2/pk.DAY2YEAR, 6/pk.DAY2YEAR] for el in range(len(seq_key)-1)
    # ]
    tof = [
        [3/pk.DAY2YEAR, 3.5/pk.DAY2YEAR],
        [4/pk.DAY2YEAR, 4.5/pk.DAY2YEAR],
    ]

    pop_list = []
    prob_list = []

    if t0range is None:
        t0_earliest = pk.epoch_from_string('2036-01-01 00:00:00.000').mjd2000
        t0_latest   = pk.epoch_from_string('2044-01-01 00:00:00.000').mjd2000
    else:
        t0_earliest = pk.epoch_from_string(t0range[0]).mjd2000
        t0_latest   = pk.epoch_from_string(t0range[1]).mjd2000
    t0_range = np.linspace(t0_earliest, t0_latest, n_t0)
    dt0 = 1.1*(t0_range[1] - t0_range[0])  # have some overlap

    for i_window in tqdm(range(n_t0), desc="scanning launch window"):
        t0_iter = [
            t0_range[i_window],
            t0_range[i_window]+dt0
        ]

        # run problem
        seq = [ssdict[el] for el in seq_key]
        prob_iter = pxp.get_depart_problem(seq, t0_iter, tof, vinf_launch_max)
        pop = pg.population(prob=prob_iter, size=pop_size)
        pop = algo.evolve(pop)

        # store
        pop_list.append(pop)
        prob_list.append(prob_iter)

    return pop_list, prob_list



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run interplanetary transfer")
    parser.add_argument('-s', '--sequence', help='sequence of flyby: ')
    parser.add_argument('-d', '--data', help='store data to file: ')
    parser.add_argument('-a', '--algorithm', help='choice of algorithm ')
    args = parser.parse_args()
    if args.sequence:
        seq_key = [
            planet_int2str(int(args.sequence[el])) for el in range(len(args.sequence))
        ]
    else:
        seq_key = ["earth", "jupiter", "saturn"]

    # get sequence name string
    seq_name = ""
    for el in seq_key:
        seq_name += el + "-"
    seq_name = seq_name[:-1]

    print(f"Running sequence: {seq_name}")

    # tune algorithm
    if args.algorithm:
        choice_algo = args.algorithm
    else:
        choice_algo = "de"
    algo = pxp.algo_factory(choice=choice_algo)
    pop_size = 30
    print(f"Using algorithm: {choice_algo}, pop_size: {pop_size}")
    t0range = [
        '2034-01-01 00:00:00.000',
        '2034-12-31 00:00:00.000'
    ]
    print(f"Using window {t0range[0][0:10]} ~ {t0range[1][0:10]}")
    c3_max = 90
    n_t0 = 300
    pop_list, prob_list = run_optim_depart(
        seq_key, algo, t0range=t0range, pop_size=pop_size, n_t0=n_t0,
        vinf_launch_max=np.sqrt(c3_max)
    )

    # get combined list of xs and fs
    for idx,pop_iter in enumerate(pop_list):
        if idx == 0:
            x_combined = pop_iter.get_x()
            f_combined = pop_iter.get_f()
        else:
            x_combined = np.concatenate((x_combined, pop_iter.get_x()),axis=0)
            f_combined = np.concatenate((f_combined, pop_iter.get_f()))

    # process porkchop
    n_leg = len(seq_key)-1
    porkchop_return = pxp.porkchop_process(
        prob_list[0], x_combined, f_combined, n_leg, direction="depart",
    )

    # create and save plot
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    im0 = ax.scatter(porkchop_return['t0_matplotlib'], porkchop_return['tf_matplotlib'], #*pk.DAY2YEAR, 
                         c=porkchop_return['dsm_total']/1e3, cmap='winter', s=15, marker='x')

    fig.colorbar(im0, label='Total DSM DV, km/s')
    ax.set_title(seq_name)
    ax.set(xlabel="Departure, year", ylabel="TOF, year")

    #plt.savefig("../notebooks/plots/seq_"+seq_name+".png")

    # get number of files+str(c3_max)
    filenames = os.listdir("../notebooks/optim_res_dep_c3_"+str(c3_max))
    n_data_already = 0
    for filename in filenames:
        if seq_name in filename:
            n_data_already += 1
    # save result array
    if args.data:
        choose_save = args.data
    else:
        choose_save = input("save (y/n)? [y]: ")

    if choose_save != "n":
        saved_file = "../notebooks/optim_res_dep_c3_"+str(c3_max)+"/seq_"+seq_name+"_"+str(n_data_already+1)
        np.save(saved_file, x_combined)
        print(f"Saved data at {saved_file}")
    # plot result for fun
    plt.show()