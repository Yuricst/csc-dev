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


def planet_int2str(planet_idx):
    if planet_idx == 3:
        name = "earth"
    elif planet_idx == 4:
        name = "mars"
    elif planet_idx == 5:
        name = "jupiter"
    elif planet_idx == 6:
        name = "saturn"
    return name


def run_optim_depart(seq_key, algo, pop_size=10):
    """Construct pygmo return transfer problem with MGA1DSM model"""
    # get solar system
    ssdict = pxp.solar_system_spice()

    tof = [
        [0.2/pk.DAY2YEAR, 6/pk.DAY2YEAR] for el in range(len(seq_key)-1)
    ]

    pop_list = []
    prob_list = []

    for i_window in tqdm(range(12*11), desc="scanning launch window"):
        t0_iter = [
            pk.epoch_from_string('2036-01-01 00:00:00.000').mjd2000 + i_window*30,
            pk.epoch_from_string('2036-01-01 00:00:00.000').mjd2000 + (i_window+1)*30
            
        ]

        # run problem
        seq = [ssdict[el] for el in seq_key]
        prob_iter = pxp.get_return_problem(seq, t0_iter, tof)
        pop = pg.population(prob=prob_iter, size=pop_size)
        pop = algo.evolve(pop)

        # store
        pop_list.append(pop)
        prob_list.append(prob_iter)

    return pop_list, prob_list



def run_optim_return(seq_key, algo, pop_size=10):
    """Construct pygmo return transfer problem with MGA1DSM model"""
    # get solar system
    ssdict = pxp.solar_system_spice()

    tof = [
        [0.2/pk.DAY2YEAR, 6/pk.DAY2YEAR] for el in range(len(seq_key)-1)
    ]

    pop_list = []
    prob_list = []

    for i_window in tqdm(range(12*11), desc="scanning launch window"):
        t0_iter = [
            pk.epoch_from_string('2044-01-01 00:00:00.000').mjd2000 + i_window*30,
            pk.epoch_from_string('2044-01-01 00:00:00.000').mjd2000 + (i_window+1)*30
            
        ]

        # run problem
        seq = [ssdict[el] for el in seq_key]
        prob_iter = pxp.get_return_problem(seq, t0_iter, tof)
        pop = pg.population(prob=prob_iter, size=pop_size)
        pop = algo.evolve(pop)

        # store
        pop_list.append(pop)
        prob_list.append(prob_iter)

    return pop_list, prob_list


def algo_factory(choice):
    """Generate pygmo algorithm class.

    For list of algorithms, see:
    https://esa.github.io/pygmo2/algorithms.html
    """
    if choice == "sade":
        algo = pg.algorithm(pg.sade(gen=4000))
    elif choice == "gaco":
        algo = pg.algorithm(pg.gaco(
            gen=200,
            ker=12,
            q=1.0,
            oracle=1e9,
            acc=0.0,
            threshold=1,
            n_gen_mark=7,
            impstop=100000,
            evalstop=100000,
            focus=0.0,
            memory=False,
        ))
    return algo



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run interplanetary transfer")
    parser.add_argument('-s', '--sequence', help='sequence of flyby: ')
    args = parser.parse_args()
    if args.sequence:
        seq_key = [
            planet_int2str(int(args.sequence[el])) for el in range(len(args.sequence))
        ]
    else:
        seq_key = ["saturn", "saturn", "earth"]

    # get sequence name string
    seq_name = ""
    for el in seq_key:
        seq_name += el + "-"
    seq_name = seq_name[:-1]

    print(f"Running sequence: {seq_name}")

    # tune algorithm
    choice_algo = "gaco"
    algo = algo_factory(choice=choice_algo)
    print(f"Using algorithm: {choice_algo}")
    #algo = pg.algorithm(pg.sade(gen=4000))
    #algo = pg.algorithm(pg.gaco(10, 13, 1.0, 1e9, 0.0, 1, 7, 100000, 100000, 0.0, False, 23))
    pop_size = 20
    pop_list, prob_list = run_optim_return(seq_key, algo, pop_size)

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
        prob_list[0], x_combined, f_combined, n_leg,
    )

    # create and save plot
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    im0 = ax.scatter(porkchop_return['t0_matplotlib'], porkchop_return['tof_total']*pk.DAY2YEAR, 
                         c=porkchop_return['dsm_total']/1e3, cmap='winter', s=15, marker='x')

    fig.colorbar(im0, label='Total DSM DV, km/s')
    ax.set_title(seq_name)
    ax.set(xlabel="Departure, year", ylabel="TOF, year")

    plt.savefig("../notebooks/plots/seq_"+seq_name+".png")

    # get number of files
    filenames = os.listdir("../notebooks/optim_res")
    n_data_already = 0
    for filename in filenames:
        if seq_name in filename:
            n_data_already += 1
    # save result array
    choose_save = input("save (y/n)? [y]: ")
    if choose_save != "n":
        np.save("../notebooks/optim_res/seq_"+seq_name+str(n_data_already+1), x_combined)


    # plot result for fun
    plt.show()
