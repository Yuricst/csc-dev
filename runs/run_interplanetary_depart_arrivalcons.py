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
import time 

import sys
sys.path.append("../")
import pyxplorer as pxp

pk.util.load_spice_kernel(os.path.join(os.environ.get("SPICE"), "spk", "de440.bsp"))
pk.util.load_spice_kernel(os.path.join(os.environ.get("SPICE"), "lsk", "naif0012.tls"))



def run(seq_key, t0, tof, vinf_launch_max, tf_min, tf_max, generation, pop_size):
    # measure time
    tstart = time.time()

    # get solar system
    ssdict = pxp.solar_system_spice()
    seq = [ssdict[el] for el in seq_key]

    # construct problem
    prob = pxp.mga_1dsm_tf_con(
        seq,
        t0,
        tof,
        [0.95*vinf_launch_max, vinf_launch_max],
        False,
        add_vinf_dep=False,
        add_vinf_arr=True,
        tof_encoding="direct",
        tf_min=tf_min,
        tf_max=tf_max,
    )

    algo = pxp.algo_factory(choice=choice_algo, generation=generation)
    print(f"Algorithm: {choice_algo}, pop_size: {pop_size}, generation: {generation}")
    algo.set_verbosity(1000)
    pop = pg.population(prob=prob, size=pop_size)
    pop = algo.evolve(pop)
    tend = time.time()
    print(f"Took {(tend-tstart)/60:3.4f} minutes")
    return prob, pop



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run interplanetary transfer")
    parser.add_argument('-n', '--num_rev', help='sequence of flyby: ')
    parser.add_argument('-d', '--data', help='store data to file: ')
    parser.add_argument('-p', '--pop_size', help='population size: ')
    parser.add_argument('-g', '--generation', help='generation size: ')
    parser.add_argument('-a', '--algorithm', help='choice of algorithm ')
    parser.add_argument('-m', '--multiobjective', help='multiobjective ')
    parser.add_argument('-s', '--sequence', help='sequence of flyby: ')
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


    t0 = [
        pk.epoch_from_string('2032-03-01 00:00:00.000').mjd2000,
        pk.epoch_from_string('2039-01-01 00:00:00.000').mjd2000
    ]
    c3_max = 90
    vinf_launch_max = np.sqrt(c3_max)  # fixed

    tof = [
        [3/pk.DAY2YEAR, 4.0/pk.DAY2YEAR],
        [3/pk.DAY2YEAR, 7.5/pk.DAY2YEAR],
    ]
    tf_min = pk.epoch_from_string('2044-06-01 00:00:00.000').mjd2000
    tf_max = pk.epoch_from_string('2044-08-01 00:00:00.000').mjd2000
    prob, pop = run(
        seq_key, t0, tof, vinf_launch_max, tf_min, tf_max,
        generation, pop_size
    )


    # process porkchop
    n_leg = len(seq_key)-1
    porkchop_return = pxp.porkchop_process(
        prob, pop.get_x(), pop.get_f(), n_leg, direction="depart",
    )

    # create and save plot
    plt.rcParams["font.size"] = 12
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    im0 = ax.scatter(porkchop_return['t0_matplotlib'], 
        porkchop_return['tf_matplotlib'], #*pk.DAY2YEAR, 
        c=porkchop_return['dsm_total']/1e3, cmap='winter', s=15, marker='x'
    )

    fig.colorbar(im0, label='Total DSM DV, km/s')
    ax.set_title(seq_name)
    ax.set(xlabel="Departure, year", ylabel="Arrival year")

    #plt.savefig("../notebooks/plots/seq_"+seq_name+".png")

    # get number of files+str(c3_max)
    filenames = os.listdir("../notebooks/optim_res_dep_c3_"+str(c3_max)+"_arrc")
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
        saved_file = "../notebooks/optim_res_dep_c3_"+str(c3_max)+"_arrc"+"/seq_"+seq_name+"_"+str(n_data_already+1)
        np.save(saved_file, pop.get_x())
        print(f"Saved data at {saved_file}")
    # plot result for fun
    plt.show()