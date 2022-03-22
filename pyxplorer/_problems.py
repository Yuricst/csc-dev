"""
Methods for generating pykep problems
"""

import numpy as np
import pykep as pk
import datetime


def get_depart_problem(seq, t0, tof, vinf_launch_max=3.0):
	"""Get return problem"""
	prob_head = pk.trajopt.mga_1dsm(
        seq = seq,
        t0 = t0,
        tof=tof,
        vinf = [0.0, vinf_launch_max], 
        multi_objective = False, 
        add_vinf_dep = False, 
        add_vinf_arr = True,
        tof_encoding = 'direct'
    )
	return prob_head



def get_return_problem(seq, t0, tof):
	"""Get return problem"""
	prob_return = pk.trajopt.mga_1dsm(
	    seq = seq,
	    t0 = t0, 
	    tof = tof,
	    vinf = [0.0, 0.0], 
	    multi_objective = False, 
	    add_vinf_dep = True, 
	    add_vinf_arr = True,
	    tof_encoding = 'direct'
	)
	return prob_return



# def iterate_launch_windows(seq, t0, tof, algo, pop_size):
#     prob_iter = get_return_problem(seq, t0, tof)
#     pop = pg.population(prob=prob_iter, size=pop_size)
#     pop = algo.evolve(pop)
#     return prob_iter, pop



def porkchop_process(prob_iter, pop_x, pop_f, n_leg):
    # extract from decision vector
    data = {
        "t0": [],
        "t0_matplotlib": [],
        "tof_total": [],
        "dsm_total": [],
        "arrival_deltaV": [],
        "f": [],
        "x": [],
        "idx": [],
    }
    for idx, x in enumerate(pop_x):
        data["x"].append(x)
        data["t0"].append(x[0])
        data["t0_matplotlib"].append(
        	np.datetime64(
        		datetime.datetime.strptime(
        			pk.epoch(x[0]).__str__()[0:11], 
        			'%Y-%b-%d'
        		).strftime('%Y-%m-%d')
        	)
    	)

        # get tof
        tof_total = x[5]
        for i_leg in range(n_leg-1):
            tof_total += x[5+4*(i_leg+1)]
        data["tof_total"].append(tof_total)
        # objective
        data["f"].append(pop_f[idx])
        
        # compute objective without and with arrival delta-V
        prob_iter._add_vinf_arr = False
        dsm_total = prob_iter.fitness(x)[0]
        prob_iter._add_vinf_arr = True
        dv_total = prob_iter.fitness(x)[0]
        
        # arrival delta-V
        arrival_deltaV = dv_total - dsm_total
        
        # store
        data["dsm_total"].append(dsm_total)
        data["arrival_deltaV"].append(arrival_deltaV)
        data["idx"].append(idx)
        
    # convert to numpy array
    data["t0"] = np.array(data["t0"])
    data["tof_total"] = np.array(data["tof_total"])
    data["dsm_total"] = np.array(data["dsm_total"])
    data["arrival_deltaV"] = np.array(data["arrival_deltaV"])
    data["f"] = np.array(data["f"])
    return data