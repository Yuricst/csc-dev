"""
Methods for generating pykep problems
"""

import numpy as np
import pykep as pk
import datetime
from tqdm.auto import tqdm

import copy

from ._pygmo_helper import mga_1dsm, mga_1dsm_vinf_arr


def get_depart_problem(seq, t0, tof, vinf_launch_max=4.5):
    """Get return problem

    Key settings:
        - no launch v-infinity
        - launch v-infinity is not part of objective
        - arrival v-infinity is part of objective
    """
    prob_head = mga_1dsm(
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



def get_return_problem(seq, t0, tof, vinf_first=2.0, mo=False):
    """Get return problem

    Key settings:
        - no launch v-infinity
        - launch v-infinity is part of objective
        - arrival v-infinity is part of objective
    """
    if mo is False:
        prob_return = mga_1dsm(
            seq = seq,
            t0 = t0, 
            tof = tof,
            vinf = [0.0, vinf_first],  # no launch v-infinity
            multi_objective = False, 
            add_vinf_dep = True, 
            add_vinf_arr = True,
            tof_encoding = 'direct'
        )
    else:
        prob_return = mga_1dsm_vinf_arr(
            seq = seq,
            t0 = t0, 
            tof = tof,
            vinf = [0.0, vinf_first],  # no launch v-infinity
            add_vinf_dep = True, 
            tof_encoding = 'direct'
        )
    return prob_return



# def iterate_launch_windows(seq, t0, tof, algo, pop_size):
#     prob_iter = get_return_problem(seq, t0, tof)
#     pop = pg.population(prob=prob_iter, size=pop_size)
#     pop = algo.evolve(pop)
#     return prob_iter, pop



def porkchop_process(
        prob_iter, 
        pop_x, 
        pop_f, 
        n_leg, 
        direction, 
        entry_altitude=200.0
    ):
    # extract from decision vector
    data = {
        "t0": [],
        "t0_matplotlib": [],
        "tf_matplotlib": [],
        "tof_total": [],
        "dsm_total": [],
        "dv_total": [],
        "launch_deltaV": [],
        "arrival_deltaV": [],
        "entry_v": [],
        "f": [],
        "x": [],
        "idx": [],
    }
    # create two problems
    if direction=="depart":
        prob_no_vinf_dep = copy.deepcopy(prob_iter)
        prob_no_vinf_dep._add_vinf_arr = False

        prob_with_vinf_dep = copy.deepcopy(prob_iter)
        prob_with_vinf_dep._add_vinf_arr = True

    elif direction == "return":
        prob_no_vinf_arr = copy.deepcopy(prob_iter)
        prob_no_vinf_arr._add_vinf_arr = False

        prob_with_vinf_arr = copy.deepcopy(prob_iter)
        prob_with_vinf_arr._add_vinf_arr = True


    for idx, x in tqdm(enumerate(pop_x), total=len(pop_x)):
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

        # arrival epoch
        data["tf_matplotlib"].append(
            np.datetime64(
                datetime.datetime.strptime(
                    pk.epoch(x[0]+tof_total).__str__()[0:11], 
                    '%Y-%b-%d'
                ).strftime('%Y-%m-%d')
            )
        )

        # objective
        data["f"].append(pop_f[idx])

        if direction=="depart":
            dvs1, _, _, _, _ = prob_no_vinf_dep._compute_dvs(x)
            dsm_total = sum(dvs1)
            dvs2, _, _, _, _ = prob_with_vinf_dep._compute_dvs(x)
            dv_total = sum(dvs2)

            # launch delta-V
            launch_deltaV = dv_total - dsm_total

        elif direction == "return":
            dvs1, _, _, _, _ = prob_no_vinf_arr._compute_dvs(x)
            dsm_total = sum(dvs1)
            dvs2, _, _, _, _ = prob_with_vinf_arr._compute_dvs(x)
            dv_total = sum(dvs2)

            # arrival delta-V
            arrival_deltaV = dv_total - dsm_total

        else:
            raise NotImplementedError("direction should be depart or return!")
        
        # arrival delta-V
        arrival_deltaV = dv_total - dsm_total
        
        # store
        data["dsm_total"].append(dsm_total)
        data["dv_total"].append(dv_total)
        data["idx"].append(idx)

        if direction == "depart":
            data["launch_deltaV"].append(launch_deltaV)
        else:
            data["arrival_deltaV"].append(arrival_deltaV)
            entry_v = np.sqrt(arrival_deltaV**2 + 2*pk.MU_EARTH / (1e3*(6378.0 + entry_altitude)))
            data["entry_v"].append(entry_v)
        
    # convert to numpy array
    data["t0"] = np.array(data["t0"])
    data["tof_total"] = np.array(data["tof_total"])
    data["dsm_total"] = np.array(data["dsm_total"])
    data["dv_total"]  = np.array(data["dv_total"])
    data["launch_deltaV"] = np.array(data["launch_deltaV"])
    data["arrival_deltaV"] = np.array(data["arrival_deltaV"])
    data["entry_v"] = np.array(data["entry_v"])
    data["f"] = np.array(data["f"])
    return data