"""
Helper functions around pygmo
"""

import pygmo as pg
import pykep as pk

from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_prop, AU, epoch
from pykep.planet import jpl_lp
from pykep.trajopt._lambert import lambert_problem_multirev
from math import pi, cos, sin, acos, log, sqrt
import numpy as np
from numpy.linalg import norm
from typing import Any, Dict, List, Tuple
from bisect import bisect_left


def algo_factory(choice, generation=2000):
    """Generate pygmo algorithm class.

    For list of algorithms, see:
    https://esa.github.io/pygmo2/overview.html#list-of-algorithms
    https://esa.github.io/pygmo2/algorithms.html

    Args:
        choice (str): name of algorithm

    Returns:
        (pg.algorithm): pygmo algorithm class
    """
    if choice == "sade":
        # Self-adaptive Differential Evolution.
        algo = pg.algorithm(pg.sade(
            gen=4000,
        ))
    elif choice == "sadequick":
        # Self-adaptive Differential Evolution.
        algo = pg.algorithm(pg.sade(
            gen=1000,
        ))
    elif choice == "gaco":
        algo = pg.algorithm(pg.gaco(
            gen=generation,
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
    elif choice == "mbh":
        gradient_method = pg.ipopt()
        gradient_method.set_numeric_option("tol", 1e-6) # Change the relative convergence tolerance
        gradient_method.set_numeric_option("max_cpu_time", 300)
        algo = pg.algorithm(pg.mbh(
            pg.algorithm(gradient_method), 
            stop=10, 
            perturb=0.1
        ))
        algo.set_verbosity(0)

    elif choice == "de":
        algo = pg.algorithm(pg.de(
            gen=generation, 
            F=0.8, 
            CR=0.9, 
            variant=2, 
            ftol=1e-06, 
            xtol=1e-06
        ))
    elif choice == "pso":
        algo = pg.algorithm(pg.pso(
            gen=generation, 
            omega=0.7298, eta1=2.05, eta2=2.05, 
            max_vel=0.5, variant=5, neighb_type=2, neighb_param=4, 
            memory=False
        ))
    # MO algorithms
    elif choice == "moead":
        algo = pg.algorithm(pg.moead(
            gen=generation, 
            weight_generation='grid', 
            decomposition='tchebycheff', 
            neighbours=20, CR=1, F=0.5, eta_m=20, realb=0.9, limit=2, 
            preserve_diversity=True
        ))
    elif choice == "maco":
        algo = pg.algorithm(pg.maco(
            gen=generation, 
            ker=25, q=1.0, threshold=1, n_gen_mark=7, 
            evalstop=100000, focus=0.0, memory=False,
        ))
    else:
        raise NotImlpementedError(f"algorithm of name {choice} is not defined!")
    return algo



# create mga1dsm inherited class
class mga_1dsm(pk.trajopt.mga_1dsm):
    """MGA1DSM class inherited from pykep"""
    def __init__(self, seq, t0, tof, vinf, multi_objective, add_vinf_dep, add_vinf_arr, tof_encoding):
        super().__init__(
            seq=seq,
            t0=t0,
            tof=tof,
            vinf=vinf,
            multi_objective=multi_objective,
            add_vinf_dep=add_vinf_dep,
            add_vinf_arr=add_vinf_arr,
            tof_encoding=tof_encoding
        )
        self._add_vinf_dep = add_vinf_dep
        self._add_vinf_arr = add_vinf_arr

    # provide gradients
    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

    # FIXME - inherit?
    def _compute_dvs(self, x: List[float]) -> Tuple[
        List[float], # DVs
        List[Any], # Lambert legs
        List[float], # T
        List[Tuple[List[float], List[float]]], # ballistic legs
        List[float], # epochs of ballistic legs
    ]:
        # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([0.0] * (self.n_legs + 1))
        for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(T[0:i]))
            r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
        ballistic_legs: List[Tuple[List[float],List[float]]] = []
        ballistic_ep: List[float] = []
        lamberts = []

        # 3 - We start with the first leg
        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        ballistic_legs.append((r_P[0], v0))
        ballistic_ep.append(t_P[0].mjd2000)
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu
        )

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        l = lambert_problem_multirev(v, lambert_problem(
                    r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]
        lamberts.append(l)

        ballistic_legs.append((r, v_beg_l))
        ballistic_ep.append(t_P[0].mjd2000 + x[4] * T[0])

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            ballistic_legs.append((r_P[i], v_out))
            ballistic_ep.append(t_P[i].mjd2000)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC
            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                                  self.common_mu, cw=False, max_revs=self.max_revs))
            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            lamberts.append(l)
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])

            ballistic_legs.append((r, v_beg_l))
            ballistic_ep.append(t_P[i].mjd2000 + x[8 + (i - 1) * 4] * T[i])

        # Last Delta-v
        if self._add_vinf_arr:
            DV[-1] = norm([a - b for a, b in zip(v_end_l, v_P[-1])])
            if self._orbit_insertion:
                # In this case we compute the insertion DV as a single pericenter
                # burn
                DVper = np.sqrt(DV[-1] * DV[-1] + 2 *
                                self._seq[-1].mu_self / self._rp_target)
                DVper2 = np.sqrt(2 * self._seq[-1].mu_self / self._rp_target -
                                self._seq[-1].mu_self / self._rp_target * (1. - self._e_target))
                DV[-1] = np.abs(DVper - DVper2)

        if self._add_vinf_dep:
            DV[0] += x[3]

        return (DV, lamberts, T, ballistic_legs, ballistic_ep)

    # redefine fitness
    def fitness(self, x):
        DV, _, T, _, _ = self._compute_dvs(x)
        if not self._multi_objective:
            return [sum(DV),]
        else:
            return (sum(DV), sum(T))

    # FIXME - inherit?
    def get_trajectory_components(self, x: List[float]):
        # trajectory components
        traj_comp = []
        dsm_info = []

        # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([0.0] * (self.n_legs + 1))
        for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(T[0:i]))
            r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
        ballistic_legs: List[Tuple[List[float],List[float]]] = []
        ballistic_ep: List[float] = []
        lamberts = []

        # 3 - We start with the first leg
        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        ballistic_legs.append((r_P[0], v0))
        ballistic_ep.append(t_P[0].mjd2000)
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)
        # STORE
        traj_comp.append(
            (r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)
        )

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        l = lambert_problem_multirev(v, lambert_problem(
                    r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]
        lamberts.append(l)
        dsm_info.append(
            (r, [a - b for a, b in zip(v_beg_l, v)])
        )

        ballistic_legs.append((r, v_beg_l))
        ballistic_ep.append(t_P[0].mjd2000 + x[4] * T[0])

        # STORE
        traj_comp.append(
            (r, v_beg_l, dt, self.common_mu)
        )


        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            ballistic_legs.append((r_P[i], v_out))
            ballistic_ep.append(t_P[i].mjd2000)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            # STORE
            traj_comp.append(
                (r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            )

            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC
            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                                  self.common_mu, cw=False, max_revs=self.max_revs))
            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            lamberts.append(l)
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])

            ballistic_legs.append((r, v_beg_l))
            ballistic_ep.append(t_P[i].mjd2000 + x[8 + (i - 1) * 4] * T[i])
            # STORE
            traj_comp.append(
                (r, v_beg_l, dt, self.common_mu)
            )
            dsm_info.append(
                (r, [a - b for a, b in zip(v_beg_l, v)])
            )

        

        # Last Delta-v
        if self._add_vinf_arr:
            DV[-1] = norm([a - b for a, b in zip(v_end_l, v_P[-1])])
            if self._orbit_insertion:
                # In this case we compute the insertion DV as a single pericenter
                # burn
                DVper = np.sqrt(DV[-1] * DV[-1] + 2 *
                                self._seq[-1].mu_self / self._rp_target)
                DVper2 = np.sqrt(2 * self._seq[-1].mu_self / self._rp_target -
                                self._seq[-1].mu_self / self._rp_target * (1. - self._e_target))
                DV[-1] = np.abs(DVper - DVper2)

        if self._add_vinf_dep:
            DV[0] += x[3]

        return traj_comp, dsm_info



class mga_1dsm_vinf_arr(pk.trajopt.mga_1dsm):
    """Multiobjective MGA1DSM class inherited from pykep, with penalty on arrival v-infinity"""
    def __init__(self, seq, t0, tof, vinf, add_vinf_dep, tof_encoding):
        super().__init__(
            seq=seq,
            t0=t0,
            tof=tof,
            vinf=vinf,
            add_vinf_dep=add_vinf_dep,
            add_vinf_arr=True,   # has to be True
            tof_encoding=tof_encoding
        )


    # FIXME - inherit?
    def _compute_dvs(self, x: List[float]) -> Tuple[
        List[float], # DVs
        List[Any], # Lambert legs
        List[float], # T
        List[Tuple[List[float], List[float]]], # ballistic legs
        List[float], # epochs of ballistic legs
    ]:
        # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([0.0] * (self.n_legs + 1))
        for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(T[0:i]))
            r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
        ballistic_legs: List[Tuple[List[float],List[float]]] = []
        ballistic_ep: List[float] = []
        lamberts = []

        # 3 - We start with the first leg
        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        ballistic_legs.append((r_P[0], v0))
        ballistic_ep.append(t_P[0].mjd2000)
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        l = lambert_problem_multirev(v, lambert_problem(
                    r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]
        lamberts.append(l)

        ballistic_legs.append((r, v_beg_l))
        ballistic_ep.append(t_P[0].mjd2000 + x[4] * T[0])

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            ballistic_legs.append((r_P[i], v_out))
            ballistic_ep.append(t_P[i].mjd2000)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC
            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                                  self.common_mu, cw=False, max_revs=self.max_revs))
            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            lamberts.append(l)
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])

            ballistic_legs.append((r, v_beg_l))
            ballistic_ep.append(t_P[i].mjd2000 + x[8 + (i - 1) * 4] * T[i])

        # Last Delta-v
        if self._add_vinf_arr:
            DV[-1] = norm([a - b for a, b in zip(v_end_l, v_P[-1])])
            if self._orbit_insertion:
                # In this case we compute the insertion DV as a single pericenter
                # burn
                DVper = np.sqrt(DV[-1] * DV[-1] + 2 *
                                self._seq[-1].mu_self / self._rp_target)
                DVper2 = np.sqrt(2 * self._seq[-1].mu_self / self._rp_target -
                                self._seq[-1].mu_self / self._rp_target * (1. - self._e_target))
                DV[-1] = np.abs(DVper - DVper2)

        if self._add_vinf_dep:
            DV[0] += x[3]

        return (DV, lamberts, T, ballistic_legs, ballistic_ep)

    # redefine fitness
    def fitness(self, x):
        DV, _, T, _, _ = self._compute_dvs(x)
        return [sum(DV[:-1]), DV[-1]]

    def get_nobj(self):
        return 2




# create mga1dsm inherited class
class mga_1dsm_fixed_vexcess(pk.trajopt.mga_1dsm):
    """MGA1DSM class inherited from pykep"""
    def __init__(
        self, 
        seq, t0, tof, 
        vinf,              # bounds
        multi_objective, 
        add_vinf_dep, 
        add_vinf_arr, 
        tof_encoding, 
        vexcess_dep=0.0, vexcess_arr=0.0, fb_angles_dep=None, fb_angles_arr=None
    ):
        super().__init__(
            seq=seq,
            t0=t0,
            tof=tof,
            vinf=vinf,
            multi_objective=multi_objective,
            add_vinf_dep=add_vinf_dep,
            add_vinf_arr=add_vinf_arr,
            tof_encoding=tof_encoding
        )
        self._add_vinf_dep = add_vinf_dep
        self._add_vinf_arr = add_vinf_arr

        self.vexcess_dep = vexcess_dep
        if fb_angles_dep is None:
            self.fb_angles_dep = [0.0, 0.0]
        else:
            self.fb_angles_dep = fb_angles_dep

        self.vexcess_arr = vexcess_arr
        if fb_angles_arr is None:
            self.fb_angles_arr = [0.0, 0.0]
        else:
            self.fb_angles_arr = fb_angles_arr

    # FIXME - inherit?
    def _compute_dvs(self, x: List[float]) -> Tuple[
        List[float], # DVs
        List[Any], # Lambert legs
        List[float], # T
        List[Tuple[List[float], List[float]]], # ballistic legs
        List[float], # epochs of ballistic legs
    ]:
        # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([0.0] * (self.n_legs + 1))
        for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(T[0:i]))

            # 2.1 - We append additional direction to first/last velocity vector
            if i == 0:
                r_P[i], v_P0 = self._seq[i].eph(t_P[i])
                #v_P0_norm = norm(v_P0)
                v_P[i] = (
                    # v_P0[0] + self.vexcess_dep*v_P0[0]/v_P0_norm,
                    # v_P0[1] + self.vexcess_dep*v_P0[1]/v_P0_norm,
                    # v_P0[2] + self.vexcess_dep*v_P0[2]/v_P0_norm
                    v_P0[0] + self.vexcess_dep*np.cos(self.fb_angles_dep[0])*np.cos(self.fb_angles_dep[1]),
                    v_P0[1] + self.vexcess_dep*np.sin(self.fb_angles_dep[0])*np.cos(self.fb_angles_dep[1]),
                    v_P0[2] + self.vexcess_dep*np.sin(self.fb_angles_dep[1]),
                )
            elif i == len(self._seq)-1:
                r_P[i], v_Pf = self._seq[i].eph(t_P[i])
                #v_P0_norm = norm(v_P0)
                v_P[i] = (
                    v_Pf[0] + self.vexcess_arr*np.cos(self.fb_angles_arr[0])*np.cos(self.fb_angles_arr[1]),
                    v_Pf[1] + self.vexcess_arr*np.sin(self.fb_angles_arr[0])*np.cos(self.fb_angles_arr[1]),
                    v_Pf[2] + self.vexcess_arr*np.sin(self.fb_angles_arr[1]),
                )
            else:
                r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
        ballistic_legs: List[Tuple[List[float],List[float]]] = []
        ballistic_ep: List[float] = []
        lamberts = []

        # 3 - We start with the first leg
        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        ballistic_legs.append((r_P[0], v0))
        ballistic_ep.append(t_P[0].mjd2000)
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        l = lambert_problem_multirev(v, lambert_problem(
                    r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]
        lamberts.append(l)

        ballistic_legs.append((r, v_beg_l))
        ballistic_ep.append(t_P[0].mjd2000 + x[4] * T[0])

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            ballistic_legs.append((r_P[i], v_out))
            ballistic_ep.append(t_P[i].mjd2000)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC
            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                                  self.common_mu, cw=False, max_revs=self.max_revs))
            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            lamberts.append(l)
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])

            ballistic_legs.append((r, v_beg_l))
            ballistic_ep.append(t_P[i].mjd2000 + x[8 + (i - 1) * 4] * T[i])

        # Last Delta-v
        if self._add_vinf_arr:
            DV[-1] = norm([a - b for a, b in zip(v_end_l, v_P[-1])])
            if self._orbit_insertion:
                # In this case we compute the insertion DV as a single pericenter
                # burn
                DVper = np.sqrt(DV[-1] * DV[-1] + 2 *
                                self._seq[-1].mu_self / self._rp_target)
                DVper2 = np.sqrt(2 * self._seq[-1].mu_self / self._rp_target -
                                self._seq[-1].mu_self / self._rp_target * (1. - self._e_target))
                DV[-1] = np.abs(DVper - DVper2)

        if self._add_vinf_dep:
            DV[0] += x[3]

        return (DV, lamberts, T, ballistic_legs, ballistic_ep)

    # redefine fitness
    def fitness(self, x):
        DV, _, T, _, _ = self._compute_dvs(x)
        return [sum(DV),]


    # redefine pretty
    def pretty(self, x):
        """
        prob.pretty(x)
        - x: encoded trajectory
        Prints human readable information on the trajectory represented by the decision vector x
        Example::
          print(prob.pretty(x))
        """
       # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([0.0] * (self.n_legs + 1))
        # for i in range(len(self._seq)):
        #     t_P[i] = epoch(x[0] + sum(T[0:i]))
        #     r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
        for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(T[0:i]))

            # 2.1 - We append additional direction to first/last velocity vector
            if i == 0:
                r_P[i], v_P0 = self._seq[i].eph(t_P[i])
                #v_P0_norm = norm(v_P0)
                v_P[i] = (
                    # v_P0[0] + self.vexcess_dep*v_P0[0]/v_P0_norm,
                    # v_P0[1] + self.vexcess_dep*v_P0[1]/v_P0_norm,
                    # v_P0[2] + self.vexcess_dep*v_P0[2]/v_P0_norm
                    v_P0[0] + self.vexcess_dep*np.cos(self.fb_angles_dep[0])*np.cos(self.fb_angles_dep[1]),
                    v_P0[1] + self.vexcess_dep*np.sin(self.fb_angles_dep[0])*np.cos(self.fb_angles_dep[1]),
                    v_P0[2] + self.vexcess_dep*np.sin(self.fb_angles_dep[1]),
                )
            elif i == len(self._seq)-1:
                r_P[i], v_Pf = self._seq[i].eph(t_P[i])
                #v_P0_norm = norm(v_P0)
                v_P[i] = (
                    v_Pf[0] + self.vexcess_arr*np.cos(self.fb_angles_arr[0])*np.cos(self.fb_angles_arr[1]),
                    v_Pf[1] + self.vexcess_arr*np.sin(self.fb_angles_arr[0])*np.cos(self.fb_angles_arr[1]),
                    v_Pf[2] + self.vexcess_arr*np.sin(self.fb_angles_arr[1]),
                )
            else:
                r_P[i], v_P[i] = self._seq[i].eph(t_P[i])

        # 3 - We start with the first leg
        print("First Leg: " + self._seq[0].name + " to " + self._seq[1].name)
        print("Departure: " + str(t_P[0]) +
              " (" + str(t_P[0].mjd2000) + " mjd2000) ")
        print("Duration: " + str(T[0]) + "days")
        print("VINF: " + str(x[3] / 1000) + " km/sec")

        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)

        print("DSM after " + str(x[4] * T[0]) + " days")

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        l = lambert_problem_multirev(v, lambert_problem(
            r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])
        print("DSM magnitude: " + str(DV[0]) + "m/s")

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            print("\nleg no. " + str(i + 1) + ": " +
                  self._seq[i].name + " to " + self._seq[i + 1].name)
            print("Duration: " + str(T[i]) + "days")
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            print(
                "Fly-by epoch: " + str(t_P[i]) + " (" + str(t_P[i].mjd2000) + " mjd2000) ")
            print(
                "Fly-by radius: " + str(x[7 + (i - 1) * 4]) + " planetary radii")
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            print("DSM after " + str(x[8 + (i - 1) * 4] * T[i]) + " days")
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC
            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                                self.common_mu, cw=False, max_revs=self.max_revs))
            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])
            print("DSM magnitude: " + str(DV[i]) + "m/s")

        # Last Delta-v
        print("\nArrival at " + self._seq[-1].name)
        DV[-1] = norm([a - b for a, b in zip(v_end_l, v_P[-1])])
        print(
            "Arrival epoch: " + str(t_P[-1]) + " (" + str(t_P[-1].mjd2000) + " mjd2000) ")
        print("Arrival Vinf: " + str(DV[-1]) + "m/s")
        if self._orbit_insertion:
            # In this case we compute the insertion DV as a single pericenter
            # burn
            DVper = np.sqrt(DV[-1] * DV[-1] + 2 *
                            self._seq[-1].mu_self / self._rp_target)
            DVper2 = np.sqrt(2 * self._seq[-1].mu_self / self._rp_target -
                            self._seq[-1].mu_self / self._rp_target * (1. - self._e_target))
            DVinsertion = np.abs(DVper - DVper2)
            print("Insertion DV: " + str(DVinsertion) + "m/s")

        print("Total mission time: " + str(sum(T) / 365.25) + " years (" + str(sum(T)) + " days)")

    # Plot of the trajectory
    def plot(self, x, ax = None, camera = None, get_legend = True):
        """
        ax = prob.plot(x, ax
        - x: encoded trajectory
        - ax: matplotlib axis where to plot. If None figure and axis will be created
        - [out] ax: matplotlib axis where to plot
        Plots the trajectory represented by a decision vector x on the 3d axis ax
        Example::
          ax = prob.plot(x)
        """
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

        if ax is None:
            mpl.rcParams['legend.fontsize'] = 10
            fig = plt.figure()
            axis = fig.gca(projection='3d')
        else:
            axis = ax

        axis.scatter(0, 0, 0, color='y')

        if camera is not None:
            axis.view_init(elev=camera[0], azim=camera[1])

        # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([None] * (self.n_legs + 1))

        for i, planet in enumerate(self._seq): #for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(T[0:i]))

            # 2.1 - We append additional direction to first/last velocity vector
            if i == 0:
                r_P[i], v_P0 = self._seq[i].eph(t_P[i])
                #v_P0_norm = norm(v_P0)
                v_P[i] = (
                    # v_P0[0] + self.vexcess_dep*v_P0[0]/v_P0_norm,
                    # v_P0[1] + self.vexcess_dep*v_P0[1]/v_P0_norm,
                    # v_P0[2] + self.vexcess_dep*v_P0[2]/v_P0_norm
                    v_P0[0] + self.vexcess_dep*np.cos(self.fb_angles_dep[0])*np.cos(self.fb_angles_dep[1]),
                    v_P0[1] + self.vexcess_dep*np.sin(self.fb_angles_dep[0])*np.cos(self.fb_angles_dep[1]),
                    v_P0[2] + self.vexcess_dep*np.sin(self.fb_angles_dep[1]),
                )
            elif i == len(self._seq)-1:
                r_P[i], v_Pf = self._seq[i].eph(t_P[i])
                #v_P0_norm = norm(v_P0)
                v_P[i] = (
                    v_Pf[0] + self.vexcess_arr*np.cos(self.fb_angles_arr[0])*np.cos(self.fb_angles_arr[1]),
                    v_Pf[1] + self.vexcess_arr*np.sin(self.fb_angles_arr[0])*np.cos(self.fb_angles_arr[1]),
                    v_Pf[2] + self.vexcess_arr*np.sin(self.fb_angles_arr[1]),
                )
            else:
                r_P[i], v_P[i] = self._seq[i].eph(t_P[i])

            plot_planet(planet, t0=t_P[i], color=(
                0.8, 0.6, 0.8), legend=get_legend, units=AU, axes=axis, N=150)

        # for i, planet in enumerate(self._seq): #for i in range(len(self._seq)):
        #     t_P[i] = epoch(x[0] + sum(T[0:i]))

        #     # 2.1 - We append additional direction to first velocity vector
        #     if i == 0:
        #         r_P[i], v_P0 = self._seq[i].eph(t_P[i])
        #         v_P0_norm = norm(v_P0)
        #         v_P[i] = (
        #             v_P0[0] + self.vexcess_dep*v_P0[0]/v_P0_norm,
        #             v_P0[1] + self.vexcess_dep*v_P0[1]/v_P0_norm,
        #             v_P0[2] + self.vexcess_dep*v_P0[2]/v_P0_norm
        #         )
        #         plot_planet(planet, t0=t_P[i], color=(
        #         0.8, 0.6, 0.8), legend=get_legend, units=AU, axes=axis, N=150)
        #     else:
        #         r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
        #         plot_planet(planet, t0=t_P[i], color=(
        #         0.8, 0.6, 0.8), legend=get_legend, units=AU, axes=axis, N=150)

        # 3 - We start with the first leg
        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)

        plot_kepler(r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu,
                    N=100, color='b', units=AU, axes=axis)

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        
        l = lambert_problem_multirev(v, lambert_problem(
            r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        
        plot_lambert(l, sol=0, color='r', units=AU, axes=axis)
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            plot_kepler(r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC,
                        self.common_mu, N=100, color='b', units=AU, axes=axis)
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC

            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                self.common_mu, cw=False, max_revs=self.max_revs))

            plot_lambert(l, sol=0, color='r', legend=False,
                         units=AU, N=1000, axes=axis)

            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])
        plt.show()
        return fig, axis

    # Plot of the trajectory
    def plot_2D(self, x, ax = None, figsize=(8,6), camera=None):
        """
        ax = prob.plot(x, ax
        - x: encoded trajectory
        - ax: matplotlib axis where to plot. If None figure and axis will be created
        - [out] ax: matplotlib axis where to plot
        Plots the trajectory represented by a decision vector x on the 3d axis ax
        Example::
          ax = prob.plot(x)
        """
        import matplotlib as mpl
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from pykep.orbit_plots import plot_planet, plot_lambert, plot_kepler

        if ax is None:
            mpl.rcParams['legend.fontsize'] = 10
            # fig = plt.figure()
            # axis = fig.gca(projection='2d')
            fig, axis = plt.subplots(1,1,figsize=figsize)
        else:
            axis = ax

        if camera is not None:
            axis.view_init(elev=ele, azim=azm)

        axis.scatter(0, 0, color='y')

        # 1 -  we 'decode' the chromosome recording the various times of flight
        # (days) in the list T and the cartesian components of vinf
        T, Vinfx, Vinfy, Vinfz = self._decode_times_and_vinf(x)

        # 2 - We compute the epochs and ephemerides of the planetary encounters
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([None] * (self.n_legs + 1))

        # for i, planet in enumerate(self._seq):
        #     t_P[i] = epoch(x[0] + sum(T[0:i]))
        #     r_P[i], v_P[i] = planet.eph(t_P[i])
        #     plot_planet(planet, t0=t_P[i], color=(
        #         0.8, 0.6, 0.8), legend=True, units=AU, axes=axis, N=150)
        # for i in range(len(self._seq)):
        #     t_P[i] = epoch(x[0] + sum(T[0:i]))
        #     r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
        for i, planet in enumerate(self._seq): #for i in range(len(self._seq)):
            t_P[i] = epoch(x[0] + sum(T[0:i]))

            # 2.1 - We append additional direction to first velocity vector
            if i == 0:
                r_P[i], v_P0 = self._seq[i].eph(t_P[i])
                v_P0_norm = norm(v_P0)
                v_P[i] = (
                    v_P0[0] + self.vexcess_dep*v_P0[0]/v_P0_norm,
                    v_P0[1] + self.vexcess_dep*v_P0[1]/v_P0_norm,
                    v_P0[2] + self.vexcess_dep*v_P0[2]/v_P0_norm
                )
                plot_planet(planet, t0=t_P[i], color=(
                0.8, 0.6, 0.8), legend=True, units=AU, axes=axis, N=150)
            else:
                r_P[i], v_P[i] = self._seq[i].eph(t_P[i])
                plot_planet(planet, t0=t_P[i], color=(
                0.8, 0.6, 0.8), legend=True, units=AU, axes=axis, N=150)

        # 3 - We start with the first leg
        v0 = [a + b for a, b in zip(v_P[0], [Vinfx, Vinfy, Vinfz])]
        r, v = propagate_lagrangian(
            r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu)

        plot_kepler(r_P[0], v0, x[4] * T[0] * DAY2SEC, self.common_mu,
                    N=100, color='b', units=AU, axes=axis)

        # Lambert arc to reach seq[1]
        dt = (1 - x[4]) * T[0] * DAY2SEC
        
        l = lambert_problem_multirev(v, lambert_problem(
            r, r_P[1], dt, self.common_mu, cw=False, max_revs=self.max_revs))
        
        plot_lambert(l, sol=0, color='r', units=AU, axes=axis)
        v_end_l = l.get_v2()[0]
        v_beg_l = l.get_v1()[0]

        # First DSM occuring at time nu1*T1
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # 4 - And we proceed with each successive leg
        for i in range(1, self.n_legs):
            # Fly-by
            v_out = fb_prop(v_end_l, v_P[i], x[
                            7 + (i - 1) * 4] * self._seq[i].radius, x[6 + (i - 1) * 4], self._seq[i].mu_self)
            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC, self.common_mu)
            plot_kepler(r_P[i], v_out, x[8 + (i - 1) * 4] * T[i] * DAY2SEC,
                        self.common_mu, N=100, color='b', units=AU, axes=axis)
            # Lambert arc to reach Earth during (1-nu2)*T2 (second segment)
            dt = (1 - x[8 + (i - 1) * 4]) * T[i] * DAY2SEC

            l = lambert_problem_multirev(v, lambert_problem(r, r_P[i + 1], dt,
                self.common_mu, cw=False, max_revs=self.max_revs))

            plot_lambert(l, sol=0, color='r', legend=False,
                         units=AU, N=1000, axes=axis)

            v_end_l = l.get_v2()[0]
            v_beg_l = l.get_v1()[0]
            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])
        plt.show()
        return axis

    def get_extra_info(self):
        return ("\n\t Sequence: " + [pl.name for pl in self._seq].__repr__() +
                "\n\t Add launcher vinf to the objective?: " + self._add_vinf_dep.__repr__() +
                "\n\t Add final vinf to the objective?: " + self._add_vinf_arr.__repr__())
       
