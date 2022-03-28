"""
Custom VILT problem
"""


from pykep.core import epoch, DAY2SEC, MU_SUN, lambert_problem, propagate_lagrangian, fb_prop, AU, epoch
from pykep.planet import jpl_lp
from pykep.trajopt._lambert import lambert_problem_multirev
from math import pi, cos, sin, acos, log, sqrt
import numpy as np
from numpy.linalg import norm
from typing import Any, Dict, List, Tuple
from bisect import bisect_left


import pykep as pk
import numpy.linalg as la
from numpy.linalg import norm

from pykep import DAY2SEC, propagate_lagrangian, lambert_problem
from pykep.trajopt._lambert import lambert_problem_multirev


class mga_1dsm_vilt:
    def __init__(self, t0, vinf, seq, tof, common_mu, separate_objectives=False):
        assert len(tof)+1 == len(seq), "Give me all bodies visitd (including first one)"
        self.t0 = t0
        self.vinf = vinf
        self.seq = seq   # still give first body!
        self.tof = tof   # bounds
        self.common_mu = common_mu
        self.separate_objectives = separate_objectives

        # number of legs
        self.n_legs = len(seq)-1
        self.max_revs = 4

        # store problem bounds
        self.__get_bounds()


    def evaluate_trajectory(self, x):
        # unpack x
        x_per_legs = []
        for idx in range(self.n_legs):
            x_per_legs.append(
                x[4*idx:4*idx+4]
            )

        # storage
        t_P = list([None] * (self.n_legs + 1))
        r_P = list([None] * (self.n_legs + 1))
        v_P = list([None] * (self.n_legs + 1))
        DV = list([0.0] * (self.n_legs + 1))
        tofs = []
        traj_comp = []
        dsm_info = []

        # first leg
        alpha, delta, eta1, tof1 = x_per_legs[0]
        #print(x_per_legs[0])

        # compute initial position from v-infinity
        r_P[0], v_P[0] = self.seq[0].eph(self.t0 + tof1*DAY2SEC)
        v_start = [
            v_P[0][0] + self.vinf*np.cos(alpha)*np.cos(delta),
            v_P[0][1] + self.vinf*np.sin(alpha)*np.cos(delta),
            v_P[0][2] + self.vinf*np.sin(delta)
        ]

        # first leg
        r_P[1], v_P[1] = self.seq[1].eph(self.t0 + tof1*DAY2SEC)

        # s/c propagation before the DSM
        r, v = propagate_lagrangian(
            r_P[0], v_start, eta1 * tof1 * DAY2SEC, self.common_mu
        )

        # Lambert arc to reach Earth during (1-eta)*tof (second segment)
        dt = (1 - eta1) * tof1* DAY2SEC
        l = lambert_problem_multirev(v, 
            lambert_problem(
                r, r_P[1], dt,
                self.common_mu, cw=False, max_revs=self.max_revs
            )
        )
        v_beg_l = l.get_v1()[0]
        v_end_l = l.get_v2()[0]

        # DSM occuring at time nu2*T2
        DV[0] = norm([a - b for a, b in zip(v_beg_l, v)])

        # STORE
        traj_comp.append(
            (r_P[0], v_start, eta1 * tof1 * DAY2SEC, self.common_mu)
        )
        traj_comp.append(
            (r, v_beg_l, (1-eta1) * tof1 * DAY2SEC, self.common_mu)
        )
        dsm_info.append(
            (r, [a - b for a, b in zip(v_beg_l, v)])
        )

        t_running = self.t0 + tof1

        # subsequent legs, if any
        for i in range(1, self.n_legs):
            # unpack
            beta, rprv, eta, tof = x_per_legs[i]

            # get next planet position
            r_P[i+1], v_P[i+1] = self.seq[i+1].eph(t_running+tof)
            t_running += tof

            # fly-by from last leg --- fb_prop(v,v_pla,rp,beta,mu)
            v_out = fb_prop(
                v_end_l, 
                v_P[i], 
                rprv * self.seq[i].radius, 
                beta, 
                self.seq[i].mu_self
            )

            # s/c propagation before the DSM
            r, v = propagate_lagrangian(
                r_P[i], v_out, eta * tof * DAY2SEC, self.common_mu)

            # Lambert arc to reach Earth during (1-eta)*tof (second segment)
            dt = (1 - eta) * tof * DAY2SEC
            #print(f"r: {r}, r_P: {r_P[i + 1]}, dt: {dt}, x: {x_per_legs[i]}")
            l = lambert_problem_multirev(v, 
                lambert_problem(
                    r, r_P[i + 1], dt,
                    self.common_mu, cw=False, max_revs=self.max_revs
                )
            )
            v_beg_l = l.get_v1()[0]
            v_end_l = l.get_v2()[0]

            # DSM occuring at time nu2*T2
            DV[i] = norm([a - b for a, b in zip(v_beg_l, v)])

            # STORE
            traj_comp.append(
                (r_P[i], v_out, eta * tof * DAY2SEC, self.common_mu)
            )
            traj_comp.append(
                (r, v_beg_l, (1-eta) * tof * DAY2SEC, self.common_mu)
            )
            dsm_info.append(
                (r, [a - b for a, b in zip(v_beg_l, v)])
            )

        # get final v-infinity
        vinf_final = [
            v_P[-1][0] - v_end_l[0],
            v_P[-1][1] - v_end_l[1],
            v_P[-1][2] - v_end_l[2],
        ]

        return DV, r_P, v_P, vinf_final, traj_comp, dsm_info


    # Lower and Upper bounds on x
    def __get_bounds(self):
        self.lb = [-120*np.pi/180, -2*np.pi, 0.1, self.tof[0][0]]
        self.ub = [ 120*np.pi/180,  2*np.pi, 0.9, self.tof[0][1]]

        for i in range(self.n_legs-1):
            # append beta
            self.lb.append(-2*np.pi)
            self.ub.append( 2*np.pi)
            # rprv
            self.lb.append(self.seq[i+1].safe_radius/self.seq[i+1].radius)
            self.ub.append(100)
            # eta
            self.lb.append(0.1)
            self.ub.append(0.9)
            # tof
            self.lb.append(self.tof[i+1][0])
            self.ub.append(self.tof[i+1][1])
        return

    def get_bounds(self):
        return (self.lb, self.ub)


    def fitness(self, x):
        DV, _, _, vinf_final, _, _ = self.evaluate_trajectory(x)
        if self.separate_objectives == False:
            f_vector = [sum(DV) + np.linalg.norm(vinf_final),]
        else:
            f_vector = [sum(DV), np.linalg.norm(vinf_final),]
        return f_vector

    def get_nobj(self):
        return self.separate_objectives + 1

    def pretty(self, x):
        # unpack x
        x_per_legs = []
        for idx in range(self.n_legs):
            x_per_legs.append(
                x[4*idx:4*idx+4]
            )

        # compute tof
        tofs = []
        for x in x_per_legs:
            tofs.append(x[3])

        # evaluate trajectory
        DV, _, _, vinf_final, _, _ = self.evaluate_trajectory(x)
        print(f"TOF:                {np.linalg.norm(tofs)}")
        print(f"DSM Cost:           {np.linalg.norm(DV)}")
        print(f"Arrival v-inf Cost: {np.linalg.norm(vinf_final)}")
        print(f"Total Cost:         {np.linalg.norm(vinf_final) + np.linalg.norm(DV)}")
        return



# create mga1dsm inherited class
class mga_1dsm_tf_con(pk.trajopt.mga_1dsm):
    """MGA1DSM class inherited from pykep"""
    def __init__(
            self, 
            seq, t0, tof, vinf, multi_objective, 
            add_vinf_dep, add_vinf_arr, tof_encoding,
            tf_min,
            tf_max,
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
        self.tf_min = tf_min
        self.tf_max = tf_max

    # Number inequality Constraints
    def get_nic(self):
        return 2

    # redefine fitness
    def fitness(self, x):
        DV, _, T, _, _ = self._compute_dvs(x)
        # check final eopch
        tf = x[0] + sum(T)
        # earliest
        c_tf_l = self.tf_min - tf  # <= 0
        c_tf_u = tf - self.tf_max  # <= 0

        if not self._multi_objective:
            return [sum(DV),c_tf_l,c_tf_u]
        else:
            return (sum(DV), sum(T))


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