"""
Get planets object
"""

import pykep as pk
import numpy as np


MU_VENUS = 3.2485859200000006E+05*1e9   # in SI
MU_MARS  = 4.2828375214000022E+04*1e9   # in SI
MU_JUPITER = 1.2671276480000021E+08 * 1e9  # in SI
MU_SATURN = 3.7940585200000003E+07*1e9  # in SI
MU_TITAN = 8.978138845307376E+03 * 1e9   # SI

SMA_TITAN = 1221914042.5450587  # in SI

R_VENUS   = 12104/2 * 1e3
R_EARTH   = 6378 *1e3
R_MARS    = 6792/2 *1e3
R_JUPITER = 142984.0 / 2 * 1e3
R_SATURN  = 120536.0 / 2 * 1e3
R_TITAN   = 2575.0 * 1e3 


def solar_system_spice():
	venus   = pk.planet.spice('2', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_JUPITER, R_VENUS, R_VENUS * 1.05)
	earth   = pk.planet.spice('3', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH, R_EARTH, R_EARTH * 1.05)
	mars    = pk.planet.spice('4', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_MARS, R_MARS, R_MARS * 1.05)
	jupiter = pk.planet.spice('5', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_JUPITER, R_JUPITER, R_JUPITER * 1.05)
	saturn  = pk.planet.spice('6', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN, R_SATURN, R_SATURN * 1.05)

	solar_system_list = {
		"venus": venus,
		"earth": earth,
		"mars": mars,
		"jupiter": jupiter,
		"saturn": saturn,
	}
	return solar_system_list


def saturn_system(titan_safety_altitude=1000000):
	"""Construct Saturnian system from Keplerian elements

	Data from: 
	~https://nssdc.gsfc.nasa.gov/planetary/factsheet/saturniansatfact.html~
	https://ssd.jpl.nasa.gov/sats/elem/
	"""
	# elements of titan
	elements_titan = [
		1221914042.5450587,     # a
		 0.028659655999962765,  # e
	 	 0.34854*np.pi/180,     # inclination w.r.t. Saturn equator
		 2.949265038410724, 
		 3.2731913085559383,
		 1.491791176502704
	]
	titan  = pk.planet.keplerian(
		pk.epoch_from_string("2044-01-01 00:00:00"),
		elements_titan,  # a,e,i,W,w,M (SI units, i.e. meters and radiants)
		MU_SATURN,
		MU_TITAN,
		R_TITAN,
		R_TITAN + titan_safety_altitude,
		"titan",
	)

	saturn_system_list = {
		"titan": titan,
	}
	return saturn_system_list



def coord_planet(planet, t0, t_step):
    period = planet.compute_period(pk.epoch(t0, 'mjd2000'))
    n = int(round(period/t_step))
    ts_lambert = np.linspace(0, period, n)
    r0, v0 = planet.eph(pk.epoch(t0, 'mjd2000'))
    coord = np.zeros((6,n))
    for idx,t in enumerate(ts_lambert):
        rf,vf = pk.propagate_lagrangian(r0, v0, tof = t, mu = pk.MU_SUN)
        coord[0:3,idx] = rf
        coord[3:6,idx] = vf
    return coord


def coord_mga_1dsm(prob, x, t_step):
    # get solution stuff
    traj_comp, dsm_info = prob.get_trajectory_components(x)
    t0 = x[0]  # this is in MJD
    
    # trajectory storage
    rs_list, vs_list, ts_list = [], [], []
    t_running = 0
    
    # propagate each leg
    for idx, leg in enumerate(traj_comp):
        n = int(round(leg[2]/t_step))
        ts_lambert = np.linspace(0, leg[2], n)
    
        for t in ts_lambert:
            rf,vf = pk.propagate_lagrangian(r0 = leg[0], v0 = leg[1], tof = t, mu = leg[3])
            rs_list.append(rf)
            vs_list.append(vf)
            ts_list.append(t0 + t_running/pk.DAY2SEC + t/pk.DAY2SEC)
        t_running += leg[2]
            
    # convert to array
    n_elements = len(rs_list)
    coord = np.zeros((7,n_elements))
    for idx in range(n_elements):
        coord[0, idx]  = ts_list[idx]
        coord[1:4,idx] = rs_list[idx]
        coord[4:7,idx] = vs_list[idx]
    return coord, dsm_info, traj_comp



def coord_mga_1dsm_v2(traj_comp, t0, t_step):
    """Get coordinates from MGA1DSM problem"""
    # trajectory storage
    rs_list, vs_list, ts_list = [], [], []
    t_running = 0
    
    # propagate each leg
    for idx, leg in enumerate(traj_comp):
        n = int(round(leg[2]/t_step))
        ts_lambert = np.linspace(0, leg[2], n)
    
        for t in ts_lambert:
            rf,vf = pk.propagate_lagrangian(r0 = leg[0], v0 = leg[1], tof = t, mu = leg[3])
            rs_list.append(rf)
            vs_list.append(vf)
            ts_list.append(t0 + t_running/pk.DAY2SEC + t/pk.DAY2SEC)
        t_running += leg[2]
            
    # convert to array
    n_elements = len(rs_list)
    coord = np.zeros((7,n_elements))
    for idx in range(n_elements):
        coord[0, idx]  = ts_list[idx]
        coord[1:4,idx] = rs_list[idx]
        coord[4:7,idx] = vs_list[idx]
    return coord