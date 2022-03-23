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
		1221914042.5450587,
		 0.028659655999962765,
		 0.48280065937311095,
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


