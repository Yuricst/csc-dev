"""
Get planets object
"""

import pykep as pk



MU_VENUS = 3.2485859200000006E+05*1e9   # in SI
MU_JUPITER = 1.2671276480000021E+08 * 1e9  # in SI
MU_SATURN = 3.7940585200000003E+07*1e9  # in SI

R_VENUS  = 12104/2 * 1e3
R_EARTH  = 6378 *1e3
R_JUPITER = 142984.0 / 2 * 1e3
R_SATURN = 120536.0 / 2 * 1e3


def solar_system_spice():
	venus = pk.planet.spice('2', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_JUPITER, R_VENUS, R_VENUS * 1.05)
	earth = pk.planet.spice('3', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, pk.MU_EARTH, R_EARTH, R_EARTH * 1.05)
	jupiter = pk.planet.spice('5', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_JUPITER, R_JUPITER, R_JUPITER * 1.05)
	saturn = pk.planet.spice('6', 'SUN', 'ECLIPJ2000', 'NONE', pk.MU_SUN, MU_SATURN, R_SATURN, R_SATURN * 1.05)

	solar_system_list = {
		"venus": venus,
		"earth": earth,
		"jupiter": jupiter,
		"saturn": saturn,
	}
	return solar_system_list