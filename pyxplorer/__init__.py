"""
Init file for explorer team
"""

import pykep as pk
import os



try:
	pk.util.load_spice_kernel(os.path.join(os.environ.get("SPICE"), "spk", "de440.bsp"))
	pk.util.load_spice_kernel(os.path.join(os.environ.get("SPICE"), "lsk", "naif0012.tls"))
except:
	print("SPICE furnish unsuccessful; please load `de440.bsp` and `naif0012.tls` by hand!")
	pass


from ._planets import *
from ._problems import *
