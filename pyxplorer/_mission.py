"""
Mission parameters
"""


import numpy as np
import datetime


def blackout_window_2040s():
	blackout_windows = [
		[
			np.datetime64(datetime.datetime.strptime("2040-09-26", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2040-10-18", '%Y-%m-%d'))	
		],
		[
			np.datetime64(datetime.datetime.strptime("2041-10-09", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2041-10-31", '%Y-%m-%d'))	
		],
		[
			np.datetime64(datetime.datetime.strptime("2042-10-21", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2042-11-12", '%Y-%m-%d'))	
		],
		[
			np.datetime64(datetime.datetime.strptime("2043-11-02", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2043-11-24", '%Y-%m-%d'))	
		],
		[
			np.datetime64(datetime.datetime.strptime("2044-11-13", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2044-12-05", '%Y-%m-%d'))	
		],
		[
			np.datetime64(datetime.datetime.strptime("2045-11-25", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2045-12-17", '%Y-%m-%d'))	
		],
		[
			np.datetime64(datetime.datetime.strptime("2046-12-06", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2046-12-28", '%Y-%m-%d'))	
		],
		[
			np.datetime64(datetime.datetime.strptime("2047-12-17", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2048-01-08", '%Y-%m-%d'))	
		],
		[
			np.datetime64(datetime.datetime.strptime("2048-12-28", '%Y-%m-%d')),
			np.datetime64(datetime.datetime.strptime("2049-01-19", '%Y-%m-%d'))	
		],
	]
	return blackout_windows