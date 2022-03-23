"""
SPICE based integrator
"""


import numpy as np
import spiceypy as spice
from numba import njit
from scipy.interpolate import CubicSpline


def interp_spkssb(
        et0, 
        et1, 
        bodyID, 
        frame='ECLIPJ2000', 
        lstar=1., 
        vstar=1., 
        dt=3.0, 
        normalize_et=False, 
        n=None,
    ):
    """Create cubic interpolation of ephemerides obtained from spice
    Args:
        et0 (float): earliest epoch for interpolation, in mjd
        et1 (float): latest epoch for interpolation, in mjd
        bodyID (np.array): np.array of integer bodies
        frame (str): frame to compute state from spice
        lstar (float): canonical units for length
        vstar (float): canonical units for velocity
        dt (float): step-size for sampling points, in seconds
        normalize_et (bool): if set to True, use et0 to normalize epoch
        n (int): over-write n
        
    Returns:
        (tuple): cs_x: 6-by-n breakpoints, cs_c: 6-by-4-by-n coefficients
    """
    if normalize_et==True:
        et_scale = et0
        et1 = et1/et_scale
        dt = dt/et_scale
        et0 = 1.0

    if n==None:
        n = max(int(np.ceil((et1-et0)/dt)), 2)

    ets = np.linspace(et0, et1, n)
    # arrays to store
    xs, ys, zs,   = np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    vxs, vys, vzs = np.zeros((n,)), np.zeros((n,)), np.zeros((n,))
    for idx, et in enumerate(ets):
        if normalize_et==False:
            sv = spice.spkssb(bodyID, et, frame)
            state = np.concatenate((sv[0:3]/lstar, sv[3:6]/vstar))
        else:
            sv = spice.spkssb(bodyID, et*et_scale, frame)
            state = np.concatenate((sv[0:3]/lstar, sv[3:6]/vstar))

        xs[idx] = state[0]
        ys[idx] = state[1]
        zs[idx] = state[2]
        vxs[idx] = state[3]
        vys[idx] = state[4]
        vzs[idx] = state[5]
    # fit spline
    cxs = CubicSpline(ets, xs)
    cys = CubicSpline(ets, ys)
    czs = CubicSpline(ets, zs)
    cvxs = CubicSpline(ets, vxs)
    cvys = CubicSpline(ets, vys)
    cvzs = CubicSpline(ets, vzs)
    # coefficient length
    _, c_shape = cxs.c.shape
    # store breakpoints
    cs_x = np.zeros((6,len(cxs.x)),)
    cs_x[0,:] = cxs.x
    cs_x[1,:] = cys.x
    cs_x[2,:] = czs.x
    cs_x[3,:] = cvxs.x
    cs_x[4,:] = cvys.x
    cs_x[5,:] = cvzs.x
    # store coefficients
    cs_c = np.zeros((6,4,c_shape),)
    cs_c[0,:,:] = cxs.c
    cs_c[1,:,:] = cys.c
    cs_c[2,:,:] = czs.c
    cs_c[3,:,:] = cvxs.c
    cs_c[4,:,:] = cvys.c
    cs_c[5,:,:] = cvzs.c
    return cs_x, cs_c


@njit
def get_controlnode_spline(et, coefs, bps, et_normalize=1.):
    """Get control-node from spline-interpolated object
    Args:
        et (float): time where state is to be interpolated
        coefs (np.array): 6-by-4-by-n array of coefficients corresponds to cs.c
        bps (np.array): 6-by-(n+1) array of breakpoints, corresponds to cs.x
        et_normalize (float): normalization scalar used for producing spline (et/et_normalize)
    Returns:
        (np.state): length-6 state of spacecraft
    """
    state = np.zeros(6,)
    for j in range(6):
        state[j] = eval_spline(teval=et/et_normalize, coefs=coefs[j,:,:], bps=bps[j,:])
    return state