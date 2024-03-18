import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq
from scipy import stats
from scipy.interpolate import interp1d

from astropy.cosmology import Planck15  

G = 4.301721e-6 # [kpc.(km/s)^2/Msol]


def calc_ebindrel(dat):
    e = dat['ParticleBindingEnergy'] # already specific binding e 
    dat['ebindrel'] = e/np.min(e)
    return dat


def calc_rx(dat, x=0.5):
    '''Calculate the radius that contains x% of the stellar mass.
    x to be input as decimal'''

    dat = dat.sort_values("rad")
    dat['menc'] = dat['Mass'].cumsum()
    dat['fmass'] = dat['menc']/dat['Mass'].sum()

   # first occurrence above 0.5
    rid = np.argmax(dat['fmass'] >= x)
    fid = dat['fmass'].iloc[rid]

    if fid < x:
        rx_arr = np.array(dat.iloc[rid:rid+2]['rad'])
        fx_arr = np.array(dat.loc[rid:rid+2]['fmass'])

    else:    
        rx_arr = np.array(dat.iloc[rid-1:rid+1]['rad'])
        fx_arr = np.array(dat.iloc[rid-1:rid+1]['fmass'])

    try:
        if fx_arr[1] - fx_arr[0] < 0:
            rx = dat['rad'].loc[rid]
        else:
            interp = interp1d(fx_arr, rx_arr)
            rx = interp(x)

    except IndexError:
        rx = dat['rad'].loc[rid]

    return float(rx)


def centre_pos(dat, boxsize):
    '''
    Centre positions on centre of potential and account for periodicity
    '''

    dat['x'] = np.mod(dat['x'] - dat['cop_x']+0.5*boxsize, boxsize)-0.5*boxsize
    dat['y'] = np.mod(dat['y'] - dat['cop_y']+0.5*boxsize, boxsize)-0.5*boxsize
    dat['z'] = np.mod(dat['z'] - dat['cop_z']+0.5*boxsize, boxsize)-0.5*boxsize
    dat['rad'] = np.sqrt(dat['x']**2 + dat['y']**2 + dat['z']**2)

    return dat


def centre_of_mass_velocity(dat):
    """
    Return the center of mass velocity
    """
    cut_off = calc_rx(dat, 0.8)
    tmp = dat[dat['rad'] < cut_off].copy()

    mtot = tmp["Mass"].sum()
    vx = np.sum(tmp["Mass"] * tmp["vx"]) / mtot
    vy = np.sum(tmp["Mass"] * tmp["vy"]) / mtot
    vz = np.sum(tmp["Mass"] * tmp["vz"]) / mtot

    dat['vx'] = dat['vx'] - vx
    dat['vy'] = dat['vy'] - vy
    dat['vz'] = dat['vz'] - vz

    return dat


def tot_ang_mom(dat):
    '''
    Total angular momentum vector in central region,
    Assumes positions and velocities have been centred.
    '''
    # calculate vector in inner regions (we only care about J of disk)
    tmp = dat[(dat['rad'] > 2) & (dat['rad'] < 30)].copy()

    pos = np.array([tmp['x'], tmp['y'], tmp['z']]).T
    vel = np.array([tmp['vx'], tmp['vy'], tmp['vz']]).T
    mass = np.array(tmp['Mass'])

    return (mass[:,None] * np.cross(pos, vel)).sum(axis=0)


def find_rotation_matrix(j_vector):
    '''
    Returns a scipy.spatial.transform.Rotation object.
    '''
    # rotate until x coord = 0
    fy = lambda y : Rotation.from_euler('y', y, degrees=True).apply(j_vector)[0]
    y = brentq(fy, 0, 180)

    #rotate until y coord = 0
    fx = lambda x : Rotation.from_euler('yx', [y,x], degrees=True).apply(j_vector)[1]
    x = brentq(fx, 0, 180)

    # check we're not upside down
    j_tot = Rotation.from_euler('yx', [y,x], degrees=True).apply(j_vector)

    if j_tot[2] < 0:
        x += 180

    return Rotation.from_euler('yx', [y,x], degrees=True)


def calc_j(dat):
    '''
    Calculates specific j for each particle and total specfic angular momentum vector.
    '''
    dat['jx'] = (dat['y']*dat['vz'] - dat['z']*dat['vy'])
    dat['jy'] = (dat['z']*dat['vx'] - dat['x']*dat['vz'])
    dat['jz'] = (dat['x']*dat['vy'] - dat['y']*dat['vx'])
    dat['J'] = np.sqrt(dat['jx']**2 + dat['jy']**2 + dat['jz']**2)

    return dat


def calc_jcirc_num(dat):
    '''
    bin particles in binding energy space and use the max j as the jcirc(e) val
    '''

    dat = dat.sort_values('ParticleBindingEnergy')

    # binned values of jc
    max_jc, bin_edges, bin_no = stats.binned_statistic(dat['ParticleBindingEnergy'], dat['J'], 'max', bins=150)
    de = bin_edges[1] - bin_edges[0]
    ebin  = bin_edges[0:-1] + de/2
    
    jc_arr = np.array([])

    for i in range(len(ebin)):
        len_bin = len(bin_no[bin_no == i+1])
        jc_bin = np.repeat(max_jc[i], len_bin)
        jc_arr = np.append(jc_arr, jc_bin)

    dat['jcirc'] = jc_arr
    dat['jz/jcirc'] = dat['jz']/dat['jcirc']
    dat['jp/jcirc'] = np.sqrt(dat['jx']**2+dat['jy']**2)/dat['jcirc']

    return dat


def run(dat, boxsize):
    ''' 
    Aligns galaxy with net angular momentum vector between 2 and 30 kpc
    Calculates angular momentum and jcirc values for all star particles
    Inputs:
    dat: pd.DataFrame(), processed particle data
    '''
    dat = centre_pos(dat, boxsize)
    dat = centre_of_mass_velocity(dat)

    # rotate coords such that J is normal to the disk (if disk exists)
    jtot = tot_ang_mom(dat)
    rot_mat = find_rotation_matrix(jtot)
    rotated = rot_mat.apply(np.array([dat['x'], dat['y'], dat['z']]).T)
    rotated_v = rot_mat.apply(np.array([dat['vx'], dat['vy'], dat['vz']]).T)
    dat['x'], dat['y'], dat['z'] = rotated.T[0], rotated.T[1], rotated.T[2]
    dat['vx'], dat['vy'], dat['vz'] = rotated_v.T[0], rotated_v.T[1], rotated_v.T[2]
    dat = dat[dat['ParticleBindingEnergy'] < 0].copy()
    
    print("Calculating jcirc for %d bound particles..." % dat.shape[0], flush=True)
    dat = calc_j(dat)
    dat = calc_jcirc_num(dat)
    print("Finished jcirc calcs", flush = True)

    # clean up data for modelling
    drop_cols = ["J","jx", "jy", "jz",
                "cop_x", "cop_y", "cop_z"]

    dat = dat.drop(drop_cols, axis=1)

    return dat

