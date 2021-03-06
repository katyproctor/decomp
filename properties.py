# Functions to:
#   - align particles to an angular momentum vector
#   - calculate jz/jcirc for each particle
#   - classify components via various methods in literature

G = 4.301721e-6 # [kpc.(km/s)^2/Msol]

import numpy as np
from scipy import interpolate
from scipy.spatial.transform import Rotation
from scipy.optimize import brentq
from scipy import stats


def calc_rx(dat, x):
    '''Calculate the radius that contains x% of the stellar mass.
    x to be input as decimal'''

    dat = dat.sort_values("rad")
    dat['menc'] = dat['Mass'].cumsum()
    x_mtot = np.max(dat['menc'])*x
    dat['diff'] = abs(x_mtot - dat['menc'])

    # find minimum diff from m50
    rid = dat['diff'].idxmin()
    rx = dat.loc[rid]['rad']

    return rx


def centre_pos(dat):
    '''
    Centre positions on centre of potential (for EAGLE galaxies)
    '''
    dat['x'] = dat['x'] - dat['cop_x']
    dat['y'] = dat['y'] - dat['cop_y']
    dat['z'] = dat['z'] - dat['cop_z']

    dat['rad'] = np.sqrt(dat['x']**2 + dat['y']**2 + dat['z']**2)

    return dat


def center_of_mass_velocity(dat):
    """
    Return the center of mass velocity
    """
    # calculate vector in inner regions (we only care about J of disk)
    #cut_off = dat['rad'][dat['type'] == "star"].quantile(0.8)i
    cut_off = calc_rx(dat[dat['type'] == "star"], 0.8)
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
    #cut_off = dat['rad'][dat['type'] == "star"].quantile(0.8)
    cut_off = calc_rx(dat[dat['type'] == "star"], 0.8)
    tmp = dat[dat['rad'] < cut_off].copy()

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
    Calculates specific j for each particle and total angular momentum vector.
    '''
    dat['jx'] = (dat['y']*dat['vz'] - dat['z']*dat['vy'])
    dat['jy'] = (dat['z']*dat['vx'] - dat['x']*dat['vz'])
    dat['jz'] = (dat['x']*dat['vy'] - dat['y']*dat['vx'])

    dat['J'] = np.sqrt(dat['jx']**2 + dat['jy']**2 + dat['jz']**2)

    return dat


def calc_jcirc_num(dat):
    '''
    Calculates jcirc based on method of Kumar 2021:
    bin particles in order of binding energy and use the max jz as the jc val
    '''

    dat = dat.sort_values('ParticleBindingEnergy')

    # binned values of jc
    max_jc, bin_edges, bin_no = stats.binned_statistic(dat['ParticleBindingEnergy'], abs(dat['jz']), 'max', bins=100)
    dx = bin_edges[1] - bin_edges[0]
    ebin  = bin_edges[0:-1] + dx/2

    # add back onto data frame
    dat['bin'] = bin_no

    # create jc array for each particle
    jc_arr = np.array([])

    for i in range(len(ebin)):
        len_bin = len(bin_no[bin_no == i+1])
        jc_bin = np.repeat(max_jc[i], len_bin)
        jc_arr = np.append(jc_arr, jc_bin)

    dat['jcirc'] = jc_arr
    dat['jp'] = np.sqrt(dat['jx']**2 + dat['jy']**2)
    dat['jz/jcirc'] = dat['jz']/dat['jcirc']
    dat['jp/jcirc'] = dat['jp']/dat['jcirc']

    return dat


def run(dat):
    ''' 
    Aligns galaxy with net angular momentum vector within a sphere enclosing 80% of the stellar mass
    Calculates angular momentum and jcirc values for all star particles
    Inputs:
    dat: pd.DataFrame(), processed star and DM data (DM used in angular momenta calcs)
    '''
    # centre of potential subtraction
    dat = centre_pos(dat)

    # centre of mass velocity subtraction (within sphere of stars)
    dat = center_of_mass_velocity(dat)

    # align pos and vels such that J is normal to the disk (if disk exists)
    jtot = tot_ang_mom(dat)
    rot_mat = find_rotation_matrix(jtot)
    rotated = rot_mat.apply(np.array([dat['x'], dat['y'], dat['z']]).T)
    rotated_v = rot_mat.apply(np.array([dat['vx'], dat['vy'], dat['vz']]).T)

    # redefine pos and vel so that they are aligned with the disk
    dat['x'], dat['y'], dat['z'] = rotated.T[0], rotated.T[1], rotated.T[2]
    dat['vx'], dat['vy'], dat['vz'] = rotated_v.T[0], rotated_v.T[1], rotated_v.T[2]

    # potential and angular momenta calcs
    print("Calculating jcirc for %d bound particles..." % dat.shape[0], flush=True)
    
    # bound particles only
    dat = dat[dat['ParticleBindingEnergy'] < 0].copy()
    dat = calc_j(dat)

    # get rid of interlopers
    dat = dat[dat['rad'] < 2*dat['r200']].copy()
    dat = calc_jcirc_num(dat)
    print("Finished jcirc calcs", flush = True)

    # only interested in stars from here on
    dat = dat[dat['type'] == "star"].copy()

    # clean up data for modelling
    drop_cols = ["J","bin", "jx", "jy", "jz", "jp",
                "cop_x", "cop_y", "cop_z",  "type", "GroupNumber"]
    dat = dat.drop(drop_cols, axis=1)

    return dat


