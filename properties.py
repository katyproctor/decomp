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


def calc_ebindrel(dat):
    e = dat['ParticleBindingEnergy']/dat['Mass']
    dat['ebindrel'] = e/np.min(e)

    return dat


def calc_kappa_co(dat, rcut):
    '''
    Calculate kappa_rot & kappa_co as per Correa 2017
    '''
    dat['R'] = np.sqrt(dat['x']**2 +dat['y']**2)

    dat['jz'] = dat['jz/jcirc']*dat['jcirc']
    dat['Krot'] = 0.5*dat['Mass']*(dat['jz']/dat['R'])**2
    dat['K'] = 0.5*dat['Mass']*(dat['vx']**2 + dat['vy']**2 + dat['vz']**2)

    kappa_rot = sum(dat['Krot'][dat['rad'] < rcut]) / sum(dat['K'][dat['rad'] < rcut])
    kappa_co = sum(dat['Krot'][(dat['rad'] < rcut) & (dat['jz'] > 0)]) / sum(dat['K'][dat['rad'] < rcut])

    return kappa_rot, kappa_co


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
    Centre positions on centre of potential
    '''
    dat['x'] = dat['x'] - dat['cop_x']
    dat['y'] = dat['y'] - dat['cop_y']
    dat['z'] = dat['z'] - dat['cop_z']
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
    Calculates specific j for each particle and total angular momentum vector.
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

    return dat


def run(dat):
    ''' 
    Aligns galaxy with net angular momentum vector within a sphere enclosing 80% of the stellar mass
    Calculates angular momentum and jcirc values for all star particles
    Inputs:
    dat: pd.DataFrame(), processed star data
    '''
    # centre pos and vel
    dat = centre_pos(dat)
    dat = centre_of_mass_velocity(dat)

    # in rare cases, there are clumps of (apparently bound) stars very far from the galaxy centre
    # remove these
    dat = dat[dat['rad'] < 2000].copy()

    # align pos and vels such that J is normal to the disk (if disk exists)
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
                "cop_x", "cop_y", "cop_z",  "type", "GroupNumber"]
    dat = dat.drop(drop_cols, axis=1)

    return dat


