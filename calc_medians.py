import numpy as np
import pandas as pd
import re
import glob

import calc_morphology

def calc_kappa_co(dat, rcut):
    '''
    Calculate kappa_rot & kappa_co as per Correa 2017
    '''
    dat['R'] = np.sqrt(dat['x']**2 +dat['y']**2)

    dat['jz'] = dat['jz/jcirc']*dat['jcirc'] # dumb but didnt save jz in processing...
    dat['Krot'] = 0.5*dat['Mass']*(dat['jz']/dat['R'])**2
    dat['K'] = 0.5*dat['Mass']*(dat['vx']**2 + dat['vy']**2 + dat['vz']**2)

    kappa_rot = sum(dat['Krot'][dat['rad'] < rcut]) / sum(dat['K'][dat['rad'] < rcut])
    kappa_co = sum(dat['Krot'][(dat['rad'] < rcut) & (dat['jz'] > 0)]) / sum(dat['K'][dat['rad'] < rcut])

    return kappa_co


def calc_rx(dat, x):
    '''Calculate the radius that contains x% of the stellar mass.
    x entered as  decimal'''
    dat = dat.sort_values("rad")
    dat['menc'] = dat['Mass'].cumsum()
    x_mtot = np.max(dat['menc'])*x
    dat['diff'] = abs(x_mtot - dat['menc'])

    # find minimum diff from m50
    rid = dat['diff'].idxmin()
    rx = dat.loc[rid]['rad']

    return rx


def classify_kinematic_cuts(dat):
    '''Classify disk stars
      Method follows Zolotov 2009, Section 2.1
      Decompose spheroid into bulge+halo based on energy cut.'''

    # disk stars
    dat['disk_zolotov'] = np.where(dat['jz/jcirc'] >= 0.8, True, False)
    # add extra z +/- bounds for mccarthy disks
    dat['disk_mccarthy'] = np.where((dat['jz/jcirc'] >= 0.8) & (abs(dat['z']) < 2), True, False)

    # unclear from paper how this is energy cut-off actually defined - just using median
    energy_cutoff_z = np.median(dat['ParticleBindingEnergy'][dat['disk_zolotov'] == False])
    energy_cutoff_m = np.median(dat['ParticleBindingEnergy'][dat['disk_mccarthy'] == False])

    # IHL/bulge separation
    dat['bulge_zolotov'] = np.where((dat['ParticleBindingEnergy']<energy_cutoff_z) & (dat['disk_zolotov']==False), True, False)
    dat['ihl_zolotov'] = np.where((dat['ParticleBindingEnergy']>=energy_cutoff_z) & (dat['disk_zolotov']==False), True, False)

    dat['bulge_mccarthy'] = np.where((dat['ParticleBindingEnergy']<energy_cutoff_m) & (dat['disk_mccarthy']==False), True, False)
    dat['ihl_mccarthy'] = np.where((dat['ParticleBindingEnergy']>=energy_cutoff_m) & (dat['disk_mccarthy']==False), True, False)
    
    dat['kinematic_cuts'] = np.where(dat['disk_zolotov'] == True, "disk",
                                   np.where(dat['bulge_zolotov'] == True, "bulge",
                                           np.where(dat['ihl_zolotov'] == True, "IHL", "check")))

    return dat


def classify_aperture_cut(dat):
    '''
    Input data: stars only
    Define stellar halo by anything beyond 2*half mass stellar radius as in Elias 2018
    '''
    dat['ihl_20kpc'] = np.where(dat['rad'] > 20, "IHL", "galaxy")

    # half stellar mass radius
    dat = dat.sort_values(by=['rad'])
    dat['stellar_massenc'] = dat['Mass'].cumsum()

    half_mtot = np.max(dat['stellar_massenc'])/2 # half mass value
    dat['diff'] = abs(half_mtot - dat['stellar_massenc'])

    # find minimum diff from m50
    id = dat['diff'].idxmin()
    r50 = dat.loc[id]['rad']
    dat['ihl_2halfmass'] = np.where(dat['rad']>2*r50, "IHL", "galaxy")

    return dat


def calc_fractions(dat):
    '''
    Calculate mass fraction in disk, bulge or IHL stars for various methods
    '''
    # apertures
    dat['fihl_20kpc'] = dat['Mass'][dat['ihl_20kpc'] == "IHL"].sum()/dat['Mass'].sum()
    dat['fihl_2halfmass'] = dat['Mass'][dat['ihl_2halfmass'] == "IHL"].sum()/dat['Mass'].sum()
    
    # kinematic cuts
    dat['fihl_kinematic'] = dat['Mass'][dat['kinematic_cuts'] == "IHL"].sum()/dat['Mass'].sum()
    dat['fdisk_kinematic'] = dat['Mass'][dat['kinematic_cuts'] == "disk"].sum()/dat['Mass'].sum()
    dat['fbulge_kinematic'] = dat['Mass'][dat['kinematic_cuts'] == "bulge"].sum()/dat['Mass'].sum()
    
    # our GMM method
    dat['fihl'] = dat['Mass'][dat['gmm_pred'] == "IHL"].sum()/dat['Mass'].sum()
    dat['fdisk'] = dat['Mass'][dat['gmm_pred'] == "disk"].sum()/dat['Mass'].sum()
    dat['fbulge'] = dat['Mass'][dat['gmm_pred'] == "bulge"].sum()/dat['Mass'].sum()
    
    return dat


def median_calc(dat, col, col_str, var_list):
    '''
    Calculate medians for each component for each var in var_list
    '''
    meds = dat.groupby([col])[var_list].median().reset_index().copy()
    meds['method'] = col_str
    meds.rename(columns={ meds.columns[0]: "component" }, inplace = True)
    
    return meds



def main():
    # particle and group directories
    fpath = '/fred/oz009/kproctor/L0100N1504/processed/'
    flist = glob.glob(fpath + "*.pkl")

    # initialise output dataframes
    global_rows = []
    all_meds_rows = []

    for i, f in enumerate(flist):
        gpn = int(re.findall(r'\d+', f)[-1]) # extract group number, last entry because numbers in filepath
        dat = pd.read_pickle(f)
        nstar = dat.shape[0]
      
        # only process galaxies with at least 1000 star particles
        if nstar < 1000:
            continue
       
        dat['GroupNumber'] = gpn
        dat['Fe/O'] = dat['ElementAbundance/Iron']/dat['ElementAbundance/Oxygen']
        dat["Formation_z"] = 1/dat['StellarFormationTime'] - 1
        dat['mstar'] = dat['Mass'].sum()

        dat['nihl'] = dat[dat['gmm_pred'] == "IHL"].shape[0]
        dat['ndisk'] = dat[dat['gmm_pred'] == "disk"].shape[0]
        dat['nbulge'] = dat[dat['gmm_pred'] == "bulge"].shape[0]

        ## Calculate kappa_rot and ellipticity as morphology indicator
        r50 = calc_rx(dat, 0.5)
        dat['kappa_co_30kpc'] = calc_kappa_co(dat, 30)
        dat['kappa_co_2r50'] = calc_kappa_co(dat, 2*r50) # calculate kappa_co at 2*r50
        ellip,triax,Transform,abc = calc_morphology.morphological_diagnostics(dat)
        ellip_2r50,triax_2r50,Transform,abc_2r50 = calc_morphology.morphological_diagnostics(dat, aperture = 2*r50)
        dat['ellip'] = ellip
        dat['triax'] = triax
        dat['a'] = abc[0]
        dat['b'] = abc[1]
        dat['c'] = abc[2]
        dat['a2r50'] = abc_2r50[0]
        dat['b2r50'] = abc_2r50[1]
        dat['c2r50'] = abc_2r50[2]
       
        ## Calculate IHL mass estimates for other methods
        dat = classify_aperture_cut(dat)
        dat = classify_kinematic_cuts(dat)

        # mass fractions for various methods
        dat = calc_fractions(dat)

        # save dataframe of just mass fractions and global properties
        global_cols = ['GroupNumber','mstar', 'm200', 'r200',
             'kappa_co_30kpc', 'kappa_co_2r50',
             'ellip', 'triax', 'a', 'b', 'c', 'a2r50', 'b2r50', 'c2r50',
            "ihl_mad", "nihl", "ndisk", "nbulge",
            'fihl_20kpc', 'fihl_2halfmass', 'fihl_kinematic',
           'fdisk_kinematic', 'fbulge_kinematic', 'fihl', 'fdisk', 'fbulge']
        global_tmp = dat[global_cols].drop_duplicates()
        global_rows.append(global_tmp)

        # median properties by component for different methods
        dat['gmm_pred_gal'] = np.where(dat['gmm_pred'] != "IHL", "galaxy", "IHL") 
        var_list = ['jz/jcirc', 'ebindrel', 'Formation_z', 'Metallicity', 'BirthDensity', 'Fe/O']

        gmm_meds = median_calc(dat, "gmm_pred", "gmm", var_list)
        gmm_gal_meds = median_calc(dat, "gmm_pred_gal", "gmm_gal", var_list)
        aperture_meds = median_calc(dat, "ihl_2halfmass", "aperture", var_list)
        cut_meds = median_calc(dat, "kinematic_cuts", "cuts", var_list)

        # get r50 for each component & method
        if dat[dat['gmm_pred'] == "IHL"].shape[0] != 0:
            r50_ihl = calc_rx(dat[dat['gmm_pred'] == "IHL"], 0.5)
            gmm_meds.loc[gmm_meds['component'] == "IHL", "r50"] = r50_ihl
        if dat[dat['gmm_pred'] == "disk"].shape[0] != 0:
            r50_disk = calc_rx(dat[dat['gmm_pred'] == "disk"], 0.5)
            gmm_meds.loc[gmm_meds['component'] == "disk", "r50"] = r50_disk
            jzjc_0p7_diskmass = dat['Mass'][(dat['gmm_pred'] == "disk") & (dat['jz/jcirc'] > 0.7)].sum()
            disk_mass = dat['Mass'][dat['gmm_pred'] == "disk"].sum()
            f_jzjc = jzjc_0p7_diskmass/disk_mass
            gmm_meds.loc[gmm_meds['component'] == 'disk', 'f_jzjc'] = f_jzjc
        if dat[dat['gmm_pred'] == "bulge"].shape[0] != 0:
            r50_bulge = calc_rx(dat[dat['gmm_pred'] == "bulge"], 0.5)
            gmm_meds.loc[gmm_meds['component'] == "bulge", "r50"] = r50_bulge

        all_meds = pd.concat([gmm_meds, gmm_gal_meds, aperture_meds, cut_meds])
        all_meds['GroupNumber'] = gpn
        all_meds['mstar'] = dat['Mass'].sum()
        all_meds['m200'] = dat['m200'].unique()[0]

        # append rows
        all_meds_rows.append(all_meds)

    # this is more efficient than appending pandas rows
    global_dat = pd.concat(global_rows, ignore_index=True)   
    all_meds = pd.concat(all_meds_rows, ignore_index=True)

    global_dat.to_pickle(fpath + "analysis/" + "global.pkl")
    all_meds.to_pickle(fpath + "analysis/" + "meds.pkl")

if __name__ == '__main__':
    main()

