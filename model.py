import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

import classify_clusters


def fit_disk_gmms(iter_dat, min_ncomp, max_ncomp, final_ncomp):
    
    allocs_arr = []
    model_dat = iter_dat[['jz/jcirc', 'ebindrel', 'jp/jcirc']]

    ns = np.arange(min_ncomp, max_ncomp+1)
    if final_ncomp not in ns:
        ns = np.append(ns, final_ncomp)
    
    for n in ns:
        model = GaussianMixture(n_components=n, n_init = 5,
        covariance_type='full', random_state=0).fit(model_dat)
        
        preds = model.predict(model_dat)
        probs = model.predict_proba(model_dat)

        # save predicted cluster
        col_name = "clus_" + str(n)
        iter_dat[col_name] = preds

        # classify Gaussian clusters as disk, bulge or IHL
        iter_dat["comp_" + str(n)], allocs, ecut_tmp  = classify_clusters.disk_allocation(iter_dat, col_name, model, probs)
        allocs_arr.append(allocs)

        if n == final_ncomp:
            # save binding energy cut value used for final allocation
            ecut = ecut_tmp

    return [iter_dat, allocs_arr, ecut]


def fit_sph_gmms(iter_dat, min_ncomp, max_ncomp, final_ncomp, priors = False):
    
    allocs_arr = []
    model_dat = iter_dat[['jz/jcirc', 'ebindrel']]

    ns = np.arange(min_ncomp, max_ncomp+1)
    if final_ncomp not in ns:
        ns = np.append(ns, final_ncomp)

    for n in ns:

        if priors:

            # priors
            spacing = 1/(n - 1)
            ebind_guess = [i*spacing for i in range(n)]
            jzjc_guess = [0]*n
            priors = np.array([jzjc_guess, ebind_guess]).T

            model = GaussianMixture(n_components=n, n_init = 5,
            covariance_type='full', means_init = priors, random_state=0).fit(model_dat)

        else:
            model = GaussianMixture(n_components=n, n_init = 5,
            covariance_type='full', random_state=0).fit(model_dat)

        preds = model.predict(model_dat)
        probs = model.predict_proba(model_dat)
        
        col_name = "clus_" + str(n)
        iter_dat[col_name] = preds
        iter_dat["comp_" + str(n)], allocs, ecut_tmp = classify_clusters.sph_allocation(iter_dat, col_name, model, probs)
        allocs_arr.append(allocs)

        if n == final_ncomp:
            # save binding energy cut value used for final allocation
            ecut = ecut_tmp

    return [iter_dat, allocs_arr, ecut]


def summarise_models(dat, min_ncomp, max_ncomp, ecut):

    summary = dat[['GroupNumber', 'm200']].drop_duplicates().copy()
    summary['mstar'] = dat['Mass'].sum()
    summary['ecut'] = ecut

    # mass predicted in each component for diff n
    for i in range(min_ncomp, max_ncomp + 1):
        mdisk = dat['Mass'][dat['comp_' + str(i)] == "disk"].sum()
        mbulge = dat['Mass'][dat['comp_' + str(i)] == "bulge"].sum()
        mihl = dat['Mass'][dat['comp_' + str(i)] == "IHL"].sum()

        summary["mdisk_comp_" + str(i)] = mdisk
        summary["mbulge_comp_" + str(i)] = mbulge
        summary["mihl_comp_" + str(i)] = mihl

    disk_mass = dat['Mass'][dat['gmm_pred'] == 'disk'].sum()
    bulge_mass = dat['Mass'][dat['gmm_pred'] == 'bulge'].sum()
    ihl_mass = dat['Mass'][dat['gmm_pred'] == 'IHL'].sum()

    summary['fdisk'] = np.where(disk_mass > 0, disk_mass/dat['Mass'].sum(), np.NaN) 
    summary['fbulge'] = np.where(bulge_mass > 0, bulge_mass/dat['Mass'].sum(), np.NaN) 
    summary['fihl'] = np.where(ihl_mass > 0, ihl_mass/dat['Mass'].sum(), np.NaN) 
    
    return summary


def morph_class(iter_dat, nmorph=3):
    '''Classifies a galaxy as Disk or Spheroid morphology based on a GMM with nmorph components'''

    model_dat = iter_dat[['jz/jcirc', 'ebindrel', 'jp/jcirc']]

    # 3-comp fit
    model = GaussianMixture(n_components=nmorph, n_init = 5,
        covariance_type='full', random_state=0).fit(model_dat)
    jzjc_means = model.means_.T[0]

    if all(jzjc_means < 0.5):
        sph = True
    else:
        sph = False

    return sph


def allocate_particles(iter_dat, final_ncomp):
    
    disk_ids = iter_dat['ParticleIDs'][iter_dat['comp_'+str(final_ncomp)] == "disk"]
    bulge_ids = iter_dat['ParticleIDs'][iter_dat['comp_'+str(final_ncomp)] == "bulge"]
    ihl_ids = iter_dat['ParticleIDs'][iter_dat['comp_'+str(final_ncomp)] == "IHL"]

    iter_dat['gmm_pred'] = np.where(iter_dat['ParticleIDs'].isin(disk_ids), "disk",
                            np.where(iter_dat['ParticleIDs'].isin(bulge_ids), "bulge",
                                np.where(iter_dat['ParticleIDs'].isin(ihl_ids), "IHL", "check")))

    return iter_dat


def log_fractions(dat, ecut):
    '''
    Log mass fraction in each component
    '''
    print("-----------------------")
    mstar = dat['Mass'].sum()
    m200 = dat['m200'].unique()[0]
    fihl = dat['Mass'][dat['gmm_pred'] == "IHL"].sum()/mstar
    fb = dat['Mass'][dat['gmm_pred'] == "bulge"].sum()/mstar
    fd = dat['Mass'][dat['gmm_pred'] == "disk"].sum()/mstar

    print("m200: ", round(np.log10(m200), 2))
    print("mstar: ", round(np.log10(mstar),2))
    print("ecut: ", round(ecut, 2))
    print("disk fraction: ", fd)
    print("IHL fraction: ", fihl)
    print("bulge fraction: ", fb)
    print("nihl: ", dat[dat['gmm_pred'] == "IHL"].shape[0])
    print("-----------------------", flush = True)


def run(dat, min_ncomp=3, max_ncomp=15):

    final_ncomp = 12 # model to allocate particles

    # classify as disk or spheroid
    sph = morph_class(dat)

    # run gmms varying n
    if sph == False:
        dat, allocs_arr, ecut = fit_disk_gmms(dat, min_ncomp, max_ncomp, final_ncomp)

    else:
        dat, allocs_arr, ecut = fit_sph_gmms(dat, min_ncomp, max_ncomp, final_ncomp)

    dat = allocate_particles(dat, final_ncomp)    

    # save mass in comps for various n for this galaxy
    summary = summarise_models(dat, min_ncomp, max_ncomp, ecut)
    log_fractions(dat, ecut)

    return dat, summary

