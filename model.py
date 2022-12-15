import pandas as pd
import numpy as np
import glob
import argparse
from sklearn.mixture import GaussianMixture

import classify_clusters
import properties
import plots


def fit_gmms(iter_dat, min_ncomp, max_ncomp):
    
    models = np.array([])
    allocs_arr = []
    model_dat = iter_dat[['jz/jcirc', 'ebindrel', 'jp/jcirc']]
    
    for n in range(min_ncomp, max_ncomp+1):
        model = GaussianMixture(n_components=n, n_init = 5,
        covariance_type='full', random_state=0).fit(model_dat)

        models = np.append(model, models)
        preds = model.predict(model_dat)

        # save predicted cluster
        col_name = "clus_" + str(n)
        iter_dat[col_name] = preds

        # classify Gaussian clusters as disk, bulge or IHL
        iter_dat["comp_" + str(n)], allocs = classify_clusters.simultaneous_allocation(iter_dat, col_name, model)
        allocs_arr.append(allocs)

        # if no significant disk component at n=3, classify galaxy as spheroid
        jzjc_means = model.means_.T[0]
        if n==min_ncomp and all(jzjc_means < 0.5):
            print("Galaxy is a spheroid")
            sph = True
            break
        else:
            sph = False

    # models are saved in reverse order
    models = np.flip(models)

    return [iter_dat, models, allocs_arr, sph]


def fit_sph_gmms(iter_dat, min_ncomp, max_ncomp):

    models = np.array([])
    allocs_arr = []
    model_dat = iter_dat[['jz/jcirc', 'ebindrel']]

    for n in range(min_ncomp, max_ncomp+1):
        # priors
        spacing = 1/(n - 1)
        ebind_guess = [i*spacing for i in range(n)]
        jzjc_guess = [0]*n
        priors = np.array([jzjc_guess, ebind_guess]).T

        model = GaussianMixture(n_components=n, n_init = 5,
        covariance_type='full', means_init = priors, random_state=0).fit(model_dat)

        models = np.append(model, models)
        preds = model.predict(model_dat)

        col_name = "clus_" + str(n)
        iter_dat[col_name] = preds
        iter_dat["comp_" + str(n)], allocs  = classify_clusters.sph_allocation(iter_dat, col_name, model)
        allocs_arr.append(allocs)

    # models are saved in reverse order
    models = np.flip(models)

    return [iter_dat, models, allocs_arr]


def calc_mass_comps(iter_dat, min_ncomp, max_ncomp, sph):

    m_bulge = np.array([])
    m_ihl = np.array([])
    m_disk = np.array([])
    ncomps = np.arange(min_ncomp,max_ncomp+1)

    for n in ncomps:
        cname = "comp_" + str(n)

        # disk mass
        if sph == False:
            tmp_disk = iter_dat['Mass'][iter_dat[cname] == "disk"].sum()

        else:
            tmp_disk = 0

        tmp_bulge = iter_dat['Mass'][iter_dat[cname] == "bulge"].sum()
        tmp_ihl = iter_dat['Mass'][iter_dat[cname] == "IHL"].sum()

        m_disk = np.append(m_disk, tmp_disk)
        m_bulge = np.append(m_bulge, tmp_bulge)
        m_ihl = np.append(m_ihl, tmp_ihl)

    return m_disk, m_bulge, m_ihl


def log_fractions(dat):
    '''
    Log mass fraction in each component
    '''
    print("-----------------------")
    mstar = dat['Mass'].sum()
    m200 = dat['m200'].unique()[0]
    fihl = dat['Mass'][dat['gmm_pred'] == "IHL"].sum()/mstar
    fb = dat['Mass'][dat['gmm_pred'] == "bulge"].sum()/mstar
    fd = dat['Mass'][dat['gmm_pred'] == "disk"].sum()/mstar

    print("disk fraction: ", fd)
    print("IHL fraction: ", fihl)
    print("bulge fraction: ", fb)
    print("nihl: ", dat[dat['gmm_pred'] == "IHL"].shape[0])
    print("-----------------------", flush = True)
   
    return fd, fb, fihl, mstar, m200


def run(dat, plot_folder, gpn):

        # number of Gaussian comps to run GMM with
        min_ncomp = 3
        max_ncomp = 17
     
        if (max_ncomp - min_ncomp + 1)%2 == 0:
            raise Exception("Total number of models must be odd")

        # run gmm
        dat, models, allocs_arr, sph  = fit_gmms(dat, min_ncomp, max_ncomp)

        # if galaxy is a spheroid, run spheroid variation of gmm
        if sph == True:
            min_ncomp = min_ncomp - 1
            max_ncomp = max_ncomp - 1
            dat, models, allocs_arr  = fit_sph_gmms(dat, min_ncomp, max_ncomp)

        # calculate mass in each component for each model
        m_disk, m_bulge, m_ihl = calc_mass_comps(dat, min_ncomp, max_ncomp, sph)
        
        # choose final allocation model
        disk, bulge, ihl, disk_mad, bulge_mad, ihl_mad, comp_no = classify_clusters.select_model(dat, min_ncomp, sph,
                                                                                                 m_disk, m_bulge, m_ihl)

	# assign each particle a component
        if ihl is not None:
            ihl_ids = ihl['ParticleIDs']
        else:
            ihl_ids = []

        if bulge is not None:
            bulge_ids = bulge['ParticleIDs']
        else:
            bulge_ids = []

        if disk is not None:
           disk_ids = disk['ParticleIDs']
        else:
            disk_ids = []

        # save classifications and plot components
        dat['gmm_pred'] = np.where(dat['ParticleIDs'].isin(disk_ids), "disk",
                            np.where(dat['ParticleIDs'].isin(bulge_ids), "bulge",
                                np.where(dat['ParticleIDs'].isin(ihl_ids), "IHL", "check")))
        fd, fb, fihl, mstar, m200 = log_fractions(dat)

        # plot how mass varies based on ncomp
        plots.plot_classification(dat, min_ncomp, max_ncomp, models,
                                 m_disk, m_bulge, m_ihl, comp_no,
                                 disk, bulge, ihl, allocs_arr, sph,
                                 plot_folder, str(gpn))
        plots.plot_proj(dat, plot_folder, str(gpn))

        return dat, disk_mad, bulge_mad, ihl_mad, comp_no
