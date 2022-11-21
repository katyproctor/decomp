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
    model_dat = iter_dat[['jz/jcirc', 'ebindrel']]

    for n in range(min_ncomp, max_ncomp+1):
        model = GaussianMixture(n_components=n, n_init = 5,
        covariance_type='full', random_state=0).fit(model_dat)

        models = np.append(model, models)
        preds = model.predict(model_dat)

        # save predicted cluster
        col_name = "clus_" + str(n)
        iter_dat[col_name] = preds

        # classify Gaussian clusters as disk, bulge or IHL
        iter_dat["comp_" + str(n)], allocs = classify_clusters.simultaneous_allocation(iter_dat[col_name], model)
        allocs_arr.append(allocs)

        # if no significant disk component at n=4, classify galaxy as spheroid
        jzjc_means = model.means_.T[0]
        if n==min_ncomp+1 and all(jzjc_means < 0.5):
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
        iter_dat["comp_" + str(n)], allocs = classify_clusters.sph_allocation(iter_dat[col_name], model)
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
    print("-----------------------", flush = True)
    return fd, fb, fihl, mstar, m200


def calc_comp_properties(disk, bulge, ihl):

    if disk is None or disk['Mass'].sum() == 0:
        krot_d, kco_d = np.NaN, np.NaN
        r50_d = np.NaN
    else:
        krot_d, kco_d = properties.calc_kappa_co(disk, disk['rad'].max())
        r50_d = properties.calc_rx(disk, 0.5)

    if ihl is None or ihl['Mass'].sum() == 0:
        krot_i, kco_i = np.NaN, np.NaN
        r50_i = np.NaN
    else:
        krot_i, kco_i = properties.calc_kappa_co(ihl, ihl['rad'].max())
        r50_i = properties.calc_rx(ihl, 0.5)

    krot_b, kco_b = properties.calc_kappa_co(bulge, bulge['rad'].max())
    r50_b  = properties.calc_rx(bulge, 0.5)
    
    return krot_d, kco_d, krot_b, kco_b, krot_i, kco_i, r50_d, r50_b, r50_i



def run(dat, plot_folder, gpn):

        # number of Gaussian comps to run GMM with
        min_ncomp = 3
        max_ncomp = 19

        if (max_ncomp - min_ncomp + 1)%2 == 0:
            raise Exception("Total number of models must be odd")

        # run gmms
        dat, models, allocs_arr, sph = fit_gmms(dat, min_ncomp, max_ncomp)

        # if galaxy is a spheroid, only classify stars as bulge/IHL
        if sph == True:
            min_ncomp = min_ncomp - 1
            max_ncomp = max_ncomp - 1
            dat, models, allocs_arr = fit_sph_gmms(dat, min_ncomp, max_ncomp)

        # calculate mass in each component for each model
        m_disk, m_bulge, m_ihl = calc_mass_comps(dat, min_ncomp, max_ncomp, sph)
        
        # choose final allocation model
        disk, bulge, ihl, disk_mad, bulge_mad, ihl_mad, ihl_bool = classify_clusters.classify_comps(dat, min_ncomp,
                                                        m_disk, m_bulge, m_ihl, sph)
        mtot = dat['Mass'].sum()

        # plot how mass varies based on ncomp
        plots.plot_classification(min_ncomp, max_ncomp, models,
                                m_disk, m_bulge, m_ihl,
                                disk, bulge, ihl, mtot,
                                  allocs_arr, sph, ihl_bool,
                                    plot_folder, str(gpn))

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

        # save classifications and component properties
        dat['gmm_pred'] = np.where(dat['ParticleIDs'].isin(disk_ids), "disk",
                            np.where(dat['ParticleIDs'].isin(bulge_ids), "bulge",
                                np.where(dat['ParticleIDs'].isin(ihl_ids), "IHL", "check")))

        fd, fb, fihl, mstar, m200 = log_fractions(dat)
        krot_d, kco_d, krot_b, kco_b, krot_i, kco_i, r50_d, r50_b, r50_i = calc_comp_properties(disk, bulge, ihl)

        # plot projections
        plots.plot_proj(dat, plot_folder, str(gpn))

        # model details
        summary = [[gpn, fd, fb, fihl, mstar, m200,
                  disk_mad, bulge_mad, ihl_mad,
                  krot_d, krot_b, krot_i,
                  kco_d, kco_b, kco_i, r50_d, r50_b, r50_i]]

        col_names = ["GroupNumber", "fdisk", "fbulge", "fihl", "mstar", "m200",
                    "disk_mad", "bulge_mad", "ihl_mad",
                    "krot_disk", "krot_bulge", "krot_ihl",
                    "kco_disk", "kco_bulge", "kco_ihl",
                    "r50_disk", "r50_bulge", "r50_ihl"]
        summary = pd.DataFrame(summary, columns = col_names)

        return dat, summary
