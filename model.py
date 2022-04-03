import pandas as pd
import numpy as np
import glob
import argparse

from sklearn.mixture import GaussianMixture

import classify_clusters
import plots

### This code uses GMMs to decompose galaxies iteratively by:
# - identifying disk stars via three-component GMM fit
# - separating out bulge stars from ihl


def remove_disk(dat):
    '''Takes stellar data as input, returns: disk_data, remaining_data, n_comp_disk'''
    iter_dat = dat.copy()
    removed_dat = []

    for i in np.arange(1,5):
        print("iteration: ", i)
        model_dat = iter_dat[['jz/jcirc', 'ebindrel', 'jp/jcirc',]]
        model = GaussianMixture(n_components=3, n_init = 3,
        covariance_type='full', random_state=0).fit(model_dat)

        # predicted clusters
        preds = model.predict(model_dat)
        iter_dat['comp'] = preds

        # identify likely disk stars by mean jz/jcirc value
        disk_ind = classify_clusters.get_disk_comp(model)

        if disk_ind is not None:
            # remove disk stars
            removed_dat.append(iter_dat[iter_dat['comp'] == disk_ind].copy())
            iter_dat = iter_dat.loc[iter_dat['comp'] != disk_ind,:].copy()
            print("disk cluster(s) removed \n")

        else:
            print("no disk detected")
            removed_dat.append(pd.Series([], dtype = "float64"))
            break

    # concatenate all disk stars
    removed_dat = pd.concat(removed_dat)
    disk = dat[dat.index.isin(removed_dat.index)].copy()

    return disk, iter_dat, i


def fit_bulge_gmms(iter_dat, max_ncomp):

    models = np.array([])
    allocs_arr = []
    # add predicted cluster to original data
    model_dat = iter_dat[['jz/jcirc', 'ebindrel', 'jp/jcirc', 'vx', 'vy', 'vz']]

    for n in range(2, max_ncomp+1):
        # set priors - equally spaced means for ebind
        # vcomps mean = 0
        spacing = 1/(n - 1)
        ebind_guess = [i*spacing for i in range(n)]

        vx_guess = [0]*n
        vy_guess = [0]*n
        vz_guess = [0]*n

        jzjc_guess = [0]*n
        jpjc_guess = [0.5]*n

        priors = np.array([jzjc_guess, ebind_guess, jpjc_guess, vx_guess, vy_guess, vz_guess]).T

        # save model info
        model = GaussianMixture(n_components=n, n_init = 3,
        covariance_type='full', means_init = priors, random_state=0).fit(model_dat)

        models = np.append(model, models)
        preds = model.predict(model_dat)

        col_name = "clus_" + str(n)
        iter_dat[col_name] = preds
        iter_dat["comp_" + str(n)], allocs = classify_clusters.blind_ihl_allocation(iter_dat[col_name], model)
        allocs_arr.append(allocs)

    # models are saved in reverse order
    models = np.flip(models)

    return [iter_dat, models, allocs_arr]


def calc_mass_comps(iter_dat, max_ncomp):
    m_bulge = np.array([])
    m_ihl = np.array([])

    ncomps = np.arange(2,max_ncomp+1)

    for n in ncomps:
        cname = "comp_" + str(n)
        tmp = iter_dat['Mass'][iter_dat[cname] == "bulge"].sum()
        tmp_ihl = iter_dat['Mass'][iter_dat[cname] == "ihl"].sum()

        m_bulge = np.append(m_bulge, tmp)
        m_ihl = np.append(m_ihl, tmp_ihl)

    return m_bulge, m_ihl, ncomps


def get_features(dat):
    ''' Subset star data to what is used as input to GMM
    '''
    # calculate specific binding energy, relative to minimum  
    e = dat['ParticleBindingEnergy']/dat['Mass']
    dat['ebindrel'] = e/np.min(e)
    
    return dat


def log_fractions(dat):
    '''
    Log mass fraction in each component
    '''
    print("-----------------------")
    mstar = dat['Mass'].sum()
    fihl = dat['Mass'][dat['gmm_pred'] == "IHL"].sum()/mstar
    fb = dat['Mass'][dat['gmm_pred'] == "bulge"].sum()/mstar
    fd = dat['Mass'][dat['gmm_pred'] == "disk"].sum()/mstar

    print("IHL fraction: ", fihl)
    print("disk fraction: ", fd)
    print("bulge fraction: ", fb, flush = True)


def run(dat, plot_folder, gpn):
    
        # model dat 
        dat = get_features(dat)

	# identify disk stars
        disk, remaining, ncomp_disk = remove_disk(dat)

	# classify bulge and IHL
        max_ncomp = 16
        remaining, models, allocs_arr = fit_bulge_gmms(remaining, max_ncomp)
        m_bulge, m_ihl, ncomps = calc_mass_comps(remaining, max_ncomp)

        # plot how mass in IHL and bulge varies based on ncomp
        plots.plot_ihl_classification(max_ncomp, models, m_bulge, m_ihl, allocs_arr,
                        plot_folder, str(gpn))    
 	    
	# choose median value for m_ihl
        bulge, ihl, comp_no = classify_clusters.classify_ihl(remaining, m_bulge, m_ihl)

	# calculate the median absolute deviation from this values for other mass estimates
        dat['ihl_mad'] = np.median(abs(m_ihl - np.median(m_ihl)))

	# assign particles a component
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

        dat['gmm_pred'] = np.where(dat['ParticleIDs'].isin(disk_ids), "disk",
                            np.where(dat['ParticleIDs'].isin(bulge_ids), "bulge",
                                np.where(dat['ParticleIDs'].isin(ihl_ids), "IHL", "check")))

        # print fraction in each component
        log_fractions(dat)

        return dat
